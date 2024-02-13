import os
import cv2
import torch
import numpy as np
from models import sam
from models import pipelines


DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"



def get_all_latents(img_np, models, inv_seed=1):
    generator = torch.cuda.manual_seed(inv_seed)
    cln_latents = pipelines.encode(models.model_dict, img_np, generator)
    # Magic prompt
    # Have tried using the parsed bg prompt from the LLM, but it doesn't work well
    prompt = "A realistic photo of a scene"
    input_embeddings = models.encode_prompts(
        prompts=[prompt],
        tokenizer=models.model_dict.tokenizer,
        text_encoder=models.model_dict.text_encoder,
        negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
        one_uncond_input_only=False,
    )
    # Get all hidden latents
    all_latents = pipelines.invert(
        models.model_dict,
        cln_latents,
        input_embeddings,
        num_inference_steps=50,
        guidance_scale=2.5,
    )
    return all_latents, input_embeddings


def run_sam(bbox, image_source, models):
    H, W, _ = image_source.shape
    box_xyxy = torch.Tensor(
        [
            bbox[0],
            bbox[1],
            bbox[2] + bbox[0],
            bbox[3] + bbox[1],
        ]
    ) * torch.Tensor([W, H, W, H])
    box_xyxy = box_xyxy.unsqueeze(0).unsqueeze(0)
    masks, _ = sam.sam(
        models.model_dict,
        image_source,
        input_boxes=box_xyxy,
        target_mask_shape=(H, W),
    )
    masks = masks[0][0].transpose(1, 2, 0).astype(bool)
    return masks


def run_sam_postprocess(remove_mask, H, W, config):
    remove_mask = np.mean(remove_mask, axis=2)
    remove_mask[remove_mask > 0.05] = 1.0
    k_size = int(config.get("SLD", "SAM_refine_dilate"))
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated_mask = cv2.dilate(
        (remove_mask * 255).astype(np.uint8), kernel, iterations=1
    )
    # Resize the mask from the image size to the latent size
    remove_region = cv2.resize(
        dilated_mask.astype(np.int64),
        dsize=(W // 8, H // 8),
        interpolation=cv2.INTER_NEAREST,
    )
    return remove_region


def calculate_scale_ratio(region_a_param, region_b_param):
    _, _, a_width, a_height = region_a_param
    _, _, b_width, b_height = region_b_param
    scale_ratio_width = b_width / a_width
    scale_ratio_height = b_height / a_height
    return min(scale_ratio_width, scale_ratio_height)


def resize_image(image, region_a_param, region_b_param):
    """
    Resizes the image based on the scaling ratio between two regions and performs cropping or padding.
    """
    old_h, old_w, _ = image.shape
    scale_ratio = calculate_scale_ratio(region_a_param, region_b_param)

    new_size = (int(old_w * scale_ratio), int(old_h * scale_ratio))

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    new_h, new_w, _ = resized_image.shape
    region_a_param_real = [
        int(region_a_param[0] * new_h),
        int(region_a_param[1] * new_w),
        int(region_a_param[2] * new_h),
        int(region_a_param[3] * new_w),
    ]
    if scale_ratio >= 1:  # Cropping
        new_xmin = min(region_a_param_real[0], int(new_h - old_h))
        new_ymin = min(region_a_param_real[1], int(new_w - old_w))

        new_img = resized_image[
            new_ymin : new_ymin + old_w, new_xmin : new_xmin + old_h
        ]

        new_param = [
            (region_a_param_real[0] - new_xmin) / old_h,
            (region_a_param_real[1] - new_ymin) / old_w,
            region_a_param[2] * scale_ratio,
            region_a_param[3] * scale_ratio,
        ]
    else:  # Padding
        new_img = np.ones((old_h, old_w, 3), dtype=np.uint8) * 255
        new_img[:new_h, :new_w] = resized_image
        new_param = [region_a_param[i] * scale_ratio for i in range(4)]

    return new_img, new_param


def nms(
    bounding_boxes,
    confidence_score,
    labels,
    threshold,
    input_in_pixels=False,
    return_array=True,
):
    """
    This NMS processes boxes of all labels. It not only removes the box with the same label.

    Adapted from https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []

    # Compute areas of bounding boxes
    if input_in_pixels:
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    else:
        areas = (end_x - start_x) * (end_y - start_y)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_labels.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        if input_in_pixels:
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
        else:
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    if return_array:
        picked_boxes, picked_score, picked_labels = (
            np.array(picked_boxes),
            np.array(picked_score),
            np.array(picked_labels),
        )

    return picked_boxes, picked_score, picked_labels


def post_process(box):
    new_box = []
    for item in box:
        item = min(1.0, max(0.0, item))
        new_box.append(round(item, 3))
    return new_box