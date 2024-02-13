import matplotlib.pyplot as plt
import math
import utils
from PIL import Image, ImageDraw
import numpy as np
from . import parse

save_ind = 0


def visualize(image, title, colorbar=False, show_plot=True, **kwargs):
    plt.title(title)
    plt.imshow(image, **kwargs)
    if colorbar:
        plt.colorbar()
    if show_plot:
        plt.show()


def visualize_arrays(
    image_title_pairs,
    colorbar_index=-1,
    show_plot=True,
    figsize=None,
    fname="vis_array.jpg",
    **kwargs,
):
    if figsize is not None:
        plt.figure(figsize=figsize)
    num_subplots = len(image_title_pairs)
    for idx, image_title_pair in enumerate(image_title_pairs):
        plt.subplot(1, num_subplots, idx + 1)
        if isinstance(image_title_pair, (list, tuple)):
            image, title = image_title_pair
        else:
            image, title = image_title_pair, None

        if title is not None:
            plt.title(title)

        plt.imshow(image, **kwargs)
        if idx == colorbar_index:
            plt.colorbar()

    # if show_plot:
    #     plt.show()
    plt.savefig(fname)


def visualize_masked_latents(
    latents_all, masked_latents, timestep_T=False, timestep_0=True
):
    if timestep_T:
        # from T to 0
        latent_idx = 0

        plt.subplot(1, 2, 1)
        plt.title("latents_all (t=T)")
        plt.imshow(
            (
                latents_all[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.subplot(1, 2, 2)
        plt.title("mask latents (t=T)")
        plt.imshow(
            (
                masked_latents[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.show()

    if timestep_0:
        latent_idx = -1
        plt.subplot(1, 2, 1)
        plt.title("latents_all (t=0)")
        plt.imshow(
            (
                latents_all[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.subplot(1, 2, 2)
        plt.title("mask latents (t=0)")
        plt.imshow(
            (
                masked_latents[latent_idx, 0, :3]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(float)
                / 1.5
            ).clip(0.0, 1.0),
            cmap="gray",
        )

        plt.show()


# This function has not been adapted to new `saved_attn`.
def visualize_attn(
    token_map,
    cross_attention_probs_tensors,
    stage_id,
    block_id,
    visualize_step_start=10,
    input_ca_has_condition_only=False,
):
    """
    Visualize cross attention: `stage_id`th downsampling block, mean over all timesteps starting from step start, `block_id`th Transformer block, second item (conditioned), mean over heads, show each token
    cross_attention_probs_tensors:
    One of `cross_attention_probs_down_tensors`, `cross_attention_probs_mid_tensors`, and `cross_attention_probs_up_tensors`
    stage_id: index of downsampling/mid/upsaming block
    block_id: index of the transformer block
    """

    plt.figure(figsize=(20, 8))

    for token_id in range(len(token_map)):
        token = token_map[token_id]
        plt.subplot(1, len(token_map), token_id + 1)
        plt.title(token)
        attn = cross_attention_probs_tensors[stage_id][visualize_step_start:].mean(
            dim=0
        )[block_id]

        if not input_ca_has_condition_only:
            assert (
                attn.shape[0] == 2
            ), f"Expect to have 2 items (uncond and cond), but found {attn.shape[0]} items"
            attn = attn[1]
        else:
            assert (
                attn.shape[0] == 1
            ), f"Expect to have 1 item (cond only), but found {attn.shape[0]} items"
            attn = attn[0]

        attn = attn.mean(dim=0)[:, token_id]
        H = W = int(math.sqrt(attn.shape[0]))
        attn = attn.reshape((H, W))
        plt.imshow(attn.cpu().numpy())

    plt.show()


# This function has not been adapted to new `saved_attn`.
def visualize_across_timesteps(
    token_id,
    cross_attention_probs_tensors,
    stage_id,
    block_id,
    visualize_step_start=10,
    input_ca_has_condition_only=False,
):
    """
    Visualize cross attention for one token, across timesteps: `stage_id`th downsampling block, mean over all timesteps starting from step start, `block_id`th Transformer block, second item (conditioned), mean over heads, show each token
    cross_attention_probs_tensors:
    One of `cross_attention_probs_down_tensors`, `cross_attention_probs_mid_tensors`, and `cross_attention_probs_up_tensors`
    stage_id: index of downsampling/mid/upsaming block
    block_id: index of the transformer block

    `visualize_step_start` is not used. We visualize all timesteps.
    """
    plt.figure(figsize=(50, 8))

    attn_stage = cross_attention_probs_tensors[stage_id]
    num_inference_steps = attn_stage.shape[0]

    for t in range(num_inference_steps):
        plt.subplot(1, num_inference_steps, t + 1)
        plt.title(f"t: {t}")

        attn = attn_stage[t][block_id]

        if not input_ca_has_condition_only:
            assert (
                attn.shape[0] == 2
            ), f"Expect to have 2 items (uncond and cond), but found {attn.shape[0]} items"
            attn = attn[1]
        else:
            assert (
                attn.shape[0] == 1
            ), f"Expect to have 1 item (cond only), but found {attn.shape[0]} items"
            attn = attn[0]

        attn = attn.mean(dim=0)[:, token_id]
        H = W = int(math.sqrt(attn.shape[0]))
        attn = attn.reshape((H, W))
        plt.imshow(attn.cpu().numpy())
        plt.axis("off")
        plt.tight_layout()

    plt.show()


def visualize_bboxes(bboxes, H, W):
    num_boxes = len(bboxes)
    for ind, bbox in enumerate(bboxes):
        plt.subplot(1, num_boxes, ind + 1)
        fg_mask = utils.proportion_to_mask(bbox, H, W)
        plt.title(f"transformed bbox ({ind})")
        plt.imshow(fg_mask.cpu().numpy())
    plt.show()


def reset_save_ind():
    global save_ind
    save_ind = 0


def display(image, save_prefix="", ind=None, save_ind_in_filename=True):
    """
    save_ind_in_filename: This adds a global index to the filename so that two calls to this function will not save to the same file and overwrite the previous image.
    """
    global save_ind
    if save_prefix != "":
        save_prefix = save_prefix + "_"
    if save_ind_in_filename:
        ind = f"{ind}_" if ind is not None else ""
        path = f"{parse.img_dir}/{save_prefix}{ind}{save_ind}.png"
    else:
        ind = f"{ind}" if ind is not None else ""
        path = f"{parse.img_dir}/{save_prefix}{ind}.png"

    print(f"Saved to {path}")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image.save(path)
    save_ind = save_ind + 1


def draw_bounding_boxes(entry, vis_fname):
    """
    Draw bounding boxes on a PIL image.

    :param image: PIL Image object
    :param bounding_boxes: List of bounding box coordinates in the format (x, y, width, height)
    :param color: Color of the bounding boxes (default is "red")
    :param width: Width of the bounding box lines (default is 2)
    :return: PIL Image object with bounding boxes drawn
    """
    image = Image.open(entry["output"][-1])
    initial_bboxes = entry["det_results"]
    updated_bboxes = entry["llm_suggestion"]
    w, h = image.size
    draw = ImageDraw.Draw(image)
    for bbox in initial_bboxes:
        class_name = bbox[0]
        coords = bbox[1]
        x, y, width, height = coords
        x, y, width, height = int(x * w), int(y * h), int(width * w), int(height * h)
        print(x, y, width, height, class_name)
        draw.rectangle([x, y, x + width, y + height], outline="red", width=2)
        draw.text((x, y), class_name, fill="red")
    # Another image
    blank_image = Image.new("RGB", (w, h), color="white")
    draw_new = ImageDraw.Draw(blank_image)
    for bbox in updated_bboxes:
        class_name = bbox[0]
        coords = bbox[1]
        x, y, width, height = coords
        x, y, width, height = int(x * w), int(y * h), int(width * w), int(height * h)
        print(x, y, width, height, class_name)
        draw_new.rectangle([x, y, x + width, y + height], outline="red", width=2)
        draw_new.text((x, y), class_name, fill="red")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[1].imshow(blank_image)
    # axs[0].axis("off")
    # axs[1].axis("off")
    prompt = entry["instructions"]
    fig.suptitle(f"{prompt}", fontsize=9)
    plt.tight_layout()
    plt.savefig(vis_fname)
    plt.clf()
