import torch
import numpy as np
from . import utils
from utils import torch_device
import matplotlib.pyplot as plt
from PIL import Image
import models
import torch.nn.functional as F

# from models import pipelines, encode_prompts


def get_unscaled_latents(batch_size, in_channels, height, width, generator, dtype):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    # Obtain with torch.float32 and cast to float16 if needed
    # Directly obtaining latents in float16 will lead to different latents
    latents_base = torch.randn(
        (batch_size, in_channels, height // 8, width // 8),
        generator=generator,
        dtype=dtype,
    ).to(torch_device, dtype=dtype)

    return latents_base


def get_scaled_latents(
    batch_size, in_channels, height, width, generator, dtype, scheduler
):
    latents_base = get_unscaled_latents(
        batch_size, in_channels, height, width, generator, dtype
    )
    latents_base = latents_base * scheduler.init_noise_sigma
    return latents_base


def blend_latents(latents_bg, latents_fg, fg_mask, fg_blending_ratio=0.01):
    """
    in_channels: often obtained with `unet.config.in_channels`
    """
    assert not torch.allclose(
        latents_bg, latents_fg
    ), "latents_bg should be independent with latents_fg"

    dtype = latents_bg.dtype
    latents = (
        latents_bg * (1.0 - fg_mask)
        + (
            latents_bg * np.sqrt(1.0 - fg_blending_ratio)
            + latents_fg * np.sqrt(fg_blending_ratio)
        )
        * fg_mask
    )
    latents = latents.to(dtype=dtype)

    return latents


@torch.no_grad()
def compose_latents(
    model_dict,
    latents_all_list,
    mask_tensor_list,
    num_inference_steps,
    overall_batch_size,
    height,
    width,
    bg_seed=None,
    compose_box_to_bg=True,
    use_fast_schedule=False,
    fast_after_steps=None,
    latents_bg=None,
):
    unet, scheduler, dtype = model_dict.unet, model_dict.scheduler, model_dict.dtype

    generator = torch.manual_seed(
        bg_seed
    )  # Seed generator to create the inital latent noise
    latents_bg = get_scaled_latents(
        51,
        unet.config.in_channels,
        height,
        width,
        generator,
        dtype,
        scheduler,
    )
    latents_bg = latents_bg.unsqueeze(1)

    # Other than t=T (idx=0), we only have masked latents. This is to prevent accidentally loading from non-masked part. Use same mask as the one used to compose the latents.
    if use_fast_schedule:
        # If we use fast schedule, we only compose the frozen steps because the later steps do not match.
        composed_latents = torch.zeros(
            (fast_after_steps + 1, *latents_bg.shape), dtype=dtype
        )
    else:
        # Otherwise we compose all steps so that we don't need to compose again if we change the frozen steps.
        composed_latents = torch.zeros((latents_bg.shape), dtype=dtype)
        # composed_latents = latents_bg

    foreground_indices = torch.zeros(latents_bg.shape[-2:], dtype=torch.long)

    mask_size = np.array([mask_tensor.sum().item() for mask_tensor in mask_tensor_list])

    # Compose the largest mask first
    mask_order = np.argsort(-mask_size)

    existing_objects = torch.zeros(latents_bg.shape[-2:], dtype=torch.bool)
    # print(len(mask_order))
    # exit()
    for idx, mask_idx in enumerate(mask_order):
        latents_all, mask_tensor = (
            latents_all_list[mask_idx],
            mask_tensor_list[mask_idx],
        )

        mask_tensor_expanded = mask_tensor[None, None, None, ...].repeat(51, 1, 4, 1, 1)
        composed_latents[mask_tensor_expanded == 1] = latents_all[
            mask_tensor_expanded == 1
        ]
        existing_objects |= mask_tensor

    existing_objects_expanded = existing_objects[None, None, None, ...].repeat(
        51, 1, 4, 1, 1
    )
    composed_latents[existing_objects_expanded == 0] = latents_bg.cpu()[
        existing_objects_expanded == 0
    ]

    composed_latents, foreground_indices = composed_latents.to(
        torch_device
    ), existing_objects.to(torch_device)
    return composed_latents, foreground_indices


def align_with_bboxes(
    latents_all_list, mask_tensor_list, bboxes, horizontal_shift_only=False
):
    """
    Each offset in `offset_list` is `(x_offset, y_offset)` (normalized).
    """
    new_latents_all_list, new_mask_tensor_list, offset_list = [], [], []
    for latents_all, mask_tensor, bbox in zip(
        latents_all_list, mask_tensor_list, bboxes
    ):
        x_src_center, y_src_center = utils.binary_mask_to_center(
            mask_tensor, normalize=True
        )
        x_min_dest, y_min_dest, x_max_dest, y_max_dest = bbox
        x_dest_center, y_dest_center = (x_min_dest + x_max_dest) / 2, (
            y_min_dest + y_max_dest
        ) / 2
        # print("src (x,y):", x_src_center, y_src_center, "dest (x,y):", x_dest_center, y_dest_center)
        x_offset, y_offset = x_dest_center - x_src_center, y_dest_center - y_src_center
        if horizontal_shift_only:
            y_offset = 0.0
        offset = x_offset, y_offset
        latents_all = utils.shift_tensor(
            latents_all, x_offset, y_offset, offset_normalized=True
        )
        mask_tensor = utils.shift_tensor(
            mask_tensor, x_offset, y_offset, offset_normalized=True
        )
        new_latents_all_list.append(latents_all)
        new_mask_tensor_list.append(mask_tensor)
        offset_list.append(offset)

    return new_latents_all_list, new_mask_tensor_list, offset_list


def coord_transform(coords, width):
    x_min, y_min, h, w = coords
    x_max = x_min + h
    y_max = y_min + w
    new_coords = (
        int(x_min * width),
        int(x_max * width),
        int(y_min * width),
        int(y_max * width),
    )
    return new_coords


def inverse_warp(A, roi_A, B, roi_B_target, seg_map):
    """
    Perform an inverse warping of a region of interest from matrix A to matrix B.

    Parameters:
    - A: Source Pytorch matrix.
    - roi_A: Region of interest in A as [x_min, x_max, y_min, y_max].
    - B: Target Pytorch matrix.
    - roi_B_target: Specified rectangle parameters in B as [x_min_target, x_max_target, y_min_target, y_max_target].

    Returns:
    - B with the warped region from A.
    """
    # TODO: Not sure if this function is correct or not under edge cases...
    x_min, x_max, y_min, y_max = roi_A
    x_min_target, x_max_target, y_min_target, y_max_target = roi_B_target
    x_max -= max(0, (x_max_target - 63))
    y_max -= max(0, (y_max_target - 63))

    # Extract the region of interest from A
    A = A.squeeze(1)
    B = B.squeeze(1)

    seg_map = (
        torch.from_numpy(seg_map)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(A.shape[0], A.shape[1], 1, 1)
    ).float()

    roi_content = (
        A[:, :, int(y_min) : int(y_max), int(x_min) : int(x_max)]
        * seg_map[:, :, int(y_min) : int(y_max), int(x_min) : int(x_max)]
    )

    seg_roi_content = seg_map[:, :, int(y_min) : int(y_max), int(x_min) : int(x_max)]

    # Place the resized region into B
    roi_shape = roi_content.shape[-2:]
    B[
        :,
        :,
        int(y_min_target) : int(y_min_target) + roi_shape[0],
        int(x_min_target) : int(x_min_target) + roi_shape[1],
    ] = roi_content

    A = A.unsqueeze(1)
    B = B.unsqueeze(1)
    new_mask = torch.zeros((64, 64), dtype=bool)
    seg_roi_shape = seg_roi_content.shape[-2:]
    new_mask[
        int(y_min_target) : int(y_min_target) + seg_roi_shape[0],
        int(x_min_target) : int(x_min_target) + seg_roi_shape[1],
    ] = seg_roi_content[0][0]

    return B, new_mask


def plot_feat(tensor_data, fname):
    import matplotlib.pyplot as plt

    # Convert the tensor to a NumPy array
    numpy_data = (
        tensor_data.squeeze().cpu().numpy()
    )  # Remove dimensions of size 1 and convert to NumPy array

    # Create a figure and a grid of subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Loop through the 4 images and display them in subplots
    for i in range(4):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(
            numpy_data[i], cmap="gray"
        )  # Assuming images are grayscale

    # Set titles for each subplot (optional)
    for i, ax in enumerate(axes.flat):
        ax.set_title(f"Feat {i + 1}")

    # Remove axis labels and ticks (optional)
    for ax in axes.flat:
        ax.axis("off")

    # Adjust spacing between subplots (optional)
    plt.tight_layout()

    # Show the plot
    plt.savefig(fname)
    plt.close()
    # plt.show()


@torch.no_grad()
def compose_latents_with_alignment(
    model_dict,
    latents_bg_lists,
    latents_all_list,
    mask_tensor_list,
    original_remove,
    change_objects,
    move_objects,
    num_inference_steps,
    overall_batch_size,
    height,
    width,
    align_with_overall_bboxes=True,
    overall_bboxes=None,
    horizontal_shift_only=False,
    bg_seed=1,
    **kwargs,
):
    if align_with_overall_bboxes and len(latents_all_list):
        expanded_overall_bboxes = utils.expand_overall_bboxes(overall_bboxes)
        latents_all_list, mask_tensor_list, offset_list = align_with_bboxes(
            latents_all_list,
            mask_tensor_list,
            bboxes=expanded_overall_bboxes,
            horizontal_shift_only=horizontal_shift_only,
        )
    else:
        offset_list = [(0.0, 0.0) for _ in range(len(latents_all_list))]

    # Compose Move Objects
    # import pdb

    # pdb.set_trace()
    # latents_all_list.append(latents_bg_lists)
    # mask_tensor_list.append(bg_mask)

    for obj_name, old_obj, new_obj, seg_map, all_latents in move_objects:
        # print(all_latents.shape)
        # exit()
        # x_min_old, x_max_old, y_min_old, y_max_old
        old_coords = coord_transform(old_obj, 64)
        # x_min_new, x_max_new, y_min_new, y_max_new
        new_coords = coord_transform(new_obj, 64)
        new_latents = all_latents.clone()
        new_latents, new_mask = inverse_warp(
            all_latents, old_coords, new_latents, new_coords, seg_map
        )
        # plot_feat(new_latents[-1], "feat_after.png")
        # plot_feat(all_latents[-1], "feat_before.png")
        # plot_feat(new_latents[-1], "feat_after.png")
        # exit()
        # print(new_latents.shape)
        # print(latents_bg_lists.shape)
        # print(new_mask.shape)
        # import pdb

        # pdb.set_trace()
        # new_latents[
        #     :, :, :, y_min_new:y_max_new, x_min_new:x_max_new
        # ] = latents_bg_lists[:, :, :, y_min_old:y_max_old, x_min_old:x_max_old]
        # new_mask = torch.zeros((64, 64), dtype=bool)
        # new_mask[y_min_new:y_max_new, x_min_new:x_max_new] = True
        # plt.imsave((mew_latents[].cpu().numpy() * 255).astype(np.uint8)).save(
        #     "new_mask.png"
        # )
        # Image.fromarray((new_mask.cpu().numpy() * 255).astype(np.uint8)).save(
        #     "new_mask.png"
        # )
        # old_mask = torch.zeros((64, 64), dtype=bool)
        # old_mask[y_min_old:y_max_old, x_min_old:x_max_old] = True
        # Image.fromarray((old_mask.cpu().numpy() * 255).astype(np.uint8)).save(
        #     "old_mask.png"
        # )
        # exit()
        latents_all_list.append(new_latents)
        mask_tensor_list.append(new_mask)
        # np.save()
        # break
    # N = len(mask_tensor_list)
    # for i in range(N):
    #     np.save(f"object_latent_{i:02d}.npy", latents_all_list[i].cpu().numpy())
    #     np.save(f"object_mask_{i:02d}.npy", mask_tensor_list[i].cpu().numpy())
    # exit()
    # import pdb

    # pdb.set_trace()
    for mask, latents in change_objects:
        latents_all_list.append(latents)
        mask_tensor_list.append(torch.from_numpy(mask))

    fg_mask_union = torch.zeros((64, 64), dtype=bool)
    N = len(mask_tensor_list)
    for i in range(N):
        fg_mask_union |= mask_tensor_list[i]
    bg_mask = ~fg_mask_union
    bg_mask[original_remove == True] = False
    # Image.fromarray((bg_mask.cpu().numpy() * 255).astype(np.uint8)).save("bg_mask.png")
    # print("bg_mask.png")
    # exit()
    # img_src = np.array(Image.open("sdv2_generation/round_0/30.png"))
    # print("sdv2_generation/round_0/30.png")
    # still_need_remove = original_remove.clone()
    # still_need_remove[fg_mask_union == True] = False

    # exit()
    # bg_mask = ~still_need_remove
    # import numpy as np

    # Image.fromarray((bg_mask * 255).cpu().numpy().astype(np.uint8)).save("bg_mask.png")
    # print("bg_mask.png")
    # for i in range(N):
    #     print(i, flush=True)
    #     Image.fromarray(
    #         (mask_tensor_list[i] * 255).cpu().numpy().astype(np.uint8)
    #     ).save(f"fg{i}_mask.png")
    #     print(f"fg{i}_mask.png")
    # # plot_feat(latents_bg_lists[-1], "feat_bg.png")
    # for i in range(51):
    #     plot_feat(latents_bg_lists[i] * bg_mask, f"vis_feat/BG{i}.png")
    # for j in range(N):
    #     for i in range(51):
    #         plot_feat(
    #             latents_all_list[j][i] * mask_tensor_list[j], f"vis_feat/fg{j}_{i}.png"
    #         )

    # exit()
    # input("OWO")

    # overlap_region = torch.zeros((64, 64), dtype=bool)
    # for i in range(N):
    #     overlap_region |= mask_tensor_list[i]
    # overlap_region |= bg_mask

    latents_all_list.append(latents_bg_lists)
    mask_tensor_list.append(bg_mask)

    composed_latents, foreground_indices = compose_latents(
        model_dict,
        latents_all_list,
        mask_tensor_list,
        num_inference_steps,
        overall_batch_size,
        height,
        width,
        bg_seed,
        **kwargs,
    )
    # composed_latents = latents_all_list[0].cuda()
    # foreground_indices = mask_tensor_list[0].cuda()
    # print(composed_latents.shape)
    # print(foreground_indices.shape)
    # exit()
    # for i in range(51):
    #     plot_feat(composed_latents[i], f"vis_feat/feat_final{i}.png")
    # exit()
    return composed_latents, foreground_indices, offset_list


def get_init_bg(model_dict):
    from models import pipelines

    print("haha here am I!", flush=True)
    init_image = Image.open("check.png")
    generator = torch.cuda.manual_seed(6666)
    cln_latents = pipelines.encode(model_dict, init_image, generator)

    vae, tokenizer, text_encoder, unet, scheduler, dtype = (
        model_dict.vae,
        model_dict.tokenizer,
        model_dict.text_encoder,
        model_dict.unet,
        model_dict.scheduler,
        model_dict.dtype,
    )

    input_embeddings = models.encode_prompts(
        prompts=["A forest"],
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        negative_prompt="",
        one_uncond_input_only=False,
    )
    all_latents = models.pipelines.invert(
        model_dict,
        cln_latents,
        input_embeddings,
        num_inference_steps=50,
        guidance_scale=2.5,
    )
    print(all_latents.shape)
    # all_latents_cpu = all_latents.cpu().numpy()
    # gen_latents = all_latents[0].cuda()
    # gen_latents *= scheduler.init_noise_sigma
    return all_latents


def get_input_latents_list(
    model_dict,
    latents_bg,
    bg_seed,
    fg_seed_start,
    fg_blending_ratio,
    height,
    width,
    so_prompt_phrase_box_list=None,
    so_boxes=None,
    verbose=False,
):
    """
    Note: the returned input latents are scaled by `scheduler.init_noise_sigma`
    """
    unet, scheduler, dtype = model_dict.unet, model_dict.scheduler, model_dict.dtype

    generator_bg = torch.manual_seed(
        bg_seed
    )  # Seed generator to create the inital latent noise

    # latents_bg_lists = get_init_bg(model_dict)
    # latents_bg = latents_bg_lists[1].cuda()
    if latents_bg is None:
        latents_bg = get_unscaled_latents(
            batch_size=1,
            in_channels=unet.config.in_channels,
            height=height,
            width=width,
            generator=generator_bg,
            dtype=dtype,
        )

    input_latents_list = []

    if so_boxes is None:
        # For compatibility
        so_boxes = [item[-1] for item in so_prompt_phrase_box_list]

    # change this changes the foreground initial noise
    for idx, obj_box in enumerate(so_boxes):
        H, W = height // 8, width // 8
        fg_mask = utils.proportion_to_mask(obj_box, H, W)
        # plt.imsave("fg_mask.jpg", fg_mask.cpu().numpy())
        # exit()
        if verbose:
            plt.imshow(fg_mask.cpu().numpy())
            plt.show()

        fg_seed = fg_seed_start + idx
        if fg_seed == bg_seed:
            # We should have different seeds for foreground and background
            fg_seed += 12345

        generator_fg = torch.manual_seed(fg_seed)
        latents_fg = get_unscaled_latents(
            batch_size=1,
            in_channels=unet.config.in_channels,
            height=height,
            width=width,
            generator=generator_fg,
            dtype=dtype,
        )

        input_latents = blend_latents(
            latents_bg, latents_fg, fg_mask, fg_blending_ratio=fg_blending_ratio
        )

        input_latents = input_latents * scheduler.init_noise_sigma

        input_latents_list.append(input_latents)

    latents_bg = latents_bg * scheduler.init_noise_sigma

    return input_latents_list, latents_bg
