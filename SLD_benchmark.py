import os
import json
import copy
import shutil
import random
import numpy as np
import argparse
import configparser
from PIL import Image
import logging

import torch
import diffusers

# Libraries heavily borrowed from LMD
import models
from models import sam
from utils import parse, utils

# SLD specific imports
from sld.detector import OWLVITV2Detector
from sld.sdxl_refine import sdxl_refine
from sld.utils import get_all_latents, run_sam, run_sam_postprocess, resize_image
from sld.llm_template import spot_object_template, spot_difference_template, image_edit_template
from sld.llm_chat import get_key_objects, get_updated_layout
from eval.eval import eval_prompt, Evaluator
from eval.lmd import get_lmd_prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configure logging to include a console handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Function to change the file handler
def set_file_handler(log_file_name):
    logger = logging.getLogger()  # Get the root logger
    for handler in logger.handlers[:]:  # Remove all handlers
        logger.removeHandler(handler)
    logger.addHandler(console_handler)  # Add back the console handler
    file_handler = logging.FileHandler(log_file_name, mode='w')  # Create a file handler for the new log file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)  # Add the new file handler


# Operation #1: Addition (The code is in sld/image_generator.py)

# Operation #2: Deletion (Preprocessing region mask for removal)
def get_remove_region(entry, remove_objects, move_objects, preserve_objs, models, config):
    """Generate a region mask for removal given bounding box info."""

    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape

    # if no remove objects, set zero to the whole mask
    if (len(remove_objects) + len(move_objects)) == 0:
        remove_region = np.zeros((W // 8, H // 8), dtype=np.int64)
        return remove_region

    # Otherwise, run the SAM segmentation to locate target regions
    remove_items = remove_objects + [x[0] for x in move_objects]
    remove_mask = np.zeros((H, W, 3), dtype=bool)
    for obj in remove_items:
        masks = run_sam(bbox=obj[1], image_source=image_source, models=models)
        remove_mask = remove_mask | masks

    # Preserve the regions that should not be removed
    preserve_mask = np.zeros((H, W, 3), dtype=bool)
    for obj in preserve_objs:
        masks = run_sam(bbox=obj[1], image_source=image_source, models=models)
        preserve_mask = preserve_mask | masks
    # Process the SAM mask by averaging, thresholding, and dilating.
    preserve_region = run_sam_postprocess(preserve_mask, H, W, config)
    remove_region = run_sam_postprocess(remove_mask, H, W, config)
    remove_region = np.logical_and(remove_region, np.logical_not(preserve_region))
    return remove_region


# Operation #3: Repositioning (Preprocessing latent)
def get_repos_info(entry, move_objects, models, config):
    """
    Updates a list of objects to be moved / reshaped, including resizing images and generating masks.
    * Important: Perform image reshaping at the image-level rather than the latent-level.
    * Warning: For simplicity, the object is not positioned to the center of the new region...
    """

    # if no remove objects, set zero to the whole mask
    if not move_objects:
        return move_objects
    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))

    new_move_objects = []
    for item in move_objects:
        new_img, obj = resize_image(image_source, item[0][1], item[1][1])
        old_object_region = run_sam_postprocess(run_sam(obj, new_img, models), H, W, config).astype(np.bool_)
        all_latents, _ = get_all_latents(new_img, models, inv_seed)
        new_move_objects.append(
            [item[0][0], obj, item[1][1], old_object_region, all_latents]
        )

    return new_move_objects


# Operation #4: Attribute Modification (Preprocessing latent)
def get_attrmod_latent(entry, change_attr_objects, models, config):
    """
    Processes objects with changed attributes to generate new latents and the name of the modified objects.

    Parameters:
    entry (dict): A dictionary containing output data.
    change_attr_objects (list): A list of objects with changed attributes.
    models (Model): The models used for processing.
    inv_seed (int): Seed for inverse generation.

    Returns:
    list: A list containing new latents and names of the modified objects.
    """
    if len(change_attr_objects) == 0:
        return []
    from diffusers import StableDiffusionDiffEditPipeline
    from diffusers import DDIMScheduler, DDIMInverseScheduler

    img = Image.open(entry["output"][-1])
    image_source = np.array(img)
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))

    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    new_change_objects = []
    for obj in change_attr_objects:
        # Run diffedit
        old_object_region = run_sam_postprocess(run_sam(obj[1], image_source, models), H, W, config)
        old_object_region = old_object_region.astype(np.bool_)[np.newaxis, ...]

        new_object = obj[0].split(" #")[0]
        base_object = new_object.split(" ")[-1]
        mask_prompt = f"a {base_object}"
        new_prompt = f"a {new_object}"

        image_latents = pipe.invert(
            image=img,
            prompt=mask_prompt,
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
        ).latents
        image = pipe(
            prompt=new_prompt,
            mask_image=old_object_region,
            image_latents=image_latents,
            guidance_scale=float(config.get("SLD", "diffedit_guidance_scale")),
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
            negative_prompt="",
        ).images[0]

        all_latents, _ = get_all_latents(np.array(image), models, inv_seed)
        new_change_objects.append(
            [
                old_object_region[0],
                all_latents,
            ]
        )
    return new_change_objects


def correction(
    entry, add_objects, move_objects,
    remove_region, change_attr_objects,
    models, config
):
    spec = {
        "add_objects": add_objects,
        "move_objects": move_objects,
        "prompt": entry["instructions"],
        "remove_region": remove_region,
        "change_objects": change_attr_objects,
        "all_objects": entry["llm_suggestion"],
        "bg_prompt": entry["bg_prompt"],
        "extra_neg_prompt": entry["neg_prompt"],
    }
    image_source = np.array(Image.open(entry["output"][-1]))
    # Background latent preprocessing
    all_latents, _ = get_all_latents(image_source, models, int(config.get("SLD", "inv_seed")))
    ret_dict = image_generator.run(
        spec,
        fg_seed_start=int(config.get("SLD", "fg_seed")),
        bg_seed=int(config.get("SLD", "bg_seed")),
        bg_all_latents=all_latents,
        frozen_step_ratio=float(config.get("SLD", "frozen_step_ratio")),
    )
    return ret_dict


def run_llm_parser(prompt, config):
    questions = f"User Prompt: {prompt}\nReasoning:\n"
    message = spot_object_template + questions
    results = get_key_objects(message, config)
    return results  # Extracting the object list


def run_llm_controller(prompt, det_results, config, mode="self_correction"):
    questions = (
        f"User Prompt: {prompt}\nCurrent Objects: {det_results}\nReasoning:\n"
    )
    if mode == "self_correction":
        message = spot_difference_template + questions
    else:
        message = image_edit_template + questions
    llm_suggestions = get_updated_layout(message, config)
    return llm_suggestions


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(description="Demo for the SLD pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the initial generation results", required=True)
    parser.add_argument("--config", type=str, default="benchmark_config.ini", help="Path to the config file")
    args = parser.parse_args()

    # Preparation
    save_dir = args.data_dir
    parse.img_dir = os.path.join(save_dir, "tmp_imgs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(parse.img_dir, exist_ok=True)

    config = configparser.ConfigParser()
    config.read(args.config)

    # Load models
    models.sd_key = "gligen/diffusers-generation-text-box"
    models.sd_version = "sdv1.4"
    diffusion_scheduler = None

    models.model_dict = models.load_sd(
        key=models.sd_key,
        use_fp16=False,
        load_inverse_scheduler=True,
        scheduler_cls=diffusers.schedulers.__dict__[diffusion_scheduler]
        if diffusion_scheduler is not None
        else None,
    )
    sam_model_dict = sam.load_sam()
    models.model_dict.update(sam_model_dict)
    from sld import image_generator

    det = OWLVITV2Detector()

    # Prepare the evaluator
    evaluator = Evaluator()
    prompts = get_lmd_prompts()["lmd"]

    for idx, prompt in enumerate(prompts):
        # get the initial generation results

        prompt = prompt.strip().rstrip(".")
        dirname = os.path.join(save_dir, f"{idx:03d}")
        fname = os.path.join(dirname, "initial_image.jpg")
        log_file = os.path.join(dirname, "log.txt")
        set_file_handler(log_file)

        # Check whether we need to do self-correction
        attr_threshold = float(config.get("eval", "attr_detection_threshold")) 
        prim_threshold = float(config.get("eval", "prim_detection_threshold"))
        nms_threshold = float(config.get("eval", "nms_threshold"))

        eval_type, eval_success = eval_prompt(prompt, fname, evaluator, 
                                              prim_score_threshold=prim_threshold, attr_score_threshold=attr_threshold, 
                                              nms_threshold=nms_threshold, use_class_aware_nms=True, use_cuda=True, verbose=False)
        if int(eval_success) >= 1:
            logging.info(f"Image {fname} is already correct!")
            continue
        logging.info("-" * 5 + f" [Self-Correcting {fname}] " + "-" * 5)
        logging.info(f"Target Textual Prompt: {prompt}")

        # Run the self-correction pipeline
        chatgpt_data = {
            'llm_parser': None,
            'llm_controller': []
        }

        data = {}
        data["prompt"] = prompt
        # Reset random seeds
        default_seed = int(config.get("SLD", "default_seed"))
        torch.manual_seed(default_seed)
        np.random.seed(default_seed)
        random.seed(default_seed)

        # Step 1: Spot Objects with LLM
        llm_parsed_prompt, spot_object_raw_response = run_llm_parser(prompt, config)
        chatgpt_data["llm_parser"] = (prompt, spot_object_raw_response)
        entry = {"instructions": prompt, "output": [fname],
                "objects": llm_parsed_prompt["objects"], 
                "bg_prompt": llm_parsed_prompt["bg_prompt"],
                "neg_prompt": llm_parsed_prompt["neg_prompt"]
                }
        logging.info("-" * 5 + f" Parsing Prompts " + "-" * 5)
        logging.info(f"* Objects: {entry['objects']}")
        logging.info(f"* Background: {entry['bg_prompt']}")
        logging.info(f"* Negation: {entry['neg_prompt']}")

        num_round = int(config.get("SLD", "num_rounds", fallback=1))
        for i in range(num_round):
            logging.info(f"Round {i + 1}")
            # Step 2: Run open vocabulary detector
            logging.info("-" * 5 + f" Running Detector " + "-" * 5)
            attr_threshold = float(config.get("SLD", "attr_detection_threshold")) 
            prim_threshold = float(config.get("SLD", "prim_detection_threshold"))
            nms_threshold = float(config.get("SLD", "nms_threshold"))

            det_results = det.run(prompt, entry["objects"], entry["output"][-1],
                                attr_detection_threshold=attr_threshold, 
                                prim_detection_threshold=prim_threshold, 
                                nms_threshold=nms_threshold)

            logging.info("-" * 5 + f" Getting Modification Suggestions " + "-" * 5)

            # Step 3: Spot difference between detected results and initial prompts
            llm_suggestions, spot_difference_raw_response = run_llm_controller(prompt, det_results, config)
            chatgpt_data["llm_controller"].append((prompt, spot_difference_raw_response))
            logging.info(f"* Detection Restuls: {det_results}")
            logging.info(f"* LLM Suggestions: {llm_suggestions}")
            entry["det_results"] = copy.deepcopy(det_results)
            entry["llm_suggestion"] = copy.deepcopy(llm_suggestions)
            # Compare the two layouts to know where to update
            (
                preserve_objs,
                deletion_objs,
                addition_objs,
                repositioning_objs,
                attr_modification_objs,
            ) = det.parse_list(det_results, llm_suggestions)

            logging.info("-" * 5 + f" Editing Operations " + "-" * 5)
            logging.info(f"* Preservation: {preserve_objs}")
            logging.info(f"* Addition: {addition_objs}")
            logging.info(f"* Deletion: {deletion_objs}")
            logging.info(f"* Repositioning: {repositioning_objs}")
            logging.info(f"* Attribute Modification: {attr_modification_objs}")
            total_ops = len(deletion_objs) + len(addition_objs) + len(repositioning_objs) + len(attr_modification_objs)
            # Visualization
            parse.show_boxes(
                gen_boxes=entry["det_results"],
                additional_boxes=entry["llm_suggestion"],
                img=np.array(Image.open(entry["output"][-1])).astype(np.uint8),
                fname=os.path.join(dirname, f"det_result{i+1}.jpg"),
            )
            # Check if there are any changes to apply
            if (total_ops == 0):
                print("-" * 5 + f" Results " + "-" * 5)
                output_fname = os.path.join(dirname, f"round{i+1}.jpg")
                shutil.copy(entry["output"][-1], output_fname)
                logging.info("* No changes to apply!")
                logging.info(f"* Output File: {output_fname}")
                entry["output"].append(output_fname)
                continue

            # Step 4: T2I Ops: Addition / Deletion / Repositioning / Attr. Modification
            logging.info("-" * 5 + f" Image Manipulation " + "-" * 5)

            deletion_region = get_remove_region(
                entry, deletion_objs, repositioning_objs, preserve_objs, models, config
            )
            repositioning_objs = get_repos_info(
                entry, repositioning_objs, models, config
            )
            new_attr_modification_objs = get_attrmod_latent(
                entry, attr_modification_objs, models, config
            )
            
            ret_dict = correction(
                entry, addition_objs, repositioning_objs,
                deletion_region, new_attr_modification_objs, 
                models, config
            )
            # Save an intermediate file without the SDXL refinement
            curr_output_fname = os.path.join(dirname, f"round{i+1}.jpg")
            Image.fromarray(ret_dict.image).save(curr_output_fname)
            logging.info("-" * 5 + f" Results " + "-" * 5)
            logging.info(f"* Output File: {curr_output_fname}")
            entry["output"].append(curr_output_fname)
            utils.free_memory()
            # Evaluate again after self-coorection!
            eval_type, eval_success = eval_prompt(prompt, curr_output_fname, evaluator, 
                                                prim_score_threshold=prim_threshold, attr_score_threshold=attr_threshold, 
                                                nms_threshold=nms_threshold, use_class_aware_nms=True, use_cuda=True, verbose=False)
            if int(eval_success) >= 1:
                logging.info(f"Image {fname} is already correct!")
            else:
                logging.info(f"Image {fname} is still incorrect!")

        with open(os.path.join(dirname, "chatgpt_data.json"), 'w') as f:
            json.dump(chatgpt_data, f)