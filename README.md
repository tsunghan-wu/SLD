# Self-correcting LLM-controlled Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2311.16090-red)](https://arxiv.org/abs/2311.16090)


**Authors**: [Tsung-Han Wu\*](https://tsunghan-wu.github.io/), [Long Lian\*](https://tonylian.com/), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/), [Boyi Li†](https://sites.google.com/site/boyilics/home), [Trevor Darrell†](https://people.eecs.berkeley.edu/~trevor/) at UC Berkeley. 


## :rocket: The Self-correcting LLM-controlled Diffusion (SLD) Framework Highlights:
1. **Self-correction**: Enhances generative models with LLM-integrated detectors for precise text-to-image alignment.
2. **Unified Generation and Editing**: Excels at both image generation and fine-grained editing.
3. **Universal Compatibility**: Works with ANY image generator, like DALL-E 3, requiring no extra training or data.

![](https://self-correcting-llm-diffusion.github.io/main_figure.jpg)

## :wrench: Installation Guide

### System Requirements

- System Setup: Linux with a single A100 GPU (GPUs with more than 24 GB RAM are also compatible). For Mac or Windows, minor adjustments may be necessary.

- Dependency Installation: Create a Python environment named "SLD" and install necessary dependencies:


```bash
conda create -n SLD python=3.9
pip3 install -r requirements.txt 
```

Note: Ensure the versions of transformers and diffusers match the requirements. Versions of `transformers` before 4.35 do not include `owlv2`, and our code is incompatible with some newer versions of diffusers with different API.

## :gear: Demos

Execute the following command to process images from an input directory according to the instruction in the JSON file and save the transformed images to an output directory.

```
CUDA_VISIBLE_DEVICES=X python3 SLD_demo.py \
    --json-file demo/self_correction/data.json \    # demo/image_editing/data.json
    --input-dir demo/self_correction/src_image \    # demo/image_editing/src_image
    --output-dir demo/self_correction/results \     # demo/image_editing/results
    --mode self_correction \                        # image_editing
    --config demo_config.ini
```

1. This script supports both self-correction and image editing modes. Adjust the paths and --mode flag as needed.
2. We use `gligen/diffusers-generation-text-box` (SDv1.4) as the base diffusion model for image manipulation. For enhanced image quality, we incorporate SDXL refinement techniques similar to [LMD](https://github.com/TonyLianLong/LLM-groundedDiffusion).
 

## :briefcase: Applying to Your Own Images

1. Prepare a JSON File: Structure the file as follows, providing necessary information for each image you wish to process:

```
[
    {
        "input_fname": "<input file name without .png extension>",
        "output_dir": "<output directory>",
        "prompt": "<editing instructions or text-to-image prompt>",
        "generator": "<optional; setting this for hyper-parameters selection>",
        "llm_parsed_prompt": null,             // Leave blank for automatic generation
        "llm_layout_suggestions": null         // Leave blank for automatic suggestions
    }    
]

```

Ensure you replace placeholder text with actual values for each parameter. The llm_parsed_prompt and llm_layout_suggestions are optional and can be left as null for LLM automatic generation.

2. Setting the config

- Duplicate the config/demo_config.ini file to a preferred location.
- Update this copied config file with your OpenAI API key and organization details, along with any other necessary hyper-parameter adjustments.
- **For security reasons, avoid uploading your secret key to public repositories or online platforms.**

3. Execute the Script: Run the script similarly to the provided demo, adjusting the command line arguments as needed for your specific configuration and the JSON file you've prepared.

## :question: Frequently Asked Questions (FAQ)

1. **Why are the results for my own image not optimal?**

   *The SLD framework, while training-free and effective for achieving text-to-image alignment—particularly with numeracy, spatial relationships, and attribute binding—may not consistently deliver optimal visual quality. Tailoring hyper-parameters to your specific image can enhance outcomes.*

2. **Why do the images generated differ from those in the paper?**

   *In our demonstrations, we use consistent random seeds and hyper-parameters for simplicity, differing from the iterative optimization process in our paper figure. For optimal results, we recommend fine-tuning critical hyper-parameters, such as the dilation parameter in the SAM refinement process or parameters in DiffEdit, tailored to your specific use case.*

3. **Isn't using SDXL for improved visualization results unfair?**

   *For quantitative comparisons with baselines in our paper (Table 1), we explicitly exclude the SDXL refinement step to maintain fairness. Also, we set the same hyper-parameters across all models. We will cleanup our code upload to the repo soon.*

4. **Can other LLMs replace GPT-4 in your process?**

   *Yes, other LLMs may be used as alternatives. Our tests with GPT-3.5-turbo indicate only minor performance drops. We encourage exploration with other robust open-source tools like [FastChat](https://github.com/lm-sys/FastChat).*

5. **Have more questions or encountered any bugs?**

   *Please use the GitHub issues section for bug reports. For further inquiries, contact Tsung-Han (Patrick) Wu at tsunghan_wu@berkeley.edu.*

## :pray: Acknowledgements

We are grateful for the foundational code provided by [Diffusers](https://huggingface.co/docs/diffusers/index) and [LMD](https://github.com/TonyLianLong/LLM-groundedDiffusion). Utilizing their resources implies agreement to their respective licenses. Our project benefits greatly from these contributions, and we acknowledge their significant impact on our work.

## :dart: Citation
If you use our work or our implementation in this repo, or find them helpful, please consider giving a citation.
```
@article{wu2023self,
  title={Self-correcting LLM-controlled Diffusion Models},
  author={Wu, Tsung-Han and Lian, Long and Gonzalez, Joseph E and Li, Boyi and Darrell, Trevor},
  journal={arXiv preprint arXiv:2311.16090},
  year={2023}
}
```