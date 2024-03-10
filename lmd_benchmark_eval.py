import os
import argparse
import torch
from eval.lmd import get_lmd_prompts
from glob import glob
from eval.eval import eval_prompt, Evaluator
from tqdm import tqdm

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_round", type=int, default=1)
    parser.add_argument("--prim_detection_score_threshold", default=0.20, type=float)
    parser.add_argument("--attr_detection_score_threshold", default=0.45, type=float)
    parser.add_argument("--nms_threshold", default=0.15, type=float)
    parser.add_argument("--class-aware-nms", action='store_true', default=True)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--no-cuda", action='store_true')
    args = parser.parse_args()

    prompts = get_lmd_prompts()["lmd"]
    print(f"Number of prompts: {len(prompts)}")

    evaluator = Evaluator()
    eval_success_counts = {}
    eval_all_counts = {}
    failure = []
    
    for ind, prompt in enumerate(tqdm(prompts)):

        get_path = False
        for idx in range(args.num_round, 0, -1):
            path = os.path.join(args.data_dir, f"{ind:03d}", f"round{idx}.jpg")
            if os.path.exists(path):
                get_path = True
                break
        if not get_path:
            path = os.path.join(args.data_dir, f"{ind:03d}", "initial_image.jpg")
        print(f"Image path: {path}")

        eval_type, eval_success = eval_prompt(prompt, path, evaluator, 
                                              prim_score_threshold=args.prim_detection_score_threshold, attr_score_threshold=args.attr_detection_score_threshold, 
                                              nms_threshold=args.nms_threshold, use_class_aware_nms=args.class_aware_nms, use_cuda=True, verbose=args.verbose)

        print(f"Eval success (eval_type):", eval_success)
        if int(eval_success) < 1:
            failure.append(ind)
        if eval_type not in eval_all_counts:
            eval_success_counts[eval_type] = 0
            eval_all_counts[eval_type] = 0
        eval_success_counts[eval_type] += int(eval_success)
        eval_all_counts[eval_type] += 1
    summary = []
    eval_success_conut, eval_all_count = 0, 0
    for k, v in eval_all_counts.items():
        rate = eval_success_counts[k]/eval_all_counts[k]
        print(
            f"Eval type: {k}, success: {eval_success_counts[k]}/{eval_all_counts[k]}, rate: {round(rate, 2):.2f}")
        eval_success_conut += eval_success_counts[k]
        eval_all_count += eval_all_counts[k]
        summary.append(rate)
    print(failure)
    rate = eval_success_conut/eval_all_count
    print(
        f"Overall: success: {eval_success_conut}/{eval_all_count}, rate: {rate:.2f}")
    summary.append(rate)

    summary_str = '/'.join([f"{round(rate, 2):.2f}" for rate in summary])
    print(f"Summary: {summary_str}")