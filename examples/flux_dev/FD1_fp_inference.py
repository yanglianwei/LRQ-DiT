import torch
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging

from diffusers import FluxPipeline
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    pipe = FluxPipeline.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16
    ).to(device)

    pipe.enable_model_cpu_offload()

    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    N_batch = len(prompts) // args.batch_size 
    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images
        print(f"Export image of batch {i}")
        save_path = os.path.join(args.log, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"output_{i_image + args.batch_size*i}.jpg"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/flux_schnell/logs/data_coco/coco_1024_4steps_0CFG/flux_schnell_fp16_image/generated_images")
    parser.add_argument("--cfg-scale", type=float, default=0.)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/flux_schnell/assets/coco_1024.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="/path_to_LRQ-DiT/flux1_dev_model/FLUX.1-dev")
    args = parser.parse_args()
    main(args)
