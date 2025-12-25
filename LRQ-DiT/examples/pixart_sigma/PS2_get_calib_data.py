import torch
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging

from diffusers import PixArtSigmaPipeline
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from omegaconf import OmegaConf
import torch.nn as nn

class SaveActivationHook:

    def __init__(self):
        self.hook_handle = None
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        C = module_in[0].shape[-1]
        data = module_in[0].reshape([-1,C]).abs().max(dim=0)[0]  

        self.outputs.append(data)

    def clear(self):
        self.outputs = []

def add_hook_to_module_(module, hook_cls):
    hook = hook_cls()
    hook.hook_handle = module.register_forward_hook(hook)
    return hook

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    pipe = PixArtSigmaPipeline.from_pretrained(
        "/path_to_pth/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/",
        torch_dtype=torch.float16
    ).to(device)
    
    model = pipe.transformer
    kwargs = {
        'hook_cls': SaveActivationHook,
    }
    hook_d = apply_func_to_submodules(model,
                            class_type=nn.Linear, 
                            function=add_hook_to_module_,
                            return_d={},
                            **kwargs
                            )

    calib_prompt_path = args.calib_prompt if args.calib_prompt is not None else "./prompts.txt"
    calib_prompts = []
    with open(calib_prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            calib_prompts.append(line.strip())

    N_batch = len(calib_prompts) // args.calib_batch_size 
    for i in range(N_batch):
        images = pipe(
            prompt=calib_prompts[i*args.calib_batch_size: (i+1)*args.calib_batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images

    save_d = {}
    for k,v in hook_d.items():
        save_d[k] = torch.stack(v.outputs, dim=0) 
        logger.info(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')
        v.hook_handle.remove()

    torch.save(save_d, os.path.join(args.log, "calib_data.pth"))
    logger.info(f'saved calib data in {os.path.join(args.log)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/coco_1024_20steps_4_5CFG/calib_pth/")
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--calib-prompt", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/coco_8calib.txt")
    parser.add_argument("--calib-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
