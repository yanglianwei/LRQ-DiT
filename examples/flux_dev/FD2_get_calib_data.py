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

    pipe = FluxPipeline.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16
    ).to(device)


    pipe.enable_model_cpu_offload()

    
    model = pipe.transformer
    quant_config = OmegaConf.load(args.quant_config)

    kwargs = {
        'hook_cls': SaveActivationHook,
    }
    hook_d = apply_func_to_submodules(model,
                            class_type=nn.Linear, 
                            function=add_hook_to_module_,
                            return_d={},
                            **kwargs
                            )

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

    save_d = {}
    for k,v in hook_d.items():
        save_d[k] = torch.stack(v.outputs, dim=0) 
        logger.info(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')
        v.hook_handle.remove()


    torch.save(save_d, os.path.join(args.log, "calib_data.pth"))
    logger.info(f'saved calib data in {os.path.join(args.log)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/flux_schnell/logs/data_coco/coco_1024_4steps_0CFG/calib_pth")
    parser.add_argument('--quant-config', type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/flux_schnell/configs_schnell/w6a6_viditq_alpha050.yaml")
    parser.add_argument("--cfg-scale", type=float, default=0.)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/flux_schnell/assets/coco_1024.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default="/path_to_LRQ-DiT/flux1_dev_model/FLUX.1-dev")
    args = parser.parse_args()
    main(args)
