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

from models.customize_pipeline_pixart_sigma import CustomizePixArtSigmaPipeline
from models.customize_pipeline_pixart_transformer_2d import CustomizePixArtTransformer2DModel
diffusers.models.PixArtTransformer2DModel = CustomizePixArtTransformer2DModel
diffusers.PixArtSigmaPipeline = CustomizePixArtSigmaPipeline
from diffusers import PixArtSigmaPipeline
from omegaconf import OmegaConf, ListConfig


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

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    pipe = PixArtSigmaPipeline.from_pretrained(
        "/path_to_pth/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/",
        torch_dtype=torch.float16  
    ).to(device)


    quant_config = OmegaConf.load(args.quant_config)
    pipe.convert_quant(quant_config)
    model = pipe.transformer
    

    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, ViDiTQuantizedLinear)
        module.k = args.k
        massive_keywords = [kw.strip().lower() for kw in args.massive_layers_keywords.split(",")]


        if args.mixed_rotation is True:
            if args.only_massive_layer_rotation is False:
                if 'attn1' in full_name.lower():            
                    module.quant_method = args.attn1_quant_method
                elif 'attn2' in full_name.lower():            
                    module.quant_method = args.attn2_quant_method
                elif 'ff' in full_name.lower():
                    module.quant_method = args.ff_quant_method
                else:
                    module.quant_method = 'NotRotate'
            elif args.only_massive_layer_rotation is True:
                if any(keyword in full_name.lower() for keyword in massive_keywords):
                    module.quant_method = args.massive_layers_quant_method
                else:
                    module.quant_method = args.normal_layers_quant_method
        elif args.mixed_rotation is False:
            module.quant_method = args.quant_method
            
        module.smooth_flag = args.smooth_flag
        module.smooth_alpha = args.smooth_alpha
        if args.smooth_flag is True and module.quant_method == 'duquant':
            act_mask = calib_data[full_name].max(dim=0)[0]  
            module.get_channel_mask(act_mask)  
        elif args.smooth_flag is True and module.quant_method == 'hadamard':
            act_mask = calib_data[full_name].max(dim=0)[0]  
            module.get_channel_mask(act_mask)  
            module.get_rotation_matrix()
        elif args.smooth_flag is True and module.quant_method == 'NotRotate':
            act_mask = calib_data[full_name].max(dim=0)[0]  
            module.get_channel_mask(act_mask)  
        elif args.smooth_flag is False and module.quant_method == 'duquant':
            pass
        elif args.smooth_flag is False and module.quant_method == 'hadamard':
            module.get_rotation_matrix()
        elif args.smooth_flag is False and module.quant_method == 'NotRotate':
            pass



    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(os.path.join(args.calib_data_path), weights_only=True)
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=ViDiTQuantizedLinear,  
                            function=init_rotation_and_channel_mask_,
                            full_name='',
                            calib_data = calib_data,
                            **kwargs
                            )



    calib_prompt_path = args.calib_prompt if args.calib_prompt is not None else "./prompts.txt"
    calib_prompts = []
    with open(calib_prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            calib_prompts.append(line.strip())



    calib_N_batch = len(calib_prompts) // args.calib_batch_size
    for i in range(calib_N_batch):
        images = pipe(
            prompt=calib_prompts[i*args.calib_batch_size: (i+1)*args.calib_batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images

    torch.cuda.empty_cache()  
    del images                


    if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
    if if_mixed_precision:
        model.bitwidth_refactor()
    model.set_init_done()



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
            
        torch.cuda.empty_cache()  
        del images      



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--log", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/coco_1024/calib_pth/debug_image/generated_images")
    parser.add_argument('--quant-config', type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/configs/w3a6_viditq_alpha050.yaml")
    parser.add_argument("--calib-data-path", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/calib_pth/calib_data.pth")
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/coco_1024.txt")
    parser.add_argument("--calib-prompt", type=str, default="/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/coco_1.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", action='store_true', help='whether to use_cuda_kernel')
    parser.add_argument("--profile", action='store_true', help='profile mode, measure the e2e latency')
    parser.add_argument('--mixed_rotation', type=str2bool, default='False', help='mixed rotation flag')
    parser.add_argument('--smooth_flag', type=str2bool, default="True", help='smoothquant flag')
    parser.add_argument('--smooth_alpha', type=float, default=0.5)
    parser.add_argument('--quant_method', type=str, default="duquant", help='quantization method, duquant or hadamard or NotRotate')
    parser.add_argument('--only_massive_layer_rotation', type=str2bool, default='False', help='')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--massive_layers_keywords", type=str, default="attn2.to_q", help="('attn1.to_out.0, attn2.to_q')")
    parser.add_argument("--massive_layers_quant_method", type=str, default="any", help="Quantization method")
    parser.add_argument("--normal_layers_quant_method", type=str, default="hadamard", help="Quantization method")
    args = parser.parse_args()
    main(args)