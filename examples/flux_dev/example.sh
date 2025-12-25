GPU_ID=0


# coco

path_to_code=/path_to_LRQ-DiT
FP16_PATH=$path_to_code/LRQ-DiT/examples/flux_dev/logs/data_coco/coco_1024_4steps_0CFG/flux_dev_fp16_image/generated_images
path_to_coco_prompt=$path_to_code/LRQ-DiT/examples/flux_dev/assets/coco_1024.txt
path_to_coco_calib_prompt=$path_to_code/LRQ-DiT/examples/flux_dev/assets/coco_8calib.txt
path_flux_dev=/path_to_flux_pth/black-forest-labs/FLUX.1-dev
w3a4_config=$path_to_code/LRQ-DiT/examples/flux_dev/configs_schnell/w3a4_viditq_alpha050.yaml


CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD1_fp_inference.py  \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_coco/coco_1024_4steps_0CFG/flux_dev_fp16_image \
    --cfg-scale 0.\
    --num-sampling-steps 4 \
    --prompt $path_to_coco_prompt \
    --batch-size 16 \
    --ckpt $path_flux_dev


CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD2_get_calib_data.py \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_coco/coco_1024_4steps_0CFG/calib_pth \
    --cfg-scale 0.\
    --num-sampling-steps 4 \
    --quant-config $w6a6_config \
    --prompt $path_to_coco_calib_prompt \
    --batch-size 8 \
    --ckpt $path_flux_dev



CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD3_quant_inference.py \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_coco/coco_1024_4steps_0CFG/coco_1024_twin_log/w3a4_viduquant \
    --quant-config $w3a4_config \
    --calib-prompt $path_to_coco_calib_prompt \
    --prompt $path_to_coco_prompt \
    --calib-data-path "$path_to_code/LRQ-DiT/examples/flux_dev/logs/data_coco/coco_1024_4steps_0CFG/calib_pth/calib_data.pth" \
    --modelname schnell \
    --ckpt $path_flux_dev \
    --num-sampling-steps 4 \
    --cfg-scale 0 \
    --calib-batch-size 4 \
    --batch-size 4 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "transformer_blocks.13.ff_context.net.0.proj, transformer_blocks.18.ff_context.net.0.proj, transformer_blocks.18.attn.to_out.0, transformer_blocks.17.attn.to_out.0, transformer_blocks.12.ff_context.net.0.proj, transformer_blocks.16.attn.to_out.0" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard









# MJHQ

path_to_code=/path_to_LRQ-DiT
FP16_PATH=$path_to_code/LRQ-DiT/examples/flux_dev/logs/data_MJHQ/MJHQ_1024_4steps_0CFG/flux_dev_fp16_image/generated_images
path_to_MJHQ_prompt=$path_to_code/LRQ-DiT/examples/flux_dev/assets/MJHQ_1024.txt
path_to_MJHQ_calib_prompt=$path_to_code/LRQ-DiT/examples/flux_dev/assets/MJHQ_8calib.txt
path_flux_dev=/path_to_flux_pth/black-forest-labs/FLUX.1-dev
w3a4_config=$path_to_code/LRQ-DiT/examples/flux_dev/configs_schnell/w3a4_viditq_alpha050.yaml



CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD1_fp_inference.py  \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_MJHQ/MJHQ_1024_4steps_0CFG/flux_dev_fp16_image \
    --cfg-scale 0.\
    --num-sampling-steps 4 \
    --prompt $path_to_MJHQ_prompt \
    --batch-size 16 \
    --ckpt $path_flux_dev


CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD2_get_calib_data.py \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_MJHQ/MJHQ_1024_4steps_0CFG/calib_pth \
    --cfg-scale 0.\
    --num-sampling-steps 4 \
    --quant-config $w6a6_config \
    --prompt $path_to_MJHQ_calib_prompt \
    --batch-size 8 \
    --ckpt $path_flux_dev



CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD3_quant_inference.py \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_MJHQ/MJHQ_1024_4steps_0CFG/MJHQ_1024_twin_log/w3a4_viduquant \
    --quant-config $w3a4_config \
    --calib-prompt $path_to_MJHQ_calib_prompt \
    --prompt $path_to_MJHQ_prompt \
    --calib-data-path "$path_to_code/LRQ-DiT/examples/flux_dev/logs/data_MJHQ/MJHQ_1024_4steps_0CFG/calib_pth/calib_data.pth" \
    --modelname schnell \
    --ckpt $path_flux_dev \
    --num-sampling-steps 4 \
    --cfg-scale 0 \
    --calib-batch-size 4 \
    --batch-size 4 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "transformer_blocks.13.ff_context.net.0.proj, transformer_blocks.18.ff_context.net.0.proj, transformer_blocks.18.attn.to_out.0, transformer_blocks.17.attn.to_out.0, transformer_blocks.12.ff_context.net.0.proj, transformer_blocks.16.attn.to_out.0" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard






# DCI

path_to_code=/path_to_LRQ-DiT
FP16_PATH=$path_to_code/LRQ-DiT/examples/flux_dev/logs/data_DCI/DCI_1024_4steps_0CFG/flux_dev_fp16_image/generated_images
path_to_DCI_prompt=$path_to_code/LRQ-DiT/examples/flux_dev/assets/DCI_1024.txt
path_to_DCI_calib_prompt=$path_to_code/LRQ-DiT/examples/flux_dev/assets/DCI_8calib.txt
path_flux_dev=/path_to_flux_pth/black-forest-labs/FLUX.1-dev
w3a4_config=$path_to_code/LRQ-DiT/examples/flux_dev/configs_schnell/w3a4_viditq_alpha050.yaml


CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD1_fp_inference.py  \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_DCI/DCI_1024_4steps_0CFG/flux_dev_fp16_image \
    --cfg-scale 0.\
    --num-sampling-steps 4 \
    --prompt $path_to_DCI_prompt \
    --batch-size 16 \
    --ckpt $path_flux_dev


CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD2_get_calib_data.py \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_DCI/DCI_1024_4steps_0CFG/calib_pth \
    --cfg-scale 0.\
    --num-sampling-steps 4 \
    --quant-config $w6a6_config \
    --prompt $path_to_DCI_calib_prompt \
    --batch-size 8 \
    --ckpt $path_flux_dev


CUDA_VISIBLE_DEVICES=$GPU_ID python $path_to_code/LRQ-DiT/examples/flux_dev/FD3_quant_inference.py \
    --log $path_to_code/LRQ-DiT/examples/flux_dev/logs/data_DCI/DCI_1024_4steps_0CFG/DCI_1024_twin_log/w3a4_viduquant \
    --quant-config $w3a4_config \
    --calib-prompt $path_to_DCI_calib_prompt \
    --prompt $path_to_DCI_prompt \
    --calib-data-path "$path_to_code/LRQ-DiT/examples/flux_dev/logs/data_DCI/DCI_1024_4steps_0CFG/calib_pth/calib_data.pth" \
    --modelname schnell \
    --ckpt $path_flux_dev \
    --num-sampling-steps 4 \
    --cfg-scale 0 \
    --calib-batch-size 4 \
    --batch-size 4 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "transformer_blocks.13.ff_context.net.0.proj, transformer_blocks.18.ff_context.net.0.proj, transformer_blocks.18.attn.to_out.0, transformer_blocks.17.attn.to_out.0, transformer_blocks.12.ff_context.net.0.proj, transformer_blocks.16.attn.to_out.0" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard

