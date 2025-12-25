GPU_ID=0

# coco
CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA1_fp_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_coco/coco_1024_20steps_4_5CFG/pixart_alpha_fp16_image/ \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/coco_1024.txt \
    --batch-size 16 

CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA2_get_calib_data.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_coco/coco_1024_20steps_4_5CFG/calib_pth \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/coco_8calib.txt \
    --calib-batch-size 16


CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA3_quant_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_coco/coco_1024_20steps_4_5CFG/coco_1024_twin_log/w3a4_viduquant \
    --quant-config /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/configs/w3a4_viditq_alpha050.yaml \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/coco_8calib.txt \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/coco_1024.txt \
    --calib-data-path "/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_coco/coco_1024_20steps_4_5CFG/calib_pth/coco_calib_data.pth" \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-batch-size 16 \
    --batch-size 16 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "transformer_blocks.26.attn2.to_q, transformer_blocks.25.attn2.to_q, transformer_blocks.24.attn2.to_q, transformer_blocks.27.attn2.to_q, transformer_blocks.23.attn2.to_q, transformer_blocks.22.attn2.to_q, transformer_blocks.21.attn2.to_q" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard





# MJHQ
CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA1_fp_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_MJHQ/MJHQ_1024_20steps_4_5CFG/pixart_alpha_fp16_image/ \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/MJHQ_1024.txt \
    --batch-size 16 

CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA2_get_calib_data.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_MJHQ/MJHQ_1024_20steps_4_5CFG/calib_pth \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/MJHQ_8calib.txt \
    --calib-batch-size 16


CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA3_quant_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_MJHQ/MJHQ_1024_20steps_4_5CFG/MJHQ_1024_twin_log/w3a4_viduquant \
    --quant-config /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/configs/w3a4_viditq_alpha050.yaml \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/MJHQ_8calib.txt \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/MJHQ_1024.txt \
    --calib-data-path "/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_MJHQ/MJHQ_1024_20steps_4_5CFG/calib_pth/MJHQ_calib_data.pth" \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-batch-size 16 \
    --batch-size 16 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "transformer_blocks.26.attn2.to_q, transformer_blocks.25.attn2.to_q, transformer_blocks.24.attn2.to_q, transformer_blocks.27.attn2.to_q, transformer_blocks.23.attn2.to_q, transformer_blocks.22.attn2.to_q, transformer_blocks.21.attn2.to_q" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard






# DCI
CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA1_fp_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_DCI/DCI_1024_20steps_4_5CFG/pixart_alpha_fp16_image/ \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/DCI_1024.txt \
    --batch-size 16 

CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA2_get_calib_data.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_DCI/DCI_1024_20steps_4_5CFG/calib_pth \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/DCI_8calib.txt \
    --calib-batch-size 16


CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/PA3_quant_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_DCI/DCI_1024_20steps_4_5CFG/DCI_1024_twin_log/w3a4_viduquant \
    --quant-config /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/configs/w3a4_viditq_alpha050.yaml \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/DCI_8calib.txt \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/assets/DCI_1024.txt \
    --calib-data-path "/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_alpha/logs/data_DCI/DCI_1024_20steps_4_5CFG/calib_pth/DCI_calib_data.pth" \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-batch-size 16 \
    --batch-size 16 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "transformer_blocks.26.attn2.to_q, transformer_blocks.25.attn2.to_q, transformer_blocks.24.attn2.to_q, transformer_blocks.27.attn2.to_q, transformer_blocks.23.attn2.to_q, transformer_blocks.22.attn2.to_q, transformer_blocks.21.attn2.to_q" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard



