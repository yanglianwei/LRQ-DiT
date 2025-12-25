
# coco
GPU_ID=0 
CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/PS3_quant_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/data_coco/coco_1024_20steps_4_5CFG/coco_1024_twin_log/w3a4_viduquant \
    --quant-config /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/configs/w3a4_viditq_alpha050.yaml \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/coco_8calib.txt \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/coco_1024.txt \
    --calib-data-path "/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/data_coco/coco_1024_20steps_4_5CFG/calib_pth/coco_calib_data.pth" \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-batch-size 16 \
    --batch-size 16 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "attn2.to_q, ff.net.2" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard





# MJHQ

CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/PS3_quant_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/data_MJHQ/MJHQ_1024_20steps_4_5CFG/MJHQ_1024_twin_log/w3a4_viduquant \
    --quant-config /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/configs/w3a4_viditq_alpha050.yaml \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/MJHQ_8calib.txt \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/MJHQ_1024.txt \
    --calib-data-path "/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/data_MJHQ/MJHQ_1024_20steps_4_5CFG/calib_pth/MJHQ_calib_data.pth" \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-batch-size 16 \
    --batch-size 16 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "attn2.to_q, ff.net.2" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard





# DCI

CUDA_VISIBLE_DEVICES=$GPU_ID python /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/PS3_quant_inference.py \
    --log /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/data_DCI/DCI_1024_20steps_4_5CFG/DCI_1024_twin_log/w3a4_viduquant \
    --quant-config /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/configs/w3a4_viditq_alpha050.yaml \
    --calib-prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/DCI_8calib.txt \
    --prompt /path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/assets/DCI_1024.txt \
    --calib-data-path "/path_to_LRQ-DiT/LRQ-DiT/examples/pixart_sigma/logs/data_DCI/DCI_1024_20steps_4_5CFG/calib_pth/DCI_calib_data.pth" \
    --num-sampling-steps 20 \
    --cfg-scale 4.5 \
    --calib-batch-size 16 \
    --batch-size 16 \
    --smooth_flag True \
    --smooth_alpha 0.5 \
    --mixed_rotation True \
    --quant_method any \
    --only_massive_layer_rotation True \
    --massive_layers_keywords "attn2.to_q, ff.net.2" \
    --massive_layers_quant_method duquant \
    --normal_layers_quant_method hadamard
