import torch
def quantize_and_save_weight_(submodule, full_name):
    fp_weight = submodule.fp_module.weight.to(torch.float16)

    submodule.w_quantizer.delta = submodule.w_quantizer.delta.view(-1).to(torch.float16)
    submodule.w_quantizer.zero_point = submodule.w_quantizer.zero_point.view(-1).to(torch.float16)
    scale = submodule.w_quantizer.delta
    zero_point = submodule.w_quantizer.zero_point  

    int_weight = torch.clamp(
            torch.round(fp_weight / scale.view(-1,1)) - zero_point.view(-1,1),
            -128, 127).to(torch.int8)  
    submodule.weight.data = int_weight