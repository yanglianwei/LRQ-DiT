import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import BaseQuantizer, StaticQuantizer, DynamicQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules

from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear
from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear

import logging
logger = logging.getLogger(__name__)

def quant_layer_refactor_(submodule,name,parent_module,quant_config,full_name,remain_fp_regex):

    quant_layer_type = QuantizedLinear
    '''
    METHOD: smooth_quant
    '''
    if quant_config.get("smooth_quant",None) is not None:
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        import re
        layer_regex = quant_config.smooth_quant.layer_name_regex
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            quant_layer_type = SQQuantizedLinear
            logger.info('[INFO] setting smooth quant for layer {}'.format(full_name))
    '''
    METHOD: quarot
    '''
    if quant_config.get("quarot",None) is not None:
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear

        import re
        layer_regex = quant_config.quarot.layer_name_regex
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            quant_layer_type = QuarotQuantizedLinear
            logger.info('setting quarot for layer {}'.format(full_name))
    '''
    METHOD: viditq - quarot + smooth_quant (both used)
    '''
    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        import re
        layer_regex = quant_config.viditq.layer_name_regex
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            quant_layer_type = ViDiTQuantizedLinear
            logger.info('setting viditq for layer {}'.format(full_name))
        

    if remain_fp_regex is not None:
        import re
        pattern = re.compile(remain_fp_regex)
        if pattern.search(full_name):
            logger.info(f"remain {full_name} quant as FP due to fp_regex")
            return

    in_features=submodule.in_features
    out_features=submodule.out_features
    bias  = True if submodule.bias is not None else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(parent_module, name, quant_layer_type(in_features,out_features,bias,device,quant_config,submodule))

    setattr(getattr(parent_module, name), 'module_name', full_name)
    if getattr(parent_module, name).w_quantizer is not None:
        setattr(getattr(parent_module, name).w_quantizer, 'module_name', full_name)
    if getattr(parent_module, name).a_quantizer is not None:
        setattr(getattr(parent_module, name).a_quantizer, 'module_name', full_name)

def bitwidth_refactor_(submodule, name, parent_module, quant_config, full_name):
    import re
    layer_regex_list_w = quant_config.mixed_precision.weight.layer_name_regex
    layer_regex_list_a = quant_config.mixed_precision.act.layer_name_regex

    for idx, layer_regex in enumerate(layer_regex_list_w):
        if len(layer_regex) == 0: 
            continue
        else:
            match = re.search(re.compile(layer_regex), full_name)
            if match:
                if idx == 0: 
                    submodule.quant_mode = False
                    logger.info(f'[Mixed Precision] set the {full_name} W as FP16')
                else:
                    submodule.w_quantizer.bitwidth_refactor(idx-1)
                    logger.info(f'[Mixed Precision] set the {full_name} W as {submodule.w_quantizer.bitwidth_list[idx-1]} bit')

    for idx, layer_regex in enumerate(layer_regex_list_a):
        if len(layer_regex) == 0: 
            continue
        else:
            match = re.search(re.compile(layer_regex), full_name)
            if match:
                if idx == 0: 
                    submodule.quant_mode = False
                    logger.info(f'[Mixed Precision] set the {full_name} A as FP16')
                else:
                    submodule.a_quantizer.bitwidth_refactor(idx-1)
                    logger.info(f'[Mixed Precision] set the {full_name} A as {submodule.a_quantizer.bitwidth_list[idx-1]} bit')
        
def load_quant_param_dict_(submodule, full_name, parent_module, quant_param_dict, model):
    submodule.delta = quant_param_dict[full_name]['delta']
    submodule.zero_point = quant_param_dict[full_name]['zero_point']

    if hasattr(parent_module, 'channel_mask') and hasattr(parent_module, 'rotation_matrix'): 
        assert isinstance(parent_module, ViDiTQuantizedLinear)
        parent_module.get_rotation_matrix()     
        parent_module.channel_mask = quant_param_dict[full_name]['channel_mask']
    elif not hasattr(parent_module, 'channel_mask') and hasattr(parent_module, 'rotation_matrix'):  
        assert isinstance(parent_module, QuarotQuantizedLinear)
        parent_module.get_rotation_matrix()   
        parent_module.update_quantized_weight_rotated()
    elif hasattr(parent_module, 'channel_mask') and not hasattr(parent_module, 'rotation_matrix'): 
        assert isinstance(parent_module, SQQuantizedLinear)
        parent_module.channel_mask = quant_param_dict[full_name]['channel_mask']
        parent_module.update_quantized_weight_scaled()
        
    model.quant_param_dict[full_name] = quant_param_dict[full_name]

def save_quant_param_dict_(submodule, full_name, parent_module, model):

    model.quant_param_dict[full_name] = {}
    model.quant_param_dict[full_name]['delta'] = submodule.delta
    model.quant_param_dict[full_name]['zero_point'] = submodule.zero_point

    if hasattr(parent_module, 'channel_mask'):
        model.quant_param_dict[full_name]['channel_mask'] = parent_module.channel_mask
    if hasattr(parent_module, 'rotation_matrix'):
        model.quant_param_dict[full_name]['rotation_matrix'] = None  

def set_init_done_(submodule):
    submodule.init_done = True

class QuantModel(nn.Module):
    def __init__(
        self,
        quant_config: dict,
        **kwargs,
    ) -> None:
        super().__init__() 

        self.q_cfg = quant_config
        self.quant_param_dict = {}

        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        apply_func_to_submodules(self, 
                class_type=nn.Linear,
                function=quant_layer_refactor_)

    def save_quant_params_dict(self):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=save_quant_param_dict_)

    def load_quant_params_dict(self, quant_param_dict):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=load_quant_param_dict_,
                quant_param_dict=quant_param_dict)

    def set_init_done(self):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=set_init_done_,)
    
    def bitwidth_refactor(self):
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=bitwidth_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None
                )

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("should be implemented in subclass.")

if __name__ == '__main__':
    # TODO: 
    pass
