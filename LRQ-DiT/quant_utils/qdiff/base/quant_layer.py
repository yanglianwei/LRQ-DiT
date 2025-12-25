import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer
from qdiff.base.mixed_precision_quantizer import MixedPrecisionStaticQuantizer, MixedPrecisionDynamicQuantizer
from omegaconf import ListConfig

class QuantizedLinear(torch.nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: torch.nn.Linear,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)

        self.fp_module = fp_module
        self.q_cfg = quant_config

        self.w_quantizer = None
        self.a_quantizer = None

        if quant_config.get('weight', None) is not None:
            if isinstance(quant_config.weight.n_bits, ListConfig):  
                self.w_quantizer = MixedPrecisionStaticQuantizer(quant_config['weight'])
            else:
                self.w_quantizer = StaticQuantizer(quant_config['weight'])

            self.weight.data = self.w_quantizer(fp_module.weight)  
            self.w_quantizer.init_done = True
        else:
            self.weight.data = fp_module.weight

        self.fp_weight = self.fp_module.weight
        self.bias = fp_module.bias

        if quant_config.get('act', None) is not None:
            if isinstance(quant_config.act.n_bits, ListConfig):
                self.a_quantizer = MixedPrecisionDynamicQuantizer(quant_config['act'])
            else:
                self.a_quantizer = DynamicQuantizer(quant_config['act'])

        self.use_kernel = False
        self.quant_mode = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        if not self.quant_mode: 
            return self.fp_module(x, *args, **kwargs)
        else:
            B, N_token, C = x.shape
            x = x.reshape([B*N_token,-1])

            if self.a_quantizer is not None:
                x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])

            y = F.linear(x, self.weight.to(x.dtype),  
                self.bias.to(x.dtype) if self.bias is not None else None, 
                *args, **kwargs)

            return y

if __name__ == '__main__':
    dummy_q_config = {
        'weight': {
            'n_bits': 8,
            'group': 'tensor',
            'sym': False
        },
        'act': {
            'n_bits': 8,
            'group': 'tensor',
            'sym': False
        }
    }
    dummy_linear = nn.Linear(8,32, device='cuda')
    dummy_q_linear = QuantizedLinear(
        in_features = dummy_linear.in_features,
        out_features = dummy_linear.out_features,
        bias = dummy_linear.bias is not None,
        device = dummy_linear.weight.device,
        quant_config = dummy_q_config,
        fp_module = dummy_linear,
    )
    dummy_input = torch.rand([4,2048,8], device='cuda')
    output = dummy_q_linear(dummy_input)
