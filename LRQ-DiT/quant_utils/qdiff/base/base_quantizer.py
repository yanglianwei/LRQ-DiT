import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time
import math
from omegaconf import ListConfig
import numpy as np

logger = logging.getLogger(__name__)



class BaseQuantizer(nn.Module):

    def __init__(self, quant_config):
        super(BaseQuantizer, self).__init__()
        
        self.n_bits = quant_config['n_bits']
        self.sym = quant_config.get('sym', False)

        if isinstance(self.n_bits, list):
            raise AssertionError("when multiple n_bits are adopted, use the MixedPrecisionBaseQuantizer")

        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)

        if not isinstance(self.n_bits, ListConfig):
            self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1


        self.init_done = False

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("should be implemented in subclass.")
    
    def init_quant_params(self, x):
        raise NotImplementedError("should be implemented in subclass.")


class DynamicQuantizer(BaseQuantizer):

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def quantize(self, x:torch.Tensor):
        assert len(x.shape) == 2  


        assert torch.isnan(x).sum() == 0  
        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = x_absmax
            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
            eps = 1.e-6
            try:
                assert torch.all(delta.abs() > eps)
            except:
                import ipdb; ipdb.set_trace()
                delta[delta < eps] = eps
                logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), self.module_name))
                
        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max<0] = 0. 
            self.x_max = x_max

            x_min = x.min(dim=1)[0]
            x_min[x_min>0] = 0.
            self.x_min = x_min

            delta = (x_max - x_min)/(self.n_levels-1)
            eps = 1.e-8
            try:
                assert torch.all(delta.abs() > eps)
            except:
                import ipdb; ipdb.set_trace()
                
                delta[delta < eps] = eps
                logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), self.module_name))
            zero_point = torch.round(x_min/delta) + (self.n_levels/2)
        self.delta = delta.unsqueeze(-1)  
        self.zero_point = zero_point.unsqueeze(-1)
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant



    def forward(self, x: torch.Tensor):
        x_quant = self.quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant
    


    def forward_with_quant_params(self, x, delta, mixed_precision=None):
        assert self.sym
 
        if mixed_precision is not None:
            n_levels = torch.pow(2,mixed_precision) -  1 
            zero_bit_mask = (n_levels != 0).int()
            n_levels[n_levels == 0] = 255  

        eps = 1.e-6
        try:
            assert torch.all(delta.abs() > eps)
        except:
            delta[delta < eps] = eps

        if mixed_precision is not None:
            delta = delta / n_levels
            x_int = torch.round(x / delta)
            x_quant = torch.where(x_int>n_levels, n_levels, x_int)
        else:
            delta = delta/ (self.n_levels*2+1)
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int, 0, self.n_levels*2+1)

        x_dequant = (x_quant) * delta

        if mixed_precision is not None:  
            x_dequant = x_dequant*zero_bit_mask

        return x_dequant





class StaticQuantizer(BaseQuantizer):
    def __init__(self, quant_config, twinlog=False):
        super().__init__(quant_config)
        self.twinlog = twinlog
        if self.sym:
            self.x_absmax = None
        else:
            self.x_max = None
            self.x_min = None
        self.yita = 0.0
        self.x_pos_delta = None
        self.x_pos_zero_point = None
        self.x_neg_delta = None
        self.x_neg_zero_point = None

    def logk(self, x, k):
        natural_log = torch.log(x)
        log_k = natural_log / torch.log(torch.tensor(k, dtype=x.dtype))
        return log_k

    def get_delta_zero_point(self, x, n_levels):
        x_max = x.max(dim=1)[0]
        x_max = torch.max(x_max.to(x_max.device), x_max) if x_max is not None else x_max
        x_min = x.min(dim=1)[0]
        x_min = torch.min(x_min.to(x_min.device), x_min) if x_min is not None else x_min
        delta = (x_max - x_min)/(n_levels)
        zero_point = torch.round(x_min/delta) 
        delta = delta.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        return delta, zero_point
    
    def get_delta_zero_point_percentile(self, x, n_levels, pct):
        x_clone = x.clone().float() 
        x_max = torch.quantile(x_clone, q=pct, dim=1, keepdim=True, interpolation='nearest')
        x_max = torch.max(x_max.to(x_max.device), x_max) if x_max is not None else x_max
        x_max = x_max.squeeze(1)  
        x_min = x.min(dim=1)[0]
        x_min = torch.min(x_min.to(x_min.device), x_min) if x_min is not None else x_min
        delta = (x_max - x_min)/(n_levels)
        zero_point = torch.round(x_min/delta) 
        delta = delta.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        return delta, zero_point



    def x_pos_quant_dequant(self, x_pos, yita, mask_pos, x_pos_delta, x_pos_zero_point):
        x_pos_int = torch.round(x_pos/x_pos_delta)-(x_pos_zero_point)
        x_pos_quant = torch.clamp(x_pos_int, 0,  2 ** (self.n_bits - 1) -1)
        x_pos_dequant = (x_pos_quant + x_pos_zero_point) * (x_pos_delta) 
        x_pos_float = 2**(x_pos_dequant) - yita
        x_pos_float = x_pos_float * mask_pos
        return x_pos_float, x_pos_delta, x_pos_zero_point
    
    def x_neg_quant_dequant(self, x_neg, yita, mask_neg, x_neg_delta, x_neg_zero_point):
        x_neg_int = torch.round(x_neg/x_neg_delta)-(x_neg_zero_point)
        x_neg_quant = torch.clamp(x_neg_int, 0,  2 ** (self.n_bits - 1))
        x_neg_dequant = (x_neg_quant + x_neg_zero_point) * (x_neg_delta) 
        x_neg_float = -(2**(x_neg_dequant) - yita)
        x_neg_float = x_neg_float * mask_neg
        return x_neg_float, x_neg_delta, x_neg_zero_point
    
    def forward(self, x: torch.Tensor):
        if self.twinlog is False: 
            self.init_quant_params(x)
        mask_pos = (x > 0)  
        mask_neg = (x < 0)  
        x_abs = torch.abs(x)
        log_x_abs = self.logk((x_abs+self.yita), 2) 
        x_pos = torch.zeros_like(x)
        x_pos[mask_pos] = log_x_abs[mask_pos]  
        assert not torch.isinf(x_pos).any(), "Error: Tensor contains Inf!"
        assert not torch.isnan(x_pos).any(), "Error: Tensor contains NaN!"
        x_neg = torch.zeros_like(x)
        x_neg[mask_neg] = log_x_abs[mask_neg]  
        assert not torch.isinf(x_neg).any(), "Error: Tensor contains Inf!" 
        assert not torch.isnan(x_neg).any(), "Error: Tensor contains NaN!"
        x_pos_int = torch.round(x_pos/self.x_pos_delta)-(self.x_pos_zero_point)
        x_pos_quant = torch.clamp(x_pos_int, 0,  2 ** (self.n_bits - 1) -1)
        x_pos_dequant = (x_pos_quant + self.x_pos_zero_point) * (self.x_pos_delta) 
        x_pos_float = 2**(x_pos_dequant) - self.yita
        x_pos_float = x_pos_float * mask_pos
        x_neg_int = torch.round(x_neg/self.x_neg_delta)-(self.x_neg_zero_point)
        x_neg_quant = torch.clamp(x_neg_int, 0,  2 ** (self.n_bits - 1))
        x_neg_dequant = (x_neg_quant + self.x_neg_zero_point) * (self.x_neg_delta) 
        x_neg_float = -(2**(x_neg_dequant) - self.yita)
        x_neg_float = x_neg_float * mask_neg
        x_all_dequant = x_pos_float + x_neg_float
        return x_all_dequant            


    def init_quant_params(self, x):
        assert len(x.shape) == 2  
        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = (torch.max(self.x_absmax, x_absmax) if self.x_absmax is not None else x_absmax).to("cuda") 
            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
            self.delta = delta.unsqueeze(-1) 
            self.zero_point = zero_point.unsqueeze(-1)
            try:
                assert torch.all(delta > 1.e-6), "unexpected small delta exists"
            except:
                import ipdb; ipdb.set_trace()
        else:
            mask_pos = (x > 0)  
            mask_neg = (x < 0)  
            x_abs = torch.abs(x)
            best_error = float('inf')
            best_yita = 0.0
            best_pos_delta = None
            best_pos_zero_point = None
            best_neg_delta = None
            best_neg_zero_point = None
            for yita in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                log_x_abs = self.logk((x_abs+yita), 2) 
                x_pos = torch.zeros_like(x)
                x_pos[mask_pos] = log_x_abs[mask_pos]  
                assert not torch.isinf(x_pos).any(), "Error: Tensor contains Inf!"
                assert not torch.isnan(x_pos).any(), "Error: Tensor contains NaN!"
                x_neg = torch.zeros_like(x)
                x_neg[mask_neg] = log_x_abs[mask_neg]  
                assert not torch.isinf(x_neg).any(), "Error: Tensor contains Inf!"
                assert not torch.isnan(x_neg).any(), "Error: Tensor contains NaN!"
                for pos_pct in [0.98, 0.99, 0.999, 0.9999, 0.99999]:
                    x_pos_delta1, x_pos_zero_point1 = self.get_delta_zero_point_percentile(x_pos, 2 ** (self.n_bits - 1), pos_pct)
                    x_pos_float, x_pos_delta, x_pos_zero_point = self.x_pos_quant_dequant(x_pos, yita, mask_pos, x_pos_delta1, x_pos_zero_point1)       
                    for neg_pct in [0.98, 0.99, 0.999, 0.9999, 0.99999]:
                        x_neg_delta1, x_neg_zero_point1 = self.get_delta_zero_point_percentile(x_neg, 2 ** (self.n_bits - 1), neg_pct)
                        x_neg_float, x_neg_delta, x_neg_zero_point = self.x_neg_quant_dequant(x_neg, yita, mask_neg, x_neg_delta1, x_neg_zero_point1)
                        x_all_dequant = x_pos_float + x_neg_float
                        all_error =  torch.mean((x_all_dequant - x)**2)
                        if all_error < best_error:
                            best_error = all_error
                            best_yita = yita
                            best_pos_delta = x_pos_delta
                            best_pos_zero_point = x_pos_zero_point
                            best_neg_delta = x_neg_delta
                            best_neg_zero_point = x_neg_zero_point
            self.yita = best_yita                    
            self.x_pos_delta = best_pos_delta
            self.x_pos_zero_point = best_pos_zero_point
            self.x_neg_delta = best_neg_delta
            self.x_neg_zero_point = best_neg_zero_point
