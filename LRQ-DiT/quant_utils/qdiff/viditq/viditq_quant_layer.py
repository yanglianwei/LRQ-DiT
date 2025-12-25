import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
from qdiff.viditq.duquant_utils import get_rot, exchange_row_col, get_hadamard
import matplotlib.pyplot as plt
import os

class ViDiTQuantizedLinear(QuantizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: torch.nn.Linear,
        smooth_flag = True,
        smooth_alpha = 0.5,
        quant_method="duquant",
        block_size=128,
        max_rotation_step=256,
        permutation_times=1,
        k = 9, 
    ) -> None:
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)
        self.smooth_flag = smooth_flag
        self.alpha = smooth_alpha
        self.channel_mask = None  
        self.quant_method = quant_method   
        if block_size == -1:
            self.block_size = 4096
        else:
            self.block_size = block_size
        self.max_rotation_step = max_rotation_step
        self.permutation_times = permutation_times
        self.init_activation_duquant_params = torch.tensor(1)
        self.init_weight_duquant_params = torch.tensor(1)        
        self.rotation_matrix = None   
        if self.quant_method == 'hadamard':
            self.H = get_hadamard(self.block_size)
        elif self.quant_method == 'duquant':
            self.R, self.permutation_list = [], []
            self.init_activation_duquant_params = torch.tensor(0)
        elif self.quant_method == 'NotRotate':
            pass
        self.k = k
    

    def get_channel_mask(self, act_mask):  
        weight_mask = self.fp_module.weight.abs().max(dim=0)[0] 
        channel_mask = (weight_mask.abs()**self.alpha) / (act_mask.abs()**(1-self.alpha)) 
        self.channel_mask = channel_mask
        assert not torch.isnan(channel_mask.any()), "nan exists in channel mask"
            
    def get_rotation_matrix(self):
        self.rotation_matrix = random_hadamard_matrix(self.in_features, "cuda")



    def init_smooth_duquant_weight(self):
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False  
        assert self.channel_mask is not None
        if self.smooth_flag is True and self.quant_method == 'duquant':
            self.weight.data = self.init_weight_duquant(self.fp_module.weight / self.channel_mask.reshape([1, C_in]))
            self.weight.data = self.w_quantizer(self.weight.data)
        if hasattr(self.w_quantizer, 'twinlog'):
            self.w_quantizer.twinlog = True
        self.w_quantizer.init_done = True

    def init_smooth_hadamard_weight(self):
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False  
        assert self.channel_mask is not None
        if self.smooth_flag is True and self.quant_method == 'hadamard':
            self.weight.data = (self.fp_module.weight / self.channel_mask.reshape([1, C_in])).to(dtype=self.rotation_matrix.dtype) 
            self.weight.data = torch.matmul(self.weight.data, self.rotation_matrix)
            self.weight.data = self.w_quantizer(self.weight.data)
        if hasattr(self.w_quantizer, 'twinlog'):
            self.w_quantizer.twinlog = True
        self.w_quantizer.init_done = True

    def init_smooth_NotRotate_weight(self):
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False  
        assert self.channel_mask is not None
        if self.smooth_flag is True and self.quant_method == 'NotRotate':
            self.weight.data = (self.fp_module.weight / self.channel_mask.reshape([1, C_in]))
            self.weight.data = self.w_quantizer(self.weight.data)
        if hasattr(self.w_quantizer, 'twinlog'):
            self.w_quantizer.twinlog = True
        self.w_quantizer.init_done = True

    def init_NotSmooth_duquant_weight(self):
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False  
        if self.smooth_flag is False and self.quant_method == 'duquant':
            self.weight.data = self.w_quantizer(self.init_weight_duquant(self.fp_module.weight))
        if hasattr(self.w_quantizer, 'twinlog'):
            self.w_quantizer.twinlog = True
        self.w_quantizer.init_done = True

    def init_NotSmooth_hadamard_weight(self):
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False
        if self.smooth_flag is False and self.quant_method == 'hadamard':
            self.weight.data = (self.fp_module.weight).to(dtype=self.rotation_matrix.dtype) 
            self.weight.data = self.w_quantizer(torch.matmul(self.weight.data, self.rotation_matrix)) 
        if hasattr(self.w_quantizer, 'twinlog'):
            self.w_quantizer.twinlog = True
        self.w_quantizer.init_done = True

    def init_NotSmooth_NotRotate_weight(self):
        C_out, C_in = self.fp_module.weight.shape
        self.w_quantizer.init_done = False  
        assert self.smooth_flag is False and self.quant_method == 'NotRotate'
        self.weight.data = self.w_quantizer(self.fp_module.weight)
        if hasattr(self.w_quantizer, 'twinlog'):
            self.w_quantizer.twinlog = True
        self.w_quantizer.init_done = True

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.quant_mode:  
            return self.fp_module(x, *args, **kwargs)
        else:
            B, N_token, C = x.shape

            if self.smooth_flag is True:
                if self.quant_method == 'duquant':
                    x = x.reshape([B*N_token,-1])
                    x = x*(self.channel_mask.reshape([1,C]))
                    x = self.init_activation_duquant(x)
                    self.init_smooth_duquant_weight()
                elif self.quant_method == 'hadamard':
                    x = x*self.channel_mask.reshape([1,1,C])
                    x = x.to(dtype=self.rotation_matrix.dtype) 
                    x = torch.matmul(x, self.rotation_matrix).to(dtype=x.dtype)
                    x = x.reshape([B*N_token,-1])
                    self.init_smooth_hadamard_weight()
                elif self.quant_method == 'NotRotate':
                    x = x*self.channel_mask.reshape([1,1,C])
                    x = x.reshape([B*N_token,-1])
                    self.init_smooth_NotRotate_weight()
            elif self.smooth_flag is False:
                if self.quant_method == 'duquant':
                    x = x.reshape([B*N_token,-1])
                    x = self.init_activation_duquant(x)
                    self.init_NotSmooth_duquant_weight()
                elif self.quant_method == 'hadamard':
                    x = x.to(dtype=self.rotation_matrix.dtype) 
                    x = torch.matmul(x, self.rotation_matrix).to(dtype=x.dtype)
                    x = x.reshape([B*N_token,-1])
                    self.init_NotSmooth_hadamard_weight()
                elif self.quant_method == 'NotRotate':
                    x = x.reshape([B*N_token,-1])
                    self.init_NotSmooth_NotRotate_weight()
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])
            y = F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None, *args, **kwargs)
            return y


    def rotation(self, weight, max_rotation_step=None, other=None, score_func=None):
        if max_rotation_step is None:
            max_rotation_step = self.max_rotation_step
        weight = weight.detach().clone()
        _weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        exchange_ids = []
        peak_values = []
        weight = weight.reshape(-1, self.block_size)
        if other is not None:
            _other = other.detach().clone()
            other = other.reshape(-1, self.block_size)


        Rot = get_rot(self.block_size, weight.device)
        for j in range(max_rotation_step):
            if score_func is not None:
                weight_max = weight.abs().max(dim=0).values
                other_max = other.abs().max(dim=0).values
                r = score_func(weight_max, other_max).argmax().item()
                peak_values.append(score_func(weight_max[r], other_max[r]).item())
                
            else:
                r, c = divmod(weight.argmax().item(), weight.shape[-1])
                r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
                peak_values.append((weight[r, c] - weight[r2, c2]).item())
            exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
            exchange_ids.append(exchange_id)
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(weight)
            weight = torch.matmul(weight, R)
            if other is not None:
                other = torch.matmul(other, R)

        if score_func is not None:
            weight_max = weight.abs().max(dim=0).values
            other_max = other.abs().max(dim=0).values
            r = score_func(weight_max, other_max).argmax().item()
            peak_values.append(score_func(weight_max[r], other_max[r]).item())
        else:
            r, c = divmod(weight.argmax().item(), weight.shape[-1])
            r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
            peak_values.append((weight[r, c] - weight[r2, c2]).item())
        exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
        exchange_ids.append(exchange_id)

        weight = _weight.detach().clone()
        if other is not None:
            other = _other.detach().clone()
        select_length = torch.argmin(torch.tensor(peak_values)).item()
        exchange_ids = exchange_ids[:select_length]
        peak_values = peak_values[:select_length+1]

        R_ = torch.eye(self.block_size).to(weight)
        for exchange_id in exchange_ids:
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(R_)
            R_ = torch.matmul(R_, R)
        weight = torch.matmul(weight.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        if other is not None:
            other = torch.matmul(other.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        return (weight, exchange_ids, R_) if other is None else (weight, other, exchange_ids, peak_values[select_length], R_)


    def permutation_zigzag(self, weight):
        k = self.k
        W = weight
        m = W.size(1)
        col_marks = [torch.max(W[:, j]) - torch.min(W[:, j]) for j in range(m)]
        col_marks = torch.tensor(col_marks)
        sorted_indices = torch.argsort(col_marks, descending=True)
        num_blocks = (m + k - 1) // k
        blocks = [sorted_indices[i*k : (i+1)*k] for i in range(num_blocks)]
        perm = []
        for i in range(k):
            for block in blocks:
                if i < len(block):
                    perm.append(block[i])
        perm = torch.tensor(perm)
        W_new = W[:, perm]
        weight = W_new
        return weight, perm



    def online_duquant_cali(self, weight):
        weight = weight.detach().clone()
        T = {}
        
        self.permutation_list = None
        self.R = None
        for i in range(self.permutation_times):
            weight, _, R = self.rotation(weight)
            if self.R is None:
                self.R = R.unsqueeze(0)
            else:
                self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
            
            weight, perm = self.permutation_zigzag(weight)
            if self.permutation_list is None:
                self.permutation_list = perm.unsqueeze(0)
            else:
                self.permutation_list = torch.cat((self.permutation_list, perm.unsqueeze(0)), dim=0)

        weight, _, R = self.rotation(weight)
        if self.R is None:
            self.R = R.unsqueeze(0)
        else:
            self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
        return weight

    
    def init_weight_duquant(self, weight: torch.Tensor):
        if self.quant_method is None:
            return weight
        if self.quant_method == 'hadamard':
            weight_shape = weight.shape  
            hadamard = self.H.to(weight)
            weight = weight.reshape(-1, self.block_size)
            weight = weight.matmul(hadamard).view(weight_shape)
        elif self.quant_method == 'duquant':
            if not self.init_weight_duquant_params:
                weight = self.online_duquant_cali(weight)
                self.init_weight_duquant_params = torch.tensor(1)
            else:
                weight_size = weight.shape
                weight_type = weight.dtype
                if self.permutation_list is not None:
                    for i in range(len(self.permutation_list)):
                        weight = weight.reshape(-1, self.block_size)
                        R = self.R[i].to(weight)
                        weight = weight.matmul(R).reshape(weight_size).squeeze(0)
                        if True:
                            if len(self.permutation_list.shape) == 3:
                                perm = (self.permutation_list[i, 0].to(weight.device), self.permutation_list[i, 1].to(weight.device))
                                weight[:, perm[0]], weight[:, perm[1]] = weight[:, perm[1]], weight[:, perm[0]]
                            else:
                                perm = self.permutation_list[i].to(weight.device)
                                weight = weight[:, perm]
                if len(self.R) > 0:
                    weight = weight.reshape(-1, self.block_size)
                    R = self.R[-1].to(weight)
                    weight = weight.matmul(R).reshape(weight_size) 
        else:
            raise NotImplementedError
        return weight
    

    def init_activation_duquant(self, x: torch.Tensor):
        if self.quant_method is None:
            return x
        if self.quant_method == 'hadamard':
            x_shape = x.shape  
            hadamard = self.H.to(x)
            x = x.reshape(-1, self.block_size)
            x = x.matmul(hadamard).view(x_shape)
        elif self.quant_method == 'duquant':
            if not self.init_activation_duquant_params:
                x = self.online_duquant_cali(x)
                self.init_activation_duquant_params = torch.tensor(1)
            else:
                x_size = x.shape
                x_type = x.dtype
                if self.permutation_list is not None:
                    for i in range(len(self.permutation_list)):
                        x = x.reshape(-1, self.block_size)
                        R = self.R[i].to(x)
                        x = x.matmul(R).reshape(x_size).squeeze(0)
                        if True:
                            if len(self.permutation_list.shape) == 3:
                                perm = (self.permutation_list[i, 0].to(x.device), self.permutation_list[i, 1].to(x.device))
                                x[:, perm[0]], x[:, perm[1]] = x[:, perm[1]], x[:, perm[0]]
                            else:
                                perm = self.permutation_list[i].to(x.device)
                                x = x[:, perm]
                if len(self.R) > 0:
                    x = x.reshape(-1, self.block_size)
                    R = self.R[-1].to(x)
                    x = x.matmul(R).reshape(x_size) 
        else:
            raise NotImplementedError
        return x
