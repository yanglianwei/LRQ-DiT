from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from diffusers.models.attention import _chunked_feed_forward, GatedSelfAttentionDense

import xformers.ops

logger = logging.get_logger(__name__)

class BasicTransformerBlockWithCudaKernel(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        quant_params: QuantParams = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        
        self.quant_params = quant_params

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None
            
        self.use_kernel = [True, True, True]  


        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            
        if self.use_kernel[0]:
            self.attn1 = QuantAttentionWithCudaKernel(
                dim=dim,
                num_heads=num_attention_heads,
                quant_params=quant_params,
                has_bias=attention_bias,  
                weight_sym=False,       
            )
            self.norm1 = LayerNormGeneral(dim, act_sum=True, eps=1e-6)
        else:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )


        if cross_attention_dim is not None or double_self_attention:
            
            if self.use_kernel[1]:
                self.attn2 = QuantCrossAttentionWithCudaKernel(
                    dim=dim,
                    num_heads=num_attention_heads,
                    quant_params=quant_params,
                    has_bias=attention_bias,  
                    weight_sym=False,         
                )

            else:
                self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    out_bias=attention_out_bias,
                )  

        else:
            self.norm2 = None
            self.attn2 = None
    

        if norm_type == "ada_norm":
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm2 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else: 
            self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)


        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )
        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None
        
        if self.use_kernel[2]:
            self.ff = QuantMlpWithCudaKernel(
                in_features=dim,
                hidden_features=dim*4, 
                quant_params=quant_params,
                has_bias=ff_bias,
                weight_sym=False,
            )

            self.norm2 = LayerNormGeneral(dim, act_sum=True, eps=1e-6)
        else:
            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )
        
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")
            

        if not self.use_kernel[0]:  
        
            batch_size = hidden_states.shape[0]
            assert self.norm_type == "ada_norm_single"
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1) 
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1) 
            else:
                raise ValueError("Incorrect norm used")

            assert self.pos_embed is None
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
            
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,  
                attention_mask=attention_mask,  
                **cross_attention_kwargs,   
            )
            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
                

            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
            
        else:

            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
            
            assert self.norm_type == "ada_norm_single"
            assert gligen_kwargs is None
            assert cross_attention_kwargs == {}

            batch_size = hidden_states.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).to(torch.float16).chunk(6, dim=1)  
            hidden_states = hidden_states.to(torch.float16)
            hidden_states = hidden_states.contiguous()
            
            residual = hidden_states
            norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa, self.quant_params)
            
            B, N, C  = norm_hidden_states.shape
            attn_out = self.attn1(norm_hidden_states)
            hidden_states = fused_kernels.gate_residual_fuse(attn_out.view(-1, C), gate_msa.view(-1, C), residual.view(-1, C)).view(B, N, C)


        if self.attn2 is not None:
            
            if not self.use_kernel[1]:
                

                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states
            else:
                B, N, C  = hidden_states.shape
                residual = hidden_states
                hidden_states = fused_kernels.quant_sum(hidden_states, self.quant_params.sum_input, self.quant_params.scale_input)
                attn_out = self.attn2(hidden_states, encoder_hidden_states, encoder_attention_mask)
                hidden_states = residual + attn_out

        if not self.use_kernel[2]:
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
                
        else:
            B, N, C  = hidden_states.shape
            residual = hidden_states
            norm_hidden_states = self.norm2(hidden_states, shift_mlp, scale_mlp, self.quant_params)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = fused_kernels.gate_residual_fuse(ff_output.view(-1, C), gate_mlp.view(-1, C), residual.view(-1, C)).view(B, N, C)
        
        return hidden_states

class QuantAttentionWithCudaKernel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads: int,
        quant_params: QuantParams,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_k = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_v = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_out = torch.nn.ModuleList([
            W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym),
            nn.Dropout(p=0.0)]
        )

        self.quant_params = quant_params

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.to_q(x, self.quant_params)
        k = self.to_k(x, self.quant_params)
        v = self.to_v(x, self.quant_params)

        dtype = q.dtype

        q = q.view(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.view(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.view(B, N, self.num_heads, C // self.num_heads).to(dtype)

        x = xformers.ops.memory_efficient_attention(q, k, v).view(B, N, C)

        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.to_out[0](x, self.quant_params)

        return x
    
class QuantCrossAttentionWithCudaKernel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        quant_params: QuantParams,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_k = nn.Linear(dim, dim).to(torch.float16)
        self.to_v = nn.Linear(dim, dim).to(torch.float16)
        self.to_out = torch.nn.ModuleList([
            W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym),
            nn.Dropout(p=0.0)]  
        )

        self.quant_params = quant_params
    

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:

        head_size = self.num_heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        _, target_length, _x = cond.shape
        self.head_dim = C // self.num_heads
        assert C % self.num_heads == 0
        attn_mask = mask.unsqueeze(1).repeat(1,self.num_heads,1,1) 
        
        
        q = self.to_q(x, self.quant_params).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)  
        k = self.to_k(cond).reshape([B,target_length,self.num_heads,-1]).transpose(1,2) 
        v = self.to_v(cond).reshape([B,target_length,self.num_heads,-1]).transpose(1,2)  
        
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(B, N, self.num_heads*self.head_dim)
 

        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.to_out[0](x, self.quant_params)

        return x

class QuantMlpWithCudaKernel(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        quant_params: QuantParams,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [
                nn.ModuleDict({
                    'proj': W8A8OF16LinearDynamicInputScale(in_features, hidden_features, has_bias=has_bias, weight_sym=weight_sym),  
                }),
                nn.Dropout(p=0.0),
                W8A8OF16LinearDynamicInputScale(hidden_features, in_features, has_bias=has_bias, weight_sym=weight_sym),  
            ]
        )
        self.quant_params = quant_params

    def forward(self, x):
        x = self.net[0].proj(x, self.quant_params)        
        x = fused_kernels.gelu_quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.net[2](x, self.quant_params)
        return x
    
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