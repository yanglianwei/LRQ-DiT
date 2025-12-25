
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor, FusedAttnProcessor2_0
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle


import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules
from qdiff.base.quant_model import quant_layer_refactor_, bitwidth_refactor_, load_quant_param_dict_, save_quant_param_dict_, set_init_done_

from models.quantize_and_save_weight_ import quantize_and_save_weight_  

logger = logging.get_logger(__name__)  

class CustomizePixArtTransformer2DModel(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "PatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        use_additional_conditions: Optional[bool] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default",
    ):
        super().__init__()

        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_additional_conditions is None:
            if sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        self.gradient_checkpointing = False

        self.height = self.config.sample_size
        self.width = self.config.sample_size

        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )
        self.caption_projection = None
        if self.config.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.config.caption_channels, hidden_size=self.inner_dim
            )
            
    
    def convert_quant(self, quant_config):
        self.quant_config = quant_config
            
        self.quant_param_dict = {}
        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        apply_func_to_submodules(self, 
                class_type=nn.Linear,
                function=quant_layer_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None,
                remain_fp_regex=self.quant_config.remain_fp_regex,
                )

    def save_quant_param_dict(self):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=save_quant_param_dict_,
                full_name=None,
                parent_module=None,
                model=self
                )

    def load_quant_param_dict(self, quant_param_dict):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=load_quant_param_dict_,
                full_name=None,
                parent_module=None,
                quant_param_dict=quant_param_dict,
                model=self,
                )

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
        
    
    def quantize_and_save_weight(self, save_path):

        for param in self.parameters():
            param.requires_grad_(False)

        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=quantize_and_save_weight_,
                full_name=None,
                )
        sd = self.state_dict()
        keys_to_delte = ['fp_weight','fp_module']
        keys_to_rename = {
                'w_quantizer.delta': 'scale_weight',
                'w_quantizer.zero_point': 'zp_weight',
                }
        for k in list(sd.keys()):
            if any(substring in k for substring in keys_to_delte):
                del sd[k]
            if any(substring in k for substring in keys_to_rename):
                original_k = k
                for substring in keys_to_rename.keys():
                    if substring in k:
                        k = k.replace(substring, keys_to_rename[substring])
                sd[k] = sd.pop(original_k)

        n_block = len(self.transformer_blocks)
        for i_block in range(n_block):
            hidden_size = self.transformer_blocks[0].ff.net[0].proj.in_features
            sd['transformer_blocks.{}.norm1.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)
            sd['transformer_blocks.{}.norm2.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)  
        
        logger.info('Remaining Keys')
        for k in list(sd.keys()):
            logger.info('key: {},  {}'.format(k, sd[k].dtype))
        torch.save(sd, save_path)

        logger.info("\nFinished Saving the Quantized Checkpoint...\n")


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:

        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):

        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):

        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):

        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):

        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        if attention_mask is not None and attention_mask.ndim == 2:

            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        from datetime import datetime
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        from datetime import datetime
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                from datetime import datetime
                current_time = datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                    **ckpt_kwargs,
                )
            else:

                from datetime import datetime
                current_time = datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

        from datetime import datetime
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
