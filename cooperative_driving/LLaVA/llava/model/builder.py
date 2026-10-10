# -*- coding: utf-8 -*-
"""
Pretrained model loading for cooperative-driving inference.

Author: DengXiumei.
"""

# Imports.

import os           # Filesystem operations.
import warnings     # Non-fatal model-loading warnings.
import shutil       # Configuration-file copying.

# Transformers model, tokenizer, and quantization interfaces.
from transformers import (
    AutoTokenizer,           # Tokenizer selected from the checkpoint configuration.
    AutoModelForCausalLM,    # Causal language-model interface.
    AutoConfig,              # Model configuration.
    BitsAndBytesConfig       # Quantization settings.
)

import torch  # PyTorch.

# LLaVA model classes with cooperative-driving feature support.
from llava.model import *

# Special tokens used by the multimodal tokenizer.
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,   # Image-patch token.
    DEFAULT_IM_START_TOKEN,      # Image-start token.
    DEFAULT_IM_END_TOKEN         # Image-end token.
)

# Pretrained-model loading.

def load_pretrained_model(
    model_path,           # Checkpoint directory or model identifier.
    
    model_base,           # Base-model identifier; required for LoRA and projector-only checkpoints.
    
    model_name,           # Name used to select the model-loading branch.
    
    load_8bit=False,      # Load weights with 8-bit quantization.
    
    load_4bit=False,      # Load weights with 4-bit quantization.
    
    device_map="cuda:0",    # Model placement passed to Transformers.
    
    device="cuda",        # A non-default device overrides the model device map.
    
    use_flash_attn=False, # Use the FlashAttention 2 implementation.
    
    **kwargs              # Additional model arguments, including my_model_config.
    ):
    
    
    # Device and precision configuration.
    
    # Forward the device map and additional loading arguments.
    kwargs = {"device_map": device_map, **kwargs}

    # Place the complete model on an explicitly selected non-default device.
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    
    # Quantized and full-precision loading options.
    
    if load_8bit:
        # Enable bitsandbytes 8-bit loading.
        kwargs['load_in_8bit'] = True
        
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,                    # Configure NF4 quantization and the compute dtype.
            
            bnb_4bit_compute_dtype=torch.float16, # Use float16 for quantized computations.
            
            bnb_4bit_use_double_quant=True,       # Quantize the quantization constants.
            
            bnb_4bit_quant_type='nf4'             # NormalFloat4 quantization.
        )
    else:
        # Use float16 when quantization is disabled.
        kwargs['torch_dtype'] = torch.float16

    # Optional FlashAttention 2 implementation.
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # Select the model-loading branch.
    
    # LLaVA models require multimodal configuration and projector weights.
    if 'llava' in model_name.lower():
        
        # LoRA checkpoints require the matching base-model weights.
        
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. '
                'If you are loading a LoRA model, please provide the `model_base` argument. '
                'Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.'
            )

        # Load the base model, non-LoRA trainable parameters, and LoRA adapter.
        if 'lora' in model_name.lower() and model_base is not None:
            
            # LLaVA configuration class.
            from llava.model.language_model.llava_llama import LlavaConfig
            
            # Use the adapter checkpoint configuration.
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            
            # Load the base-model tokenizer.
            tokenizer = AutoTokenizer.from_pretrained(
                model_base,        # Base-model tokenizer.
                use_fast=False     # Use the slow tokenizer implementation.
            )
            
            # Load base weights with the adapter checkpoint configuration.
            print('Loading LLaVA from base model...')
            
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,              # Base-model path or repository identifier.
                
                low_cpu_mem_usage=True,  # Reduce peak CPU memory during loading.

                config=lora_cfg_pretrained,  # Configuration used to construct the model.
                
                **kwargs                 # Additional placement, precision, attention, and V2V settings.
            )
            
            # Match the output and input embedding dimensions to the checkpoint.
            
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            # token_num is the vocabulary size; tokem_dim is the hidden dimension.
            
            if model.lm_head.weight.shape[0] != token_num:
                # Allocate output embeddings with the model dtype and device.
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num,       # Vocabulary size.
                        tokem_dim,       # Hidden dimension.
                        device=model.device,  # Model device.
                        dtype=model.dtype     # Model dtype.
                    )
                )
                # Allocate input embeddings with matching dimensions.
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, 
                        tokem_dim, 
                        device=model.device, 
                        dtype=model.dtype
                    )
                )
                # These uninitialized tensors must be filled by the checkpoint weights.

            # Load trainable parameters stored outside the LoRA adapter.
            
            print('Loading additional LLaVA weights...')
            
            # Use a local non-LoRA parameter file when available.
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, 'non_lora_trainables.bin'), 
                    map_location='cpu'  # Load checkpoint tensors onto the CPU.
                )
            else:
                # Otherwise resolve the parameter file from the Hugging Face Hub.
                from huggingface_hub import hf_hub_download
                
                def load_from_hf(repo_id, filename, subfolder=None):
                    """Download a checkpoint file from the Hub and load it onto the CPU."""
                    cache_file = hf_hub_download(
                        repo_id=repo_id,      # Hub repository identifier.
                        filename=filename,     # Checkpoint filename.
                        subfolder=subfolder    # Optional repository subfolder.
                    )
                    return torch.load(cache_file, map_location='cpu')
                
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            
            # Normalize parameter-name prefixes before loading.
            
            # Remove the PEFT base_model prefix.
            non_lora_trainables = {
                (k[11:] if k.startswith('base_model.') else k): v 
                for k, v in non_lora_trainables.items()
            }
            
            # Remove a duplicated model prefix.
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith('model.') else k): v 
                    for k, v in non_lora_trainables.items()
                }
            
            # Load the partial state dictionary.
            model.load_state_dict(non_lora_trainables, strict=False, assign=True)
            # Parameters absent from this dictionary retain their base-model values.
            # Move assigned projector tensors to the model device and dtype.
            target_device = next(model.parameters()).device
            model.get_model().mm_projector = model.get_model().mm_projector.to(target_device, dtype=torch.float16)
            
            if hasattr(model.get_model(), "mm_scene_projector"):
                model.get_model().mm_scene_projector = model.get_model().mm_scene_projector.to(target_device, dtype=torch.float16)
            
            
            

            # Load the LoRA adapter.
            from peft import PeftModel
            
            print('Loading LoRA weights...')
            
            # Attach the adapter to the base model.
            model = PeftModel.from_pretrained(model, model_path)
            
            # Merge the adapter weights into the model and remove its wrapper.
            print('Merging LoRA weights...')
            
            model = model.merge_and_unload()
            
            print('Model is loaded...')
            
        # Load a base model with a separately trained multimodal projector.
        elif model_base is not None:
            
            print('Loading LLaVA from base model...')
            
            # MPT models require their repository configuration module.
            if 'mpt' in model_name.lower():
                
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    # Copy the base configuration if it is absent from the checkpoint.
                    shutil.copyfile(
                        os.path.join(model_base, 'configuration_mpt.py'), 
                        os.path.join(model_path, 'configuration_mpt.py')
                    )
                
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                
                cfg_pretrained = AutoConfig.from_pretrained(
                    model_path, 
                    trust_remote_code=True  # Load the model-specific configuration implementation.
                )
                
                model = LlavaMptForCausalLM.from_pretrained(
                    model_base, 
                    low_cpu_mem_usage=True, 
                    config=cfg_pretrained, 
                    **kwargs
                )
            else:
                # LLaMA-based model.
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                
                # Read the multimodal checkpoint configuration.
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                
                # Construct the model from base weights and checkpoint settings.
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, 
                    low_cpu_mem_usage=True, 
                    config=cfg_pretrained, 
                    **kwargs
                )

            # Load the multimodal projector weights.
            mm_projector_weights = torch.load(
                os.path.join(model_path, 'mm_projector.bin'), 
                map_location='cpu'
            )
            
            # Convert projector tensors to float16.
            mm_projector_weights = {
                k: v.to(torch.float16) 
                for k, v in mm_projector_weights.items()
            }
            
            # Load only the parameters provided by the projector checkpoint.
            model.load_state_dict(mm_projector_weights, strict=False, assign=True)
            
        # Load a complete LLaVA checkpoint without a separate base model.
        else:
            
            if 'mpt' in model_name.lower():
                # MPT-based LLaVA.
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    **kwargs
                )
                
            elif 'mistral' in model_name.lower():
                # Mistral-based LLaVA.
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                
            else:
                # LLaMA-based LLaVA.
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                
    # Language-only models.
    else:
        
        if model_base is not None:
            # Load a base language model and its LoRA adapter.
            from peft import PeftModel
            
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            
            # Construct the base causal language model.
            model = AutoModelForCausalLM.from_pretrained(
                model_base, 
                low_cpu_mem_usage=True, 
                **kwargs
            )
            
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            
            print(f"Merging weights")
            model = model.merge_and_unload()
            
            print('Convert to FP16...')
            model.to(torch.float16)
            # Convert the merged language model to float16.
            
        else:
            # Load a complete language-model checkpoint.
            use_fast = False
            
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    trust_remote_code=True,  # Load the model-specific implementation.
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    **kwargs
                )

    # Multimodal tokenizer and vision components.
    
    # Language-only models do not require an image processor.
    image_processor = None

    if 'llava' in model_name.lower():
        # Add the image tokens enabled by the checkpoint configuration.
        
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # Optional image-boundary tokens.
        
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        # Optional image-patch token.
        
        # Register the image-patch token.
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        
        # Register image-start and image-end tokens.
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], 
                special_tokens=True
            )
        
        # Resize token embeddings to match the tokenizer vocabulary.
        model.resize_token_embeddings(len(tokenizer))

        # Load the vision encoder.
        vision_tower = model.get_vision_tower()
        
        # Resolve deferred vision-model loading.
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        
        # Apply explicit vision-model placement when automatic mapping is disabled.
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        
        # Return the preprocessing transform associated with the vision encoder.
        image_processor = vision_tower.image_processor

    # Context length.
    
    # Use the configured maximum sequence length when available.
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        # Fallback context length.
        context_len = 2048
    

    # Return the tokenizer, model, image processor, and context length.
    
    return tokenizer, model, image_processor, context_len









