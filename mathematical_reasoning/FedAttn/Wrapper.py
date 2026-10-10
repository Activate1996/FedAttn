# -*- coding: utf-8 -*-
"""Masked attention wrapper for Transformer prefill and autoregressive decoding.

Author: DengXiumei."""

import torch

class LayerMaskedTransformerWrapper:
    """Apply per-block attention masks to Hugging Face Llama and Qwen models.

    Prefill processes the full input. Decoding extends the attention and
    padding masks as tokens are generated."""
    def __init__(self, model, device): 
        """Initialize the wrapper and replace the model attention forward method.

        Args:
            model: A Hugging Face causal language model.
            device: Target device, such as "cuda" or "cpu"."""       
        # Store the model and target device.
        self.model = model
        self.device = device

        # Initialize mask and generation state.
        self.prefill_masks = None   # Per-layer prefill masks.
        self.decode_masks = None    # Per-layer decoding masks.
        self.is_prefill = True      # Whether the current forward pass is prefill.
        self.layer_ptr = 0          # Fallback index for the current Transformer block.
        self.n_layers = len(model.model.layers)  # Number of Transformer blocks.
        self.batch_size = None      # Batch size.
        self.input_len = None       # Current input length, including padding.
        self.current_token_counts = None  # Number of valid tokens in each sample.
        
        self.model_type = model.config.model_type                  # Model family (llama or qwen2).
        self.attention_class = self._detect_attention_class()      # Attention class used by the model.
        self._orig_attn_fwd = self.attention_class.forward         # Preserve the attention forward method for restoration.
        self._apply_patch()                                        # Install the masked attention forward method.

    def _detect_attention_class(self):
        """Return the attention class corresponding to the detected model family."""
        if self.model_type == "llama":
            from transformers.models.llama.modeling_llama import LlamaSdpaAttention
            return LlamaSdpaAttention
        elif self.model_type == "qwen2":
            # Resolve the attention class for the installed Transformers version.
            try:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
                return Qwen2SdpaAttention
            except ImportError:
                try:
                    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
                    return Qwen2Attention
                except ImportError:
                    # Fall back to an available Qwen2 attention class.
                    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
                    for attr_name in dir(qwen2_module):
                        if 'Attention' in attr_name:
                            return getattr(qwen2_module, attr_name)
                    raise ValueError("找不到Qwen2的Attention类")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _apply_patch(self):
        """Replace the attention class forward method with masked attention."""
        def patched(module, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
            """Apply the per-block mask for the current generation phase.

            Args:
                module: Attention module instance.
                hidden_states: Input hidden states.
                attention_mask: Additive attention mask.
                position_ids: Token position indices.
                past_key_value: Cached key and value states.
                **kwargs: Additional attention arguments."""
            # Determine the current Transformer block.
            # Prefer the module index; otherwise use the wrapper counter.
            layer_idx = getattr(module, 'layer_idx', self.layer_ptr % self.n_layers)
            # Advance the counter when the module has no layer index.
            if not hasattr(module, 'layer_idx'):
                self.layer_ptr += 1

            # Apply the mask for the current generation phase.
            if self.is_prefill:
                # Prefill uses the full query-by-key attention mask.
                if self.prefill_masks is not None and layer_idx < len(self.prefill_masks):
                    add_mask = self.prefill_masks[layer_idx]
                    if attention_mask is not None:
                        
                        attention_mask = torch.minimum(attention_mask, add_mask)        # Combine additive masks using the more restrictive value.
                    else:
                        
                        attention_mask = add_mask                                       # Use the custom mask when no attention mask is provided.
            else:
                # Extend the decoding mask as the sequence grows.
                if self.decode_masks is not None and layer_idx < len(self.decode_masks):
                    add_mask = self.decode_masks[layer_idx]
                    add_mask = self.extend_attention_mask(add_mask, self.input_len)    # Match the mask width to the current sequence length.
                    if attention_mask is not None:
                        attention_mask = torch.minimum(attention_mask, add_mask)
                    else:
                        attention_mask = add_mask
            
            # Call the preserved attention forward method.
            return self._orig_attn_fwd(
                module, 
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                past_key_value=past_key_value, 
                **kwargs
            )
        
        # Assign a block index to each self-attention module.
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = i
        
        # Install the masked forward method on the attention class.
        self.attention_class.forward = patched
        print(f"已将 {self.attention_class.__name__}.forward 替换为patched函数")
        

    def set_masks(self, prefill_masks=None, decode_masks=None):
        """Set the per-block prefill and decoding masks.

        Args:
            prefill_masks: Masks with shape [B, 1, T, T] for each block.
            decode_masks: Masks with shape [B, 1, 1, input_len] for each block;
                these are extended during decoding."""
        self.prefill_masks = prefill_masks
        self.decode_masks = decode_masks
        
    def extend_attention_mask(self, base_mask, max_len):
        """Extend the decoding mask to the current sequence length with zeros.

        Args:
            base_mask: Input-token mask with shape [B, 1, 1, S].
            max_len: Current sequence length, including generated tokens.
        Returns:
            A mask with shape [B, 1, 1, max_len]."""
        B, _, _, S = base_mask.shape
        extended = torch.zeros(B, 1, 1, max_len, device=self.device, dtype=base_mask.dtype)          # Initialize the extended additive mask with zeros.
        extended[:, :, :, :S] = base_mask                                                            # Copy the mask over the original input positions.
        return extended
    
    def extend_padding_mask(self, padding_mask):   
        """Append one valid-token position to the padding mask.

        Args:
            padding_mask: Current mask with shape [B, T].
        Returns:
            The extended mask with shape [B, T + 1]."""
        next_padding_mask = torch.ones((padding_mask.shape[0], 1), device=self.device, dtype=padding_mask.dtype)  # Mark the appended token as valid.
        return torch.cat([padding_mask, next_padding_mask], dim=-1)                                               # Append one position to the padding mask.

    def create_position_ids_right_padded(self, attention_mask):
        """Construct position IDs for right-padded inputs.

        Args:
            attention_mask: Valid-token mask with shape [B, T].
        Returns:
            Position IDs with shape [B, T]."""
        position_ids = torch.zeros_like(attention_mask)                                                           # Initialize position IDs with the padding-mask shape.
        for i in range(self.batch_size):                                                                          # Process each sample separately.
            real_len = attention_mask[i].sum().item()                                                             # Count valid input tokens.
            position_ids[i, :real_len] = torch.arange(real_len, dtype=torch.long, device=self.device)             # Assign consecutive positions to valid tokens.
        return position_ids

    def create_position_ids_left_padded(self, attention_mask):
        """Construct position IDs for left-padded inputs.

        Args:
            attention_mask: Valid-token mask with shape [B, T].
        Returns:
            Position IDs with shape [B, T]."""
        position_ids = torch.zeros_like(attention_mask)                                                           # Initialize position IDs with the padding-mask shape.
        for i in range(self.batch_size):           
            real_len = attention_mask[i].sum().item()
            position_ids[i, -real_len:] = torch.arange(real_len, dtype=torch.long, device=self.device)            # Assign consecutive positions to valid tokens at the right.
        return position_ids

    def create_position_ids(self, padding_mask, padding_side):
        """Construct position IDs for the specified padding side.

        Args:
            padding_mask: Valid-token mask.
            padding_side: "left" or "right".
        Returns:
            Position IDs for the input sequence."""
        if padding_side == "left":
            return self.create_position_ids_left_padded(padding_mask)
        else:
            return self.create_position_ids_right_padded(padding_mask)

    def get_next_logits(self, out_logits, step, padding_side = "left"):
        """Select the next-token logits for the current generation step.

        Args:
            out_logits: Model logits with shape [B, T, vocab_size].
            step: Generation step; zero selects logits from prefill.
            padding_side: Input padding side, "left" or "right".
        Returns:
            Next-token logits with shape [B, vocab_size]."""
        if step == 0:
            # Prefill selects the last valid input token according to padding side.
            if padding_side == "right":                                                                       # For right padding, use the valid-token count.
                return torch.stack([
                    out_logits[b, self.current_token_counts[b] - 1] for b in range(out_logits.size(0))
                ], dim=0)
            elif padding_side == "left":                                                                      # For left padding, the final position is valid.
                return out_logits[:, -1]
            else:
                raise ValueError(f"Unsupported padding_side: {padding_side}")
        else:
            # During decoding, select the final output position.
            return out_logits[:, -1]

    def sample_next_token(
            self, 
            next_logits: torch.Tensor,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
            ):
        """Select tokens by greedy decoding or filtered multinomial sampling.

        Args:
            next_logits: Next-token logits with shape [B, vocab_size].
            do_sample: Sample tokens if True; otherwise use greedy decoding.
            temperature: Temperature applied before probability normalization.
            top_k: Number of highest-probability candidates to retain.
            top_p: Cumulative-probability threshold.
        Returns:
            Selected token IDs with shape [B, 1]."""
        if not do_sample:
            # Greedy decoding selects the highest-probability token.
            return torch.argmax(next_logits, dim=-1, keepdim=True)
    
        probs = torch.nn.functional.softmax(next_logits / temperature, dim=-1)              # Apply temperature scaling and convert logits to probabilities.
    
        # Apply top-k filtering.
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)                          # Retain the k highest-probability tokens.
            probs = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)                 # Set all other probabilities to zero.
            probs = probs / probs.sum(dim=-1, keepdim=True)                                 # Renormalize the retained probabilities.
        
        # Apply cumulative-probability filtering.
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)       # Sort tokens by decreasing probability.
            cumulative_probs = sorted_probs.cumsum(dim=-1)                                  # Compute cumulative probabilities.
            mask = cumulative_probs > top_p                                                 # Identify positions whose cumulative probability exceeds top_p.
            mask[..., 0] = False                                                            # Always retain the highest-probability token.
            sorted_probs[mask] = 0.0                                                        # Zero probabilities beyond the threshold.
            probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)        # Restore the vocabulary ordering.
            probs = probs / probs.sum(dim=-1, keepdim=True)                                 # Renormalize the retained probabilities.
        
        return torch.multinomial(probs, num_samples=1)                                      # Sample one token from the resulting distribution.

    @torch.no_grad()
    def generate(self, 
                 input_ids, 
                 padding_side="left",
                 padding_mask=None, 
                 prefill_layer_masks=None, 
                 decode_layer_masks=None,
                 max_new_tokens=50, 
                 do_sample=True,
                 temperature=0.7,
                 top_k=40,
                 top_p=0.9,
                 **kwargs):
        """Generate tokens using masked prefill and autoregressive decoding.

        Args:
            input_ids: Input token IDs with shape [B, T].
            padding_side: Input padding side, "left" or "right".
            padding_mask: Mask with shape [B, T]; one denotes a valid token.
            prefill_layer_masks: Per-block prefill masks.
            decode_layer_masks: Per-block decoding masks.
            max_new_tokens: Maximum number of generated tokens.
            do_sample: Whether to sample tokens.
            temperature: Sampling temperature.
            top_k: Number of candidates retained by top-k filtering.
            top_p: Cumulative-probability threshold.
        Returns:
            Token IDs containing the input and generated continuation."""
        
        # Initialize generation state and masks.
        self.batch_size, self.input_len = input_ids.shape
        self.set_masks(prefill_masks=prefill_layer_masks, decode_masks=decode_layer_masks)
        
        # Prefill.
        self.is_prefill = True                                                           # Enter the prefill phase.
        self.layer_ptr = 0                                                               # Reset the block counter.
        self.current_token_counts = padding_mask.sum(dim=1)                              # Count valid tokens in each sample.
        
        # Run the prefill forward pass.
        out = self.model(
            input_ids=input_ids,
            attention_mask=padding_mask,
            position_ids=self.create_position_ids(padding_mask, padding_side),
            use_cache=True,                                                             # Retain KV states for decoding.
            return_dict=True,
        )
        past_kv = out.past_key_values                                                   # Store the KV cache.
        generated = input_ids.clone()                                                   # Initialize the generated sequence with the input.
        
        # Decoding.
        self.is_prefill = False  # Enter the decoding phase.
        
        # Generate tokens sequentially.
        for step in range(max_new_tokens):
            self.layer_ptr = 0  # Reset the block counter.
            
            # Select the next token from the current logits.
            next_id = self.sample_next_token(
                next_logits=self.get_next_logits(out.logits, step, padding_side),
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Stop when every sample selects EOS in the same step.
            if hasattr(self.model.config, 'eos_token_id') and (next_id == self.model.config.eos_token_id).all():
                print(f"在步骤 {step} 遇到EOS token，停止生成")
                break

            generated = torch.cat([generated, next_id], dim=-1)           # Append the selected token.
            self.current_token_counts += 1                                # Update valid-token counts.
            self.input_len += 1                                           # Update the current sequence length.
            padding_mask = self.extend_padding_mask(padding_mask)         # Extend the padding mask.

            # Run the next forward pass with the selected token.
            out = self.model(
                input_ids=next_id,                                        # Pass only the newly generated token.
                attention_mask=padding_mask,                              # Use the extended padding mask.
                position_ids=(self.current_token_counts-1).unsqueeze(1),  # Use the position of the new token.
                past_key_values=past_kv,                                  # Reuse the KV cache.
                use_cache=True,
                return_dict=True
            )
            past_kv = out.past_key_values  # Update the KV cache.

        return generated

    def __del__(self):
        """Restore the saved attention forward method when the wrapper is destroyed."""
        if hasattr(self, '_orig_attn_fwd') and hasattr(self, 'attention_class'):
            self.attention_class.forward = self._orig_attn_fwd
            print(f"已恢复原始的 {self.attention_class.__name__}.forward")

    def reset(self):
        """Reset masks, block counters, token counts, and exposed model cache state."""
        self.prefill_masks = None
        self.decode_masks = None
        self.is_prefill = True
        self.layer_ptr = 0
        self.batch_size = None
        self.input_len = None
        self.current_token_counts = None
    
        # Clear cache attributes exposed by the model.
        if hasattr(self.model, '_past_key_values'):
            self.model._past_key_values = None
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None         