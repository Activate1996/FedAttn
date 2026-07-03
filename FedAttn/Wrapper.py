# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 12:32:38 2025

@author: DengXiumei
Transformer Wrapper for sparse attention experiments
"""

import torch

class LayerMaskedTransformerWrapper:
    """
    通用封装类：为 HuggingFace Transformer 模型添加每层的自定义 attention mask
    支持 Llama 和 Qwen 系列模型（自动检测并适配）
    - 支持 prefill 阶段：使用完整输入执行一次前向（填充位置处理）
    - 支持 decode 阶段：逐 token 推理，每步自动扩展 mask 和 padding_mask
    """
    def __init__(self, model, device): 
        """
        初始化封装类，自动检测模型类型并替换对应的 attention forward 函数。

        Args:
            model: HuggingFace 加载的模型（AutoModelForCausalLM）
            device: 设备（如 'cuda' 或 'cpu'）
        """       
        # 保存模型和设备信息
        self.model = model
        self.device = device

        # 初始化掩码和状态变量
        self.prefill_masks = None   # prefill 阶段的掩码列表
        self.decode_masks = None    # decode 阶段的掩码列表
        self.is_prefill = True      # 当前是否处于 prefill 阶段
        self.layer_ptr = 0          # 当前层指针，用于跟踪处理到哪一层
        self.n_layers = len(model.model.layers)  # 模型总层数
        self.batch_size = None      # 批量大小
        self.input_len = None       # 输入序列长度
        self.current_token_counts = None  # 每个样本当前的 token 数量
        
        self.model_type = model.config.model_type                  # 自动检测模型类型（llama 或 qwen2）
        self.attention_class = self._detect_attention_class()      # 根据模型类型获取对应的 attention 类
        self._orig_attn_fwd = self.attention_class.forward         # 保存原始的 forward 方法，用于后续恢复
        self._apply_patch()                                        # 应用 monkey patch，替换 attention forward 方法

    def _detect_attention_class(self):
        """
        自动检测模型使用的 attention 类
        
        Returns:
            attention_class: 对应的 attention 类
        """
        if self.model_type == "llama":
            from transformers.models.llama.modeling_llama import LlamaSdpaAttention
            return LlamaSdpaAttention
        elif self.model_type == "qwen2":
            # 尝试不同的Qwen2 Attention类名
            try:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
                return Qwen2SdpaAttention
            except ImportError:
                try:
                    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
                    return Qwen2Attention
                except ImportError:
                    # 如果都导入失败，查找所有Attention类
                    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
                    for attr_name in dir(qwen2_module):
                        if 'Attention' in attr_name:
                            return getattr(qwen2_module, attr_name)
                    raise ValueError("找不到Qwen2的Attention类")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _apply_patch(self):
        """
        应用 monkey patch 到 attention 类，替换原始的 forward 方法
        """
        def patched(module, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
            """
            重写的 forward 方法：自动根据当前阶段、当前层，写入自定义 attention mask。
            
            Args:
                module: attention 模块实例
                hidden_states: 输入的隐藏状态
                attention_mask: 注意力掩码
                position_ids: 位置编码
                past_key_value: 过去的键值对（用于解码阶段）
                **kwargs: 其他参数
            """
            # 获取当前层索引
            # 优先使用模块自带的 layer_idx，否则使用全局指针
            layer_idx = getattr(module, 'layer_idx', self.layer_ptr % self.n_layers)
            # 如果模块没有 layer_idx 属性，则递增全局指针
            if not hasattr(module, 'layer_idx'):
                self.layer_ptr += 1

            # 根据当前阶段写入自定义 attention mask
            if self.is_prefill:
                # PREFILL 阶段：使用完整的二维掩码
                if self.prefill_masks is not None and layer_idx < len(self.prefill_masks):
                    add_mask = self.prefill_masks[layer_idx]
                    if attention_mask is not None:
                        
                        attention_mask = torch.minimum(attention_mask, add_mask)        # 如果已有掩码，则取最小值（更严格的限制）
                    else:
                        
                        attention_mask = add_mask                                       # 如果没有掩码，则直接使用自定义掩码
            else:
                # DECODE 阶段：需要动态扩展掩码
                if self.decode_masks is not None and layer_idx < len(self.decode_masks):
                    add_mask = self.decode_masks[layer_idx]
                    add_mask = self.extend_attention_mask(add_mask, self.input_len)    # 扩展掩码以匹配当前序列长度
                    if attention_mask is not None:
                        attention_mask = torch.minimum(attention_mask, add_mask)
                    else:
                        attention_mask = add_mask
            
            # 调用原始的 forward 方法
            return self._orig_attn_fwd(
                module, 
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                past_key_value=past_key_value, 
                **kwargs
            )
        
        # 为每层的 attention 模块添加 layer_idx 属性
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = i
        
        # 应用 patch，替换原始的 forward 方法
        self.attention_class.forward = patched
        print(f"已将 {self.attention_class.__name__}.forward 替换为patched函数")
        

    def set_masks(self, prefill_masks=None, decode_masks=None):
        """
        设置 wrapper 的掩码参数。

        Args:
            prefill_masks: List[Tensor]，每层的 [B, 1, T, T] 掩码
            decode_masks: List[Tensor]，每层的 [B, 1, 1, input_len] 掩码（decode 阶段动态扩展）
        """
        self.prefill_masks = prefill_masks
        self.decode_masks = decode_masks
        
    def extend_attention_mask(self, base_mask, max_len):
        """
        将 decode 阶段的 base mask 扩展为当前总序列长度（右侧补零）

        Args:
            base_mask: 原始 input 长度的掩码 [B, 1, 1, S]
            max_len: 当前总长度（S + decode_steps）
        Returns:
            扩展后的掩码 [B, 1, 1, max_len]
        """
        B, _, _, S = base_mask.shape
        extended = torch.zeros(B, 1, 1, max_len, device=self.device, dtype=base_mask.dtype)          # 创建扩展后的掩码张量，初始化为
        extended[:, :, :, :S] = base_mask                                                            # 复制原始掩码到前 S 个位置
        return extended
    
    def extend_padding_mask(self, padding_mask):   
        """
        将 padding mask 右侧追加一个 1（表示新生成的 token 是有效的）
        
        Args:
            padding_mask: 当前的 padding mask [B, T]
        Returns:
            扩展后的 padding mask [B, T+1]
        """
        next_padding_mask = torch.ones((padding_mask.shape[0], 1), device=self.device, dtype=padding_mask.dtype)  # 创建一个形状为 [B, 1] 的张量，值为 1
        return torch.cat([padding_mask, next_padding_mask], dim=-1)                                               # 在序列末尾拼接新的 padding mask  

    def create_position_ids_right_padded(self, attention_mask):
        """
        为右 padding 的输入构造 position_ids
        
        Args:
            attention_mask: 注意力掩码 [B, T]
        Returns:
            position_ids: 位置编码 [B, T]
        """
        position_ids = torch.zeros_like(attention_mask)                                                           # 初始化 position_ids，形状与 attention_mask 相同
        for i in range(self.batch_size):                                                                          # 为每个样本单独处理 
            real_len = attention_mask[i].sum().item()                                                             # 计算有效长度（非 padding 的 token 数量）
            position_ids[i, :real_len] = torch.arange(real_len, dtype=torch.long, device=self.device)             # 为有效 token 分配递增的位置编码  
        return position_ids

    def create_position_ids_left_padded(self, attention_mask):
        """
        为左 padding 的输入构造 position_ids
        
        Args:
            attention_mask: 注意力掩码 [B, T]
        Returns:
            position_ids: 位置编码 [B, T]
        """
        position_ids = torch.zeros_like(attention_mask)                                                           # 初始化 position_ids，形状与 attention_mask 相同       
        for i in range(self.batch_size):           
            real_len = attention_mask[i].sum().item()
            position_ids[i, -real_len:] = torch.arange(real_len, dtype=torch.long, device=self.device)            # 为右侧的有效 token 分配递增的位置编码 
        return position_ids

    def create_position_ids(self, padding_mask, padding_side):
        """
        根据 padding 方向创建 position_ids
        
        Args:
            padding_mask: padding 掩码
            padding_side: padding 方向（"left" 或 "right"）
        Returns:
            position_ids: 位置编码
        """
        if padding_side == "left":
            return self.create_position_ids_left_padded(padding_mask)
        else:
            return self.create_position_ids_right_padded(padding_mask)

    def get_next_logits(self, out_logits, step, padding_side = "left"):
        """
        根据当前步骤和 padding 方向，提取每个样本的下一步 logits。
    
        Args:
            out_logits (Tensor): 模型输出 logits，形状为 [B, T, vocab_size]
            step (int): 当前解码步数（0 表示 prefill 阶段）
            padding_side (str): 'left' 或 'right'，指示 input_ids 的 padding 方向
    
        Returns:
            next_logits (Tensor): 每个样本下一步的 logits，形状为 [B, vocab_size]
        """
        if step == 0:
            # prefill 阶段：需要根据 padding 方向确定最后一个有效 token 的位置
            if padding_side == "right":                                                                       # 右 padding：最后一个有效 token 的位置由 current_token_counts 确定
                return torch.stack([
                    out_logits[b, self.current_token_counts[b] - 1] for b in range(out_logits.size(0))
                ], dim=0)
            elif padding_side == "left":                                                                      # 左 padding：有效 token 在右侧，直接取最后一个位置
                return out_logits[:, -1]
            else:
                raise ValueError(f"Unsupported padding_side: {padding_side}")
        else:
            # decode 阶段：每步只生成一个 token，logits 总是在最后一个位置
            return out_logits[:, -1]

    def sample_next_token(
            self, 
            next_logits: torch.Tensor,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
            ):
        """
        根据 logits 执行采样（支持 greedy、top-k、top-p）
    
        Args:
            next_logits: [B, vocab_size]，每个样本的预测logits
            do_sample: 是否采样（True = sample；False = greedy）
            temperature: 温度系数，控制分布平滑度
            top_k: Top-k 采样中保留的候选数量
            top_p: Top-p（nucleus）采样阈值
    
        Returns:
            next_token_ids: [B, 1]，每个样本下一步选中的 token id
        """
        if not do_sample:
            # 贪心解码：选择概率最大的 token
            return torch.argmax(next_logits, dim=-1, keepdim=True)
    
        probs = torch.nn.functional.softmax(next_logits / temperature, dim=-1)              # 应用温度系数并转换为概率分布
    
        # Top-k 采样
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)                          # 保留前 k 个最高概率的 token
            probs = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)                 # 将其他位置的概率置零
            probs = probs / probs.sum(dim=-1, keepdim=True)                                 # 重新归一化
        
        # Top-p 采样
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)       # 按概率降序排列
            cumulative_probs = sorted_probs.cumsum(dim=-1)                                  # 计算累积概率
            mask = cumulative_probs > top_p                                                 # 找到累积概率超过 top_p 的位置
            mask[..., 0] = False                                                            # 至少保留第一个 token
            sorted_probs[mask] = 0.0                                                        # 将超过阈值的概率置零
            probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)        # 还原到原始顺序
            probs = probs / probs.sum(dim=-1, keepdim=True)                                 # 重新归一化
        
        return torch.multinomial(probs, num_samples=1)                                      # 多项式采样

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
        """
        带掩码的完整推理流程（Prefill + Decode 阶段）

        Args:
            input_ids: prompt 输入 [B, T]
            padding_side: padding方向 ("left" 或 "right")
            padding_mask: padding 位置标记 [B, T]（1为有效）
            prefill_layer_masks: prefill阶段的层级masks
            decode_layer_masks: decode阶段的层级masks
            max_new_tokens: 最多生成多少个 token
            do_sample: 是否采样
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: top-p采样参数

        Returns:
            output_ids: 包含输入和生成内容的完整序列
        """
        
        # 初始化生成参数
        self.batch_size, self.input_len = input_ids.shape
        self.set_masks(prefill_masks=prefill_layer_masks, decode_masks=decode_layer_masks)
        
        # ==================== PREFILL阶段 ====================
        self.is_prefill = True                                                           # 标记为 prefill 阶段
        self.layer_ptr = 0                                                               # 重置层指针
        self.current_token_counts = padding_mask.sum(dim=1)                              # 计算每个样本的有效 token 数量
        
        # 执行 prefill 前向传播
        out = self.model(
            input_ids=input_ids,
            attention_mask=padding_mask,
            position_ids=self.create_position_ids(padding_mask, padding_side),
            use_cache=True,                                                             # 缓存键值对用于后续解码
            return_dict=True,
        )
        past_kv = out.past_key_values                                                   # 保存键值对缓存
        generated = input_ids.clone()                                                   # 复制输入作为生成序列的起始
        
        # ==================== DECODE阶段 ====================
        self.is_prefill = False  # 切换到 decode 阶段
        
        # 逐步生成新 token
        for step in range(max_new_tokens):
            self.layer_ptr = 0  # 重置层指针
            
            # 根据当前 logits 采样下一个 token
            next_id = self.sample_next_token(
                next_logits=self.get_next_logits(out.logits, step, padding_side),
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # 检查是否遇到结束符
            if hasattr(self.model.config, 'eos_token_id') and (next_id == self.model.config.eos_token_id).all():
                print(f"在步骤 {step} 遇到EOS token，停止生成")
                break

            generated = torch.cat([generated, next_id], dim=-1)           # 将新 token 添加到生成序列
            self.current_token_counts += 1                                # 更新 token 计数和序列长度
            self.input_len += 1                                           # 更新 token 计数和序列长度
            padding_mask = self.extend_padding_mask(padding_mask)         # 扩展 padding mask

            # 使用新 token 进行下一步推理
            out = self.model(
                input_ids=next_id,                                        # 只输入新生成的 token
                attention_mask=padding_mask,                              # 扩展后的 padding mask
                position_ids=(self.current_token_counts-1).unsqueeze(1),  # 当前位置编码
                past_key_values=past_kv,                                  # 使用缓存的键值对
                use_cache=True,
                return_dict=True
            )
            past_kv = out.past_key_values  # 更新键值对缓存

        return generated

    def __del__(self):
        """
        析构函数：恢复原始的 attention forward 方法
        确保在对象被销毁时不会影响其他实例
        """
        if hasattr(self, '_orig_attn_fwd') and hasattr(self, 'attention_class'):
            self.attention_class.forward = self._orig_attn_fwd
            print(f"已恢复原始的 {self.attention_class.__name__}.forward")

    def reset(self):
        """
        重置 wrapper 状态，清理 KV 缓存、掩码状态、层计数等，以便重新推理。
        可用于多次调用 generate() 或更换输入后，确保不会出错。
        """
        self.prefill_masks = None
        self.decode_masks = None
        self.is_prefill = True
        self.layer_ptr = 0
        self.batch_size = None
        self.input_len = None
        self.current_token_counts = None
    
        # 清理模型的缓存
        if hasattr(self.model, '_past_key_values'):
            self.model._past_key_values = None
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None         