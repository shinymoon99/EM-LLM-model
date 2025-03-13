import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer

# 自定义 LoRA 模块
class LoRAModel(nn.Module):
    def __init__(self, base_model, target_layers, target_modules, rank=8, alpha=1.0):
        super(LoRAModel, self).__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.target_layers = target_layers
        self.target_modules = target_modules
        
        # 存储原始参数和LoRA参数的映射
        self.lora_params_mapping = {}
        
        # 在指定层和模块上添加 LoRA 参数
        for name, param in self.base_model.named_parameters():
            # 针对Mistral MoE模型的层名匹配逻辑
            if self._is_target_param(name, param):
                self._add_lora_to_layer(name, param)
                
        # 冻结原始模型的参数
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _is_target_param(self, name, param):
        """检查参数是否是目标参数"""
        # 针对Mistral MoE模型的层名匹配逻辑
        is_target_layer = any(f"layers.{layer_idx}." in name for layer_idx in self.target_layers)
        is_target_module = any(module in name for module in self.target_modules)
        is_weight = "weight" in name
        
        # 针对MoE模型的FFN专家检查
        is_expert = "experts" in name  # 如果是专家层
        
        return is_target_layer and is_target_module and is_weight

    def _add_lora_to_layer(self, name, param):
        # 添加 LoRA 参数（低秩矩阵 A 和 B）
        in_features = param.shape[1]
        out_features = param.shape[0]
        self.register_parameter(
            f"lora_A_{name}",
            nn.Parameter(torch.randn(in_features, self.rank)),
        )
        self.register_parameter(
            f"lora_B_{name}",
            nn.Parameter(torch.zeros(self.rank, out_features)),
        )

    def forward(self, **kwargs):
        # 保存原始权重
        original_weights = {}
        
        # 应用 LoRA 参数
        for name, param in self.base_model.named_parameters():
            if name in self.lora_params_mapping:
                original_weights[name] = param.data.clone()
                lora_A_name, lora_B_name = self.lora_params_mapping[name]
                lora_A = getattr(self, lora_A_name)
                lora_B = getattr(self, lora_B_name)
                param.data = param.data + self.alpha * (lora_B @ lora_A)
        
        # 前向传播
        output = self.base_model(**kwargs)
        
        # 恢复原始权重
        for name, original_weight in original_weights.items():
            param = dict(self.base_model.named_parameters())[name]
            param.data.copy_(original_weight)
            
        return output

    def get_lora_params(self):
        # 获取所有 LoRA 参数
        lora_params = {}
        for name, param in self.named_parameters():
            if "lora_" in name:
                lora_params[name] = param.data
        return lora_params

    def set_lora_params(self, lora_params):
        # 设置 LoRA 参数
        for name, param in self.named_parameters():
            if name in lora_params:
                param.data.copy_(lora_params[name])

    def save_lora_weights(self, filepath):
        # 保存 LoRA 权重到文件
        lora_params = self.get_lora_params()
        torch.save(lora_params, filepath)

    def load_lora_weights(self, filepath):
        # 从文件加载 LoRA 权重
        lora_params = torch.load(filepath)
        self.set_lora_params(lora_params)

# 加权构建新的 LoRAExpertModule
def weighted_average_lora(lora_modules, weights):
    avg_lora_params = {}
    for lora_module, weight in zip(lora_modules, weights):
        lora_params = lora_module.get_lora_params()
        for name, param in lora_params.items():
            if name not in avg_lora_params:
                avg_lora_params[name] = param * weight
            else:
                avg_lora_params[name] += param * weight
    return avg_lora_params

# 示例：加载 LLaMA-7B 模型
base_model_name = "huggyllama/llama-7b"
base_model = LlamaForCausalLM.from_pretrained(base_model_name)
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)

# 定义目标层和模块
target_layers = [0, 1, 2]  # 在 0、1、2 层上加载 LoRA
target_modules = ["q_proj", "v_proj"]  # 在 query 和 value 投影层上加载 LoRA

# 初始化多个 LoRA 模块
lora_models = [
    LoRAModel(base_model, target_layers, target_modules, rank=8, alpha=1.0) for _ in range(3)
]

# 训练 LoRA 模块（示例）
for lora_model in lora_models:
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=1e-3)
    for _ in range(100):  # 示例训练
        optimizer.zero_grad()
        input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
        output = lora_model(input_ids=input_ids)
        loss = output.logits.mean()  # 示例损失函数
        loss.backward()
        optimizer.step()  # 仅调整 lora_weights

# 保存 LoRA 权重
for idx, lora_model in enumerate(lora_models):
    lora_model.save_lora_weights(f"lora_weights_{idx}.pt")

# 加权构建新的 LoRAExpertModule
weights = [0.4, 0.3, 0.3]  # 加权权重
avg_lora_params = weighted_average_lora(lora_models, weights)

# 创建新的 LoRAExpertModule 并加载加权平均参数
new_lora_model = LoRAModel(base_model, target_layers, target_modules, rank=8, alpha=1.0)
new_lora_model.set_lora_params(avg_lora_params)

# 保存新的 LoRA 权重
new_lora_model.save_lora_weights("avg_lora_weights.pt")

# 加载新的 LoRA 权重
loaded_lora_model = LoRAModel(base_model, target_layers, target_modules, rank=8, alpha=1.0)
loaded_lora_model.load_lora_weights("avg_lora_weights.pt")

# 推理
input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
output = loaded_lora_model(input_ids=input_ids)
print(tokenizer.decode(output.logits.argmax(dim=-1)[0], skip_special_tokens=True))