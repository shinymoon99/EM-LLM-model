from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# 加载原始模型
base_model_name = "base_model_name"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 加载多个 LoRA 模型
lora_model_paths = ["lora_model_1", "lora_model_2", "lora_model_3"]  # 多个 LoRA 模型的路径
lora_models = [PeftModel.from_pretrained(base_model, path) for path in lora_model_paths]

# 初始化平均参数
avg_lora_params = {}

# 遍历每个 LoRA 模块，提取参数并计算平均
for lora_model in lora_models:
    for name, param in lora_model.named_parameters():
        if "lora_" in name:  # 假设 LoRA 参数的名字中包含 'lora_'
            if name not in avg_lora_params:
                avg_lora_params[name] = param.data.clone()  # 初始化
            else:
                avg_lora_params[name] += param.data  # 累加

# 计算平均参数
num_models = len(lora_models)
for name in avg_lora_params:
    avg_lora_params[name] /= num_models  # 取平均

# 将平均参数加载到原始模型的 LoRA 模块中
for name, param in base_model.named_parameters():
    if name in avg_lora_params:
        param.data.copy_(avg_lora_params[name])  # 替换为平均参数

# 现在 base_model 的 LoRA 模块已经被替换为平均参数

# 测试
from transformers import AutoTokenizer

# 测试推理
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 获取输出
output = base_model(input_ids=input_ids).logits
print(tokenizer.decode(output.argmax(dim=-1)[0], skip_special_tokens=True))