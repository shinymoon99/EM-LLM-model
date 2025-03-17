import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import os   
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import json
from benchmark.other_pred import get_version
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
# 创建 LoRA 配置
def create_lora_config(target_modules, layers_to_transform=[0,1,2], r=8, alpha=16):
    """创建 LoRA 配置"""
    # 构建 target_modules 列表，包含特定层的特定模块
    # 对于 Mistral MoE，我们需要针对专家层进行配置
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        layers_to_transform=layers_to_transform,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=None,
    )

# 加载基础模型和分词器
def load_base_model(model_name_or_path, device_map="auto"):
    """加载基础模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # 为 Mistral 模型设置 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用 init_empty_weights 和 load_checkpoint_and_dispatch 加载模型到多个 GPU
    # 首先创建一个空权重的模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
    )
    
    # 然后将模型分发到多个 GPU
    model = load_checkpoint_and_dispatch(
        model, 
        model_name_or_path, 
        device_map=device_map,
        dtype=torch.float16,
        no_split_module_classes=["MistralDecoderLayer"]  # 根据模型架构调整
    )
    
    return model, tokenizer


# def load_base_model(model_name_or_path, device_map="auto"):
#     """加载基础模型和分词器"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
#     # 为 Mistral 模型设置 padding token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # 使用 4-bit 量化加载模型
#     from transformers import BitsAndBytesConfig
    
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#     )
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         device_map=device_map,
#         quantization_config=quantization_config,
#     )
    
#     return model, tokenizer
# 创建 LoRA 模型
def create_lora_model(base_model, lora_config):
    """创建 LoRA 模型"""
    return get_peft_model(base_model, lora_config)

# 训练 LoRA 模型
def train_lora_model(lora_model, tokenizer,dataloader,num_epochs=10, learning_rate=1e-3):
    """训练 LoRA 模型"""
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 输入会自动分发到正确的设备
        max_length = 20  # 设置一个固定长度
        if dataloader is not None:
            for batch in dataloader:
                inputs = batch["input_ids"]
                labels = batch["labels"]
        # else:
        #     inputs = tokenizer("Hello, how are you?", return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        #     labels = tokenizer("I am fine, thank you!", return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        
                # 将输入放到模型的第一个设备上
                first_device = next(lora_model.parameters()).device
                inputs = inputs.to(first_device)
                labels = labels.to(first_device)
                
                outputs = lora_model(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 2 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    
    return lora_model

# 保存 LoRA 模型
def save_lora_model(lora_model, save_path):
    """保存 LoRA 模型"""
    lora_model.save_pretrained(save_path)

# 加载 LoRA 模型
def load_lora_model_withPeft(base_model, load_path):
    """加载 LoRA 模型"""
    return PeftModel.from_pretrained(base_model, load_path)

# 合并多个 LoRA 模型的权重
def merge_lora_weights(base_model, lora_models_paths, weights=None):
    """合并多个 LoRA 模型的权重"""
    if weights is None:
        weights = [1.0/len(lora_models_paths)] * len(lora_models_paths)
    if len(lora_models_paths) != len(weights):
        raise ValueError("模型路径和权重数量必须相同")
    
    # 加载第一个模型作为基础
    merged_model = load_lora_model_withPeft(base_model, lora_models_paths[0])
    
    # 获取第一个模型的权重并应用权重
    state_dict = merged_model.state_dict()
    for key in state_dict:
        if "lora" in key:
            state_dict[key] = state_dict[key] * weights[0]
    
    # 加载其他模型并合并权重
    for i in range(1, len(lora_models_paths)):
        model_path = lora_models_paths[i]
        weight = weights[i]
        
        # 临时加载模型
        temp_model = load_lora_model_withPeft(base_model, model_path)
        temp_state_dict = temp_model.state_dict()
        
        # 合并权重
        for key in temp_state_dict:
            if "lora" in key and key in state_dict:
                state_dict[key] += temp_state_dict[key] * weight
    
    # 将合并后的权重加载到模型中
    merged_model.load_state_dict(state_dict)
    return merged_model

# 推理
def inference(model, tokenizer, prompt):
    """使用模型进行推理
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示文本
    Returns:
        str: 生成的文本（不包含原始提示）
    """
    # 获取模型的第一个设备
    first_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(first_device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,  # 使用采样
            temperature=0.7,  # 控制随机性
            top_p=0.9,       # 核采样参数
        )
    
    # 只返回新生成的内容（去除输入提示）
    prompt_length = len(inputs["input_ids"][0])
    generated_sequence = outputs[0][prompt_length:]
    
    return tokenizer.decode(generated_sequence, skip_special_tokens=True)

# 主函数
def main():
    # 使用函数加载配置，而不是在main中重复代码
    config = load_config("tests/loraTrain/config.json")
    
    # 模型路径 - 使用 Mistral MoE 模型
    model_name = config["model_name"]
    
    # 定义目标模块
    target_modules = config["target_modules"]
    target_layers = config["target_layers"]
    save_path_base = config["save_path_base"]
    
    # 添加训练参数从配置中读取
    num_models = config.get("num_models", 3)
    weights = config.get("model_weights", [0.4, 0.3, 0.3])
    num_epochs = config.get("num_epochs", 10)
    learning_rate = config.get("learning_rate", 1e-3)
    
    # 确保权重数量与模型数量匹配
    if len(weights) != num_models:
        print(f"警告: 权重数量({len(weights)})与模型数量({num_models})不匹配，使用均等权重")
        weights = [1.0/num_models] * num_models
    
    # 确保保存路径存在
    os.makedirs(save_path_base, exist_ok=True)
    
    # 定义设备映射
    device_map = config.get("device_map", "auto")
    
    # 加载基础模型
    base_model, tokenizer = load_base_model(model_name, device_map)
    
    # 创建 LoRA 配置
    lora_config = create_lora_config(
        target_modules, 
        layers_to_transform=target_layers,
        r=config.get("lora_r", 8),
        alpha=config.get("lora_alpha", 16)
    )
    
    # 创建多个 LoRA 模型
    lora_models = []
    lora_model_paths = []
    
    for i in range(num_models):
        print(f"创建并训练 LoRA 模型 {i+1}/{num_models}")
        lora_model = create_lora_model(base_model, lora_config)
        lora_model = train_lora_model(
            lora_model, 
            tokenizer, 
            num_epochs=num_epochs, 
            learning_rate=learning_rate
        )[1]
        
        # 保存模型
        save_path = f"{save_path_base}/lora_model_{i}"
        save_lora_model(lora_model, save_path)
        lora_model_paths.append(save_path)
        lora_models.append(lora_model)
    
    # 合并 LoRA 模型
    print(f"合并 {num_models} 个LoRA模型，权重为: {weights}")
    merged_model = merge_lora_weights(base_model, lora_model_paths, weights)
    
    # 保存合并后的模型
    merged_save_path = f"{save_path_base}/merged_lora_model"
    save_lora_model(merged_model, merged_save_path)
    print(f"合并模型已保存到: {merged_save_path}")
    
    # 加载合并后的模型进行推理
    test_prompt = config.get("test_prompt", "Hello, how are you?")
    loaded_model = load_lora_model_withPeft(base_model, merged_save_path)
    result = inference(loaded_model, tokenizer, test_prompt)
    print(f"测试提示: {test_prompt}")
    print(f"推理结果: {result}")

if __name__ == "__main__":
    main()