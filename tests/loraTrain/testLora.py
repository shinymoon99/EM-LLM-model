import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer, AutoModelForCausalLM
from tests.loraTrain.buildandloadData import TextDataset
import json


# 定义LoRA层
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(original_layer.weight.shape[0], rank))
        self.lora_B = nn.Parameter(torch.randn(rank, original_layer.weight.shape[1]))

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = x @ self.lora_A @ self.lora_B
        return original_output + lora_output
if __name__ == "__main__":
    # --------加载数据--------
    # 加载文档
    files_info = []
    with open("/mnt/d/codes/Corpus/downstream_application_code/version_corpus/accelerate/0.15.0.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line)
            files_info.append(str(line_data))
            # print(str(line_data))

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)

    # 创建 Dataset 和 DataLoader
    dataset = TextDataset(files_info, tokenizer, block_size=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)    

    # --------训练-------
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    # 冻结模型的所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 选择FFN模块的第二层进行LoRA微调
    for i in range(25,26):  # 假设模型有26层
        layer = model.model.layers[i]
        ffn_second_layer = layer.mlp.fc2  # 假设FFN模块的第二层是fc2
        ffn_second_layer = LoRALayer(ffn_second_layer, rank=8)  # 假设rank为8


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 微调过程
    for epoch in range(1):  # 假设微调3个epoch
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 保存微调后的模型
    model.save_pretrained("lora_finetuned_deepseekcoder_v2_lite_base")