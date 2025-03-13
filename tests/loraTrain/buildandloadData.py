import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.input_ids = self._tokenize_and_chunk(texts)

    def _tokenize_and_chunk(self, texts):
        textid2inputids = [] # list of inputids,每
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            textid2inputids.append(self.tokenizer.convert_tokens_to_ids(tokens))
        # 按 block_size 分块
        text_chunks = []
        for textid2inputid in textid2inputids:
            chunks = [
                textid2inputid[i : i + self.block_size]
                for i in range(0, len(textid2inputid), self.block_size)
            ]
            text_chunks.extend(chunks)
        return text_chunks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        labels = input_ids[1:] + [self.tokenizer.eos_token_id]  # 下一个 token 作为 labels
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
if __name__ == "__main__":
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

    # 打印示例
    for batch in dataloader:
        print(batch)
        break