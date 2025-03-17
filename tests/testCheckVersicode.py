import json
from benchmark.other_pred import get_version
from benchmark.config.code.config import CORPUS_PATH
def getContext(data):
    pkg = data["dependency"]
    version = get_version(data["version"])
    files_info = []
    try:
        with open(f"{CORPUS_PATH}/{pkg}/{version}.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                files_info.append(str(line_data))  
    except Exception as e:
        return ""
    return "".join(files_info)  
with open("data/da_block_originprompt_lscContext.json", "r") as f:
    datas = json.load(f)
contexts_length = []
original_contexts_length = []
for data in datas:
    if len(data["context"])!=0:
        contexts_length.append(len(data["context"]))
        original_contexts_length.append(len(getContext(data)))
print(contexts_length)
print(len(contexts_length))
print(sum(contexts_length)/len(contexts_length))
print(original_contexts_length)
print(len(original_contexts_length))
print(sum(original_contexts_length)/len(original_contexts_length))
