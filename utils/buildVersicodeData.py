import json
data_base = "benchmark/data/VersiCode_Benchmark/"
blockcompletion_data_paths = [
    "code_completion/downstream_application_code/downstream_application_code_block.json",
    "code_completion/library_source_code/library_source_code_block.json",
    "code_completion/stackoverflow/python/stackoverflow_block.json",
]
edit_data_paths = [
    "code_editing/code_editing_new_to_old.json",
    "code_editing/code_editing_old_to_new.json",
]

# 将blockcompletion数据进行组合
blockcompletion_data = []
for data_path in blockcompletion_data_paths:
    with open(data_base + data_path, "r") as f:
        data = json.load(f)
    for data_piece in data["data"]:
        blockcompletion_data.append(data_piece)

# 将edit数据进行组合
edit_data = []
for data_path in edit_data_paths:
    with open(data_base + data_path, "r") as f:
        data = json.load(f)
    for data_piece in data["data"]:
        edit_data.append(data_piece)


# 写入base路径
with open(data_base + "blockcode_completion.json", "w") as f:
    json.dump(blockcompletion_data, f)

with open(data_base + "code_editing.json", "w") as f:
    json.dump(edit_data, f)



