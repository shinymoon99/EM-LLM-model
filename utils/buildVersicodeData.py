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
def cCEdata2UniFormat(data):
    uniform_data = data.copy()
    is_old_to_new = "old_to_new" in data["id"]

    # 设置统一格式的字段

    uniform_data["origin_code"] = data["old_code"] if is_old_to_new else data["new_code"]
    uniform_data["target_code"] = data["new_code"] if is_old_to_new else data["old_code"]
    uniform_data["origin_version"] = data["old_version"] if is_old_to_new else data["new_version"]
    uniform_data["target_version"] = data["new_version"] if is_old_to_new else data["old_version"]
    uniform_data["origin_time"] = data["old_time"] if is_old_to_new else data["new_time"]
    uniform_data["target_time"] = data["new_time"] if is_old_to_new else data["old_time"]
    uniform_data["origin_name"] = data["old_name"] if is_old_to_new else data["new_name"]
    uniform_data["target_name"] = data["new_name"] if is_old_to_new else data["old_name"]

    # 删除old和new这些栏目
    for key in ["old_code", "new_code", "old_version", "new_version", "old_time", "new_time", "old_name", "new_name"]:
        uniform_data.pop(key, None)

    return uniform_data
# 将blockcompletion数据进行组合
def mergeBlockCompletionData():
    blockcompletion_data = []
    for data_path in blockcompletion_data_paths:
        with open(data_base + data_path, "r") as f:
            data = json.load(f)
        for data_piece in data["data"]:
            blockcompletion_data.append(data_piece)
    # 写入base路径
    with open(data_base + "blockcode_completion.json", "w") as f:
        json.dump(blockcompletion_data, f)
def mergeCodeEditingData():
    # 将edit数据进行组合
    edit_data = []
    for data_path in edit_data_paths:
        with open(data_base + data_path, "r") as f:
            data = json.load(f)
        for data_piece in data["data"]:
            edit_data.append(cCEdata2UniFormat(data_piece))

    with open(data_base + "code_editing.json", "w") as f:
        json.dump(edit_data, f)
if __name__ == "__main__":
    mergeCodeEditingData()



