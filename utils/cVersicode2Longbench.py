import json
from datasets import Dataset

def edit_data():
    # Read the original data
    with open('benchmark/data/VersiCode_Benchmark/blockcode_completion.json', 'r') as f:
        data = json.load(f)

    # Create the projection
    projection = []
    for item in data:
        projected_item = item.copy()
        projected_item["answers"] = "code"
        projected_item["_id"] = item["id"]
        projected_item["context"] = ""
        # remove "id" and "masked_code"
        del projected_item["id"]
        del projected_item["masked_code"]
        projection.append(projected_item)
        
    # Output the projection
    # print(json.dumps(projection, indent=2))

    # Optionally save to a file
    with open('data/versicode.json', 'w') as f:
        json.dump(projection,  f)
def convert2dataset(data_path,target_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(target_path)
if __name__ == "__main__":
    # edit_data()
    convert2dataset('data/versicode.json', 'benchmark/data/longbench/versicode')
    