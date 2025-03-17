import torch
import os
from peft import get_peft_model,PeftModel
import json
from tests.loraTrain.peftMoElora import load_base_model,merge_lora_weights,inference,save_lora_model,load_lora_model_withPeft,train_lora_model,load_config,create_lora_config
from tests.loraTrain.buildandloadData import TextDataset
from torch.utils.data import DataLoader
from benchmark.other_pred import get_version
from benchmark.config.code.config import CORPUS_PATH
from tqdm import tqdm
import warnings
from utils.loraPathConfigure import pathConfigurator
# warnings.filterwarnings("error",category=UserWarning)
def get_lora_pred(data,dependencies):
    '''
    Description:
        获取lora预测结果。
        1.尝试load对应package的lora权重，若没有，则进行训练，获得对应权重
        2.使用训练好的lora权重进行推理
    Args:
        data: dict,数据
        dependencies: list,依赖
    Returns:
        lora_pred: str,lora预测结果
    '''
    config = load_config("tests/loraTrain/config.json")
    base_model, tokenizer = load_base_model(config.get("model_name"), config.get("device_map"))
    lora_models_path = {}
    for pkg,version in dependencies.items():
        try:
            lora_model_path = load_lora_model(pkg,version,config)
            lora_models_path[pkg] = lora_model_path
        except Exception as e:
            dataloader = get_dataloader(config,pkg,version,tokenizer)
            lora_model=create_lora_model(config,pkg,version,dataloader)
            pathConfig = pathConfigurator()
            path = pathConfig.getPath(config,pkg,version)
            lora_model.save_pretrained(path)
            lora_models_path[pkg] = path
        
    lora_pred = combineLoraWeights_and_predict(data,base_model,tokenizer,lora_models_path,config)
    return {"id":data["id"],"answer":lora_pred}
def get_dataloader(config,pkg,version,tokenizer):
    files_info = []
    with open(f"{CORPUS_PATH}/{pkg}/{version}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line)
            files_info.append(str(line_data))    
    files_info = files_info[:int(len(files_info)*config["traindata_percentage"])]
    if len(files_info) == 0:
        return None
    print(f"{pkg} {version} {len(files_info)}")
    dataset = TextDataset(files_info, tokenizer, block_size=128,pkg=pkg,version=version)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    return dataloader
def load_lora_model(pkg,version,config):
    '''
    Description:
        加载lora模型
    Args:
        pkg: str,包名
        version: str,版本号
        config: dict,配置
    Returns:
        lora_model_path:str,lora模型路径
    '''
    pathConfig = pathConfigurator()
    path = pathConfig.getPath(config,pkg,version)
    if os.path.exists(path):
        with open(path,"rb") as f:
            lora_model = torch.load(f)
    else:
        raise ValueError(f"lora模型不存在: {path}")
    return path
def create_lora_model(config,pkg,version,dataloader):
    '''
    Description:
        训练lora模型
    Args:
        pkg: str,包名
        version: str,版本号
    Returns:
        lora_model_path:str,lora模型路径
    '''
    base_model, tokenizer = load_base_model(config["model_name"], config["device_map"])
    lora_config = create_lora_config(config["target_modules"], config["target_layers"], config["r"], config["alpha"])
    lora_model = get_peft_model(base_model, lora_config)

    lora_model = train_lora_model(lora_model, tokenizer,dataloader, config["num_epochs"], config["learning_rate"])
    
    return lora_model

def combineLoraWeights_and_predict(data,base_model,tokenizer,lora_models_path,config):
    #TODO:加入指定data，构建input
    lora_models_path_list = list(lora_models_path.values())
    merged_model = merge_lora_weights(base_model, lora_models_path_list)
    # 保存合并后的模型
    save_path_base = config["save_path_base"]
    merged_save_path = f"{save_path_base}/merged_lora_model"
    save_lora_model(merged_model, merged_save_path)
    print(f"合并模型已保存到: {merged_save_path}")
    
    # 加载合并后的模型进行推理
    if config["dataset"]=='versicode':
        input_prompt= config.get("versicode_vscc_prompt") if config["task"]=='vscc' else config.get("versicode_vace_prompt")
        if config["task"]=='vscc':
            # vscc
            input = input_prompt.format(description=data["description"],dependency=data["dependency"],version=get_version(data["version"]))
        elif config["task"]=='vace':
            # vace
            input = input_prompt.format( description=data["description"],dependency=data["dependency"],origin_version=get_version(data["origin_version"]),origin_code=data["origin_code"],target_version=get_version(data["target_version"]))

    else: # versiBCB
        if config["task"]=='vscc':
            input = config.get("versicode_vscc_prompt").format(description=data["description"],dependency=data["dependency"])
        elif config["task"]=='vace':
            input = config.get("versicode_vace_prompt").format( description=data["description"],dependency=data["origin_dependency"],origin_code=data["origin_code"],target_version=get_version(data["target_dependency"]))
    loaded_model = PeftModel.from_pretrained(base_model, merged_save_path)
    result = inference(loaded_model, tokenizer, input)
    # print(f"测试提示: {input_prompt}")
    # print(f"推理结果: {result}")
    return result
# ----------utils----------
def getDataExistence(dependency):
    '''
    Description:
        判断数据是否存在
    Args:
        data: dict,数据
    '''
    for pkg,version in dependency.items():
        files_info = []
        try:
            with open(f"{CORPUS_PATH}/{pkg}/{version}.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line_data = json.loads(line)
                    files_info.append(str(line_data))    
        except Exception as e:
            return False
        if len(files_info) == 0:
            return False
    return True
def main():

    config = load_config("tests/loraTrain/config.json")
    lora_pred_result = []
    benchmark = "Versicode_Benchmark" if config["dataset"] == "versicode" else "VersiBCB_Benchmark"
    filename = "code_completion" if config["task"] == "vscc" else "code_editing"
    with open(f"benchmark/data/{benchmark}/{filename}.json", "r") as f:
        datas = json.load(f)
    for data in tqdm(datas):    
        if config["dataset"] == "versicode":
            pack = data["dependency"]
            version = get_version(data["version"]) if config["task"] == "vscc" else get_version(data["target_version"])
            dependencies = {pack:version}
            if not getDataExistence(dependencies):
                continue
        elif config["dataset"] == "versiBCB": # versiBCB
            dependencies = data["target_dependency"]
        else:
            raise ValueError(f"数据集不存在: {config['dataset']}")
        lora_pred = get_lora_pred(data,dependencies)
        lora_pred_result.append(lora_pred)
        # 每训练一个写一次
        with open(f"output/{benchmark}/{filename}_lora_pred_result.json", "w", encoding="utf-8") as f:
            json.dump(lora_pred_result, f, ensure_ascii=False)
if __name__ == "__main__":
    main()
