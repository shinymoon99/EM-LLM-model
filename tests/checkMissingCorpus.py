from benchmark.config.code.config import CORPUS_PATH
from utils.getVersion import remove_prefix,getBestMatchVersion
import json
def find_missing_versions():
    required_pkg_versions = {}
    with open("benchmark/data/Versicode_Benchmark/code_completion.json","r") as f:
        data = json.load(f)
        for item in data:
            pkg = item["dependency"]
            version = remove_prefix(item["version"])
            if required_pkg_versions.get(pkg) is None:
                required_pkg_versions[pkg] = []
            if all(char not in version for char in ['<', '>', 'x']):
                required_pkg_versions[pkg].append(version)
    missing_pkg_versions = {}
    for pkg,versions in required_pkg_versions.items():
        for version in versions:
            best_match_version = getBestMatchVersion(pkg,version)
            if best_match_version is None:
                if missing_pkg_versions.get(pkg) is None:
                    missing_pkg_versions[pkg] = []
                missing_pkg_versions[pkg].append(version)
    # 
    missing_pkg_versions = {pkg: list(set(versions)) for pkg, versions in missing_pkg_versions.items() if versions}
    with open("data/missing_pkg_versions.json","w") as f:
        json.dump(missing_pkg_versions,f)

# 获取所有符合prefix要求且在corpus中存在的版本
def getMatchedData(data):
    '''
    Description:
        获取所有符合prefix要求且在corpus中存在的版本
    Args:
        data: list,数据集
    Returns:
        matched_data: list,符合prefix要求且在corpus中存在的版本
    '''
    matched_data = []
    for item in data:
        pkg = item["dependency"]
        version = remove_prefix(item["target_version"])
        if all(char not in version for char in ['<', '>', 'x']):
            if getBestMatchVersion(pkg,version) is not None:
                matched_data.append(item)
    return matched_data
if __name__ == "__main__":
    with open("benchmark/data/Versicode_Benchmark/code_editing_origin.json","r") as f:
        data = json.load(f)    
    print("len(data):",len(data))
    matched_data = getMatchedData(data)
    print("len(matched_data):",len(matched_data))
    with open("benchmark/data/Versicode_Benchmark/code_editing.json","w") as f:
        json.dump(matched_data,f)