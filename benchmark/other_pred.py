from together import Together
from datasets import load_from_disk
import json
import chromadb
import os
from sentence_transformers import SentenceTransformer
import re
import tiktoken
from tqdm import tqdm
from benchmark.config.code.config import CORPUS_PATH,VSCC_LOW_BOUND,VSCC_HIGH_BOUND,RAG_DOCUMENT_NUM,FC_MAX_TOKEN_LENGTH,RAG_MAX_TOKEN_LENGTH
with open("/mnt/d/API_KEYSET/togetherai.txt", "r") as f:
    api_key = f.read()

client = Together(api_key=api_key)

# 初始化 Chroma 客户端（持久化版）使用RAG

os.makedirs("chroma_data/library", exist_ok=True)  # 确保目录存在
_client = chromadb.PersistentClient(path="chroma_data/library")  # 持久化存储
# print(response.choices[0].message.content)

# -----------utils---------- #
def get_version(text):
    # 删除prefix >= > <= < ==
    text = re.sub(r"^>=", "", text)
    text = re.sub(r"^<=", "", text)
    text = re.sub(r"^==", "", text)
    return text
def truncate_context(context, max_token_length=31000):
    # obtain tokenizer
    encoding = tiktoken.get_encoding("gpt2")
    disallowed_special = ()
    tokens = encoding.encode(context, disallowed_special=disallowed_special)
    # print(len(tokens))

    if len(tokens) > max_token_length:
        tokens = tokens[:max_token_length]

    truncated_text = encoding.decode(tokens)

    return truncated_text
def truncate_BCB_context(context, max_token_length=31000):
    '''
    Description:
        截断BCB的context,每个package分配max_token_length/len(context)的token
    Args:
        context: dict[pkg,list[str]],知识文档
        max_token_length: int,最大token长度
    Returns:
        context: str,截断后的context
    '''
    # obtain tokenizer
    encoding = tiktoken.get_encoding("gpt2")
    disallowed_special = ()
    # 计算每个pkg的token数量，依据行数分配token量
    pkg_tokens_count = {}
    pkg_tokens = {}
    pkg_distribute_tokens = {}
    for pkg,docs in context.items():
        pkg_tokens[pkg] = [encoding.encode(doc, disallowed_special=disallowed_special) for doc in docs]
        pkg_tokens_count[pkg] = len(pkg_tokens[pkg])
    all_tokens_count = sum(pkg_tokens_count.values())
    # 计算token数目
    for pkg in pkg_tokens_count:
        pkg_distribute_tokens[pkg] = int(pkg_tokens_count[pkg]/all_tokens_count*max_token_length)
    # 按比例划分token
    for pkg in pkg_tokens_count:
        pkg_tokens[pkg] = pkg_tokens[pkg][:pkg_distribute_tokens[pkg]]
    # 将token转换为字符串
    context_str = ""
    for pkg in pkg_tokens:
        context_str += pkg + "\n" + "\n".join(encoding.decode(token) for token in pkg_tokens[pkg]) + "\n"
    return context_str
def getKnowledgeDocs(data,corpus_base=CORPUS_PATH,dataset="versicode",task="VSCC"):
    '''
    Description:
        获取知识文档字符串
    Args:
        data: dict,数据
    Returns:
        knowledge_doc: list,知识文档 if dataset=="versicode"
        knowledge_doc: dict,知识文档 if dataset=="versiBCB" dict[pack:list]。如果是sourcecode，list中每个元素代表一个文件的内容；如果是doc,则需要我们再思考切分
    '''
    if dataset == "versicode":
        dependency = data["dependency"] if task == "VSCC" else data["target_dependency"]
        version = get_version(data["version"]) if task == "VSCC" else get_version(data["target_version"])
        
        #TODO:加入match_most_similar ，以解决1.7这类不包括patch的版本
        corpus_path = os.path.join(corpus_base, dependency,version + ".jsonl")
        knowledge_docs = []
        try:
            with open(corpus_path, "r") as f:
                for line in f:
                    # 
                    # item = json.loads(line)
                    knowledge_docs.append(line)   
        except Exception as e:
            print(f"Error: {e}")
            print(f"Corpus path: {corpus_path}")
            return []

    elif dataset == "versiBCB":
        dependency = data["dependency"] if task == "VSCC" else data["target_dependency"]
        for pack,version in dependency.items():
            all_corpus_path = os.path.join(corpus_base, pack,version + ".jsonl")
            knowledge_docs[pack] = []
            try:
                with open(all_corpus_path, "r") as f:
                    for line in f:
                        # item = json.loads(line)
                        knowledge_docs[pack].append(line)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Corpus path: {all_corpus_path}")
                return []
        return knowledge_docs
    else:
        raise "Wrong dataset"
    return knowledge_docs
def appendContextToData(data):
    '''
    Description:
        当context为空时，将context添加到data中
    Args:
        data: dict,数据
    Returns:
        data: dict,数据
    '''
    if data["context"] == "":
        knowledge_docs = getKnowledgeDocs(data)
        if isinstance(knowledge_docs,dict):
            data["context"] = "\n".join(knowledge_docs[pack] for pack in knowledge_docs)
        else:
            data["context"] = "\n".join(knowledge_docs)
    return data


# --------RAG utils-------- #
def generate_embedding(text):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text).tolist()
    return embedding
def get_datacollectionLocation(data):
    base_dir = "chroma_data/library"
    library_name = data["dependency"]
    version = get_version(data["version"])
    file_path = os.path.join(base_dir, library_name, version + '.jsonl')
    return file_path
def retrieve_RAGContext(data, query,dataset):
    '''
    Description:
        根据data和query获取RAG的context
    Args:
        data: dict,数据
    '''
    dep,ver = data["dependency"],get_version(data["version"])
    collection_name = f"{dep}_{ver}"
    collection = _client.get_or_create_collection(name=collection_name)
    if dataset == "versicode":
        documents = getKnowledgeDocs(data,dataset)
    elif dataset == "versiBCB":
        documents = getKnowledgeDocs(data,dataset)
        documents = [doc for pack in documents for doc in documents[pack]]
    else:
        raise "Wrong dataset"

    # 添加文档到临时集合
    if collection.count() == 0 and len(documents) > 0:
        ids = [f"{collection_name}_{idx}" for idx in range(len(documents))]
        collection.add(documents=documents, ids=ids)
    
    # 执行查询
    query_embedding = [generate_embedding(query)]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=RAG_DOCUMENT_NUM,
    )
    retrieved_context = ""
    # 获取最相关的文档
    if results["documents"] and results["documents"][0]:
        retrieved_context = "\n".join([doc for doc in results["documents"][0]])
    return retrieved_context

# --------entry point--------- #
def get_FC_pred(data, prompt,max_token_length=FC_MAX_TOKEN_LENGTH,task="VSCC",dataset="versicode"):
    if dataset == "versicode":
        context = getKnowledgeDocs(data,dataset,task)
        context = "\n".join(context)
        context = truncate_context(context, max_token_length)    
        # 根据prompt和data获取input
    elif dataset == "versiBCB":
        context = getKnowledgeDocs(data,dataset,task)
        context = truncate_BCB_context(context, max_token_length)
    else:
        raise "Wrong dataset"
    if task == "VSCC":
        input_for_api = prompt.format(knowledge_doc=context, description=data["description"],dependency=data["dependency"],version=get_version(data["version"]))
    elif task == "VACE":
        input_for_api = prompt.format(knowledge_doc=context, description=data["description"],dependency=data["dependency"],origin_version=get_version(data["origin_version"]),origin_code=data["origin_code"],target_version=get_version(data["target_version"]))
    else:
        raise "Wrong task"
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": input_for_api}],
    )
    return response.choices[0].message.content

def get_RAG_pred(data, prompt,max_token_length=RAG_MAX_TOKEN_LENGTH,task="VSCC",dataset="versicode"):
    if task == "VSCC":
        query = data["description"]
        context = retrieve_RAGContext(data, query,dataset,task)
        context = truncate_context(context, max_token_length)

        input_for_api = prompt.format(knowledge_doc=context, description=data["description"],dependency=data["dependency"],version=get_version(data["version"]))
    elif task == "VACE":
        query = data["description"]+data["origin_code"]
        context = retrieve_RAGContext(data, query,dataset,task)
        context = truncate_context(context, max_token_length)

        input_for_api = prompt.format(knowledge_doc=context, description=data["description"],dependency=data["dependency"],origin_version=get_version(data["origin_version"]),origin_code=data["origin_code"],target_version=get_version(data["target_version"]))
    else:
        raise "Wrong task"
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": input_for_api}],
    )
    return response.choices[0].message.content
if __name__ == "__main__":

    dataset = "versicode"
    from benchmark.config.code.dataset2prompt import dataset2prompt
    versicode_prompt = dataset2prompt["versicode"]
    with open("benchmark/data/VersiCode_Benchmark/blockcode_completion.json", "r") as f:
        datas = json.load(f)
    output_FC = []
    output_RAG = []
    for i, data in tqdm(enumerate(datas)):
        if i<VSCC_LOW_BOUND or i>=VSCC_HIGH_BOUND:
            continue
        # 获取FC的预测
        FC_pred = get_FC_pred(data,versicode_prompt["vscc"])
        # 获取RAG的预测
        RAG_pred = get_RAG_pred(data,versicode_prompt["vscc"])
        # output_FC.append({"id": data["id"], "answer": FC_pred, "ground_truth": data["code"]})
        output_RAG.append({"id": data["id"], "answer": RAG_pred,"ground_truth": data["code"]})
        # with open("benchmark/data/longbench/versicode_FC.json", "w") as f:
        #     json.dump(output_FC, f)
        with open("benchmark/data/longbench/versicode_RAG.json", "w") as f:
            json.dump(output_RAG, f)