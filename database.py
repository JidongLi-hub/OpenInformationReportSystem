import pymilvus
from pymilvus import MilvusClient
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import requests
from format import get_chunks_from_markdown



class VectorDatabase:
    def __init__(self, collection_name="chunk",embedding_dim=1024):
        self.base_url = "http://localhost:7979/v1"
        self.api_key = "EMPTY"
        if not self._check_server():    
            raise RuntimeError(
                f"请先在终端启动模型，执行如下脚本：\n\n"
                "bash scripts/start_embedding_model.sh\n\n"
            )
        self.openai_client = OpenAI(  # 通过vLLM部署Embedding模型
            base_url=self.base_url,
            api_key=self.api_key,
        )
        print(f"[INFO] 成功连接到 vLLM 服务 {self.base_url}")

        self.database = "./database/OIR.db"
        self.milvus_client = MilvusClient(uri=self.database)
        if not self.milvus_client.has_collection(collection_name):
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,
                metric_type="IP",  # Inner product distance
                consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
            )
        self.collect_name = collection_name
        # 查看当前集合有多少条数据
        count = self.milvus_client.get_collection_stats(collection_name)["row_count"]
        print(f"[INFO] 当前集合 {collection_name} 中已有 {count} 条数据")
        self.id_now = count
        print(f"[INFO] 成功连接到 Milvus 数据库 {self.database}")

    def _check_server(self):
        """检查 vLLM 是否启动"""
        health_url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.get(health_url, headers=headers, timeout=2)
            return r.status_code == 200
        except Exception as e:
            print(f"[ERROR] 无法连接到 vLLM 服务 {self.base_url}，错误信息如下：\n{e}")
            return False
   
    def process_a_file(self, file_path):
        """读取一个文档，并根据格式进行相应的处理，返回原始文本"""
        # text = "First, vLLM must be installed for your chosen device in either a Python or Docker environment.Then, vLLM supports the following usage patterns:Inference and Serving: Run a single instance of a model.Deployment: Scale up model instances for production.Training: Train or fine-tune a model."
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # 切分文本得到块，并插入到数据库中
        chunks = get_chunks_from_markdown(text)
        for chunk in chunks:
            embed = self.embedding(chunk)
            self.insert_embedding(
                {
                    "id": self.id_now, 
                    "type": "chunk", # chunk or summary
                    "file_name": os.path.basename(file_path),
                    "vector": embed, 
                    "text": chunk
                }
            )
            self.id_now += 1
        print(f"[INFO] 成功处理文档 {file_path}，共切分为 {len(chunks)} 个块并插入到数据库中, 当前总计 {self.id_now} 条数据")

        # 存储整个文档的摘要，待完成


    def process_files(self, file_list):
        """批量处理文档"""
        for file_path in tqdm(file_list, desc="处理文档"):
            self.process_a_file(file_path)



    def embedding(self, text, embedding_model="/data2/home/lijidong/models/bge-m3"):
        try:
            res = self.openai_client.embeddings.create(input=text, model=embedding_model)
        except Exception as e:
            print(f"failed to get embedding for: {e}")
            return None
        return res.data[0].embedding

    def insert_embedding(self, data):
        """将文档的嵌入向量插入到Milvus数据库中"""
        self.milvus_client.insert(
            collection_name=self.collect_name,
            data=data
        )
        # print(f"[INFO] 成功插入文档到 Milvus 数据库")

    def search_embedding(self, query_text, top_k=3):
        """在Milvus数据库中搜索与查询嵌入向量最相似的文档，返回相关文本的列表"""
        res = self.milvus_client.search(
            collection_name=self.collect_name,
            data=[self.embedding(query_text)],
            limit=top_k,
            output_fields=["text"]
        )
        data = res[0]
        chunks = []
        for dic in data:
            chunks.append(dic.entity.get("text"))
        return chunks
    

if __name__ == "__main__":
    # insert test
    db = VectorDatabase(collection_name="chunk")
    root_path = "/data2/home/lijidong/vllm-qwen/datafiles/guowu_data/America/downloadsnew/"
    sub_paths = os.listdir(root_path)
    sub_paths = [os.path.join(root_path, path) for path in sub_paths if os.path.isdir(os.path.join(root_path, path))]
    for sub_path in sub_paths:
        files = os.listdir(os.path.join(sub_path,"epub"))
        md_files = [file for file in files if file.endswith('.md')]
        md_files = [os.path.join(sub_path,"epub", file) for file in md_files]
        db.process_files(md_files)

    # 向量数据库检索示例
    # db = VectorDatabase()
    # query = "Citing Foreign Affairs Records" # 修改为你想查询的内容
    # res = db.search_embedding(query, top_k=3)  # 返回与查询最相似的文本块，是一个字符串组成的列表。调节 top_k 可以控制返回结果的数量
    # print(json.dumps(res))




