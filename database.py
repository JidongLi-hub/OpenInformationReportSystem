import pymilvus
from pymilvus import MilvusClient
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import requests



class VectorDatabase:
    def __init__(self, embedding_dim, collection_name="OIR"):
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
        text = "First, vLLM must be installed for your chosen device in either a Python or Docker environment.Then, vLLM supports the following usage patterns:Inference and Serving: Run a single instance of a model.Deployment: Scale up model instances for production.Training: Train or fine-tune a model."
        embed = self.embedding(text)
        self.insert_embedding(
            {
                "id": 0, 
                "vector": embed, 
                "text": text
            }
        )


    def process_files(self, file_list):
        """批量处理文档"""

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

    def search_embedding(self, query_text, top_k=3):
        """在Milvus数据库中搜索与查询嵌入向量最相似的文档，返回相关文本的列表"""
        res = self.milvus_client.search(
            collection_name=self.collect_name,
            data=[self.embedding(query_text)],
            limit=top_k,
            output_fields=["text"]
        )
        return res
    

if __name__ == "__main__":
    db = VectorDatabase(embedding_dim=1024)
    db.process_a_file("vllm-qwen/scripts/start_embedding_model.sh")


