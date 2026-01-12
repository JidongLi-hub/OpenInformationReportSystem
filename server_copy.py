import os
import re
import socket
import uvicorn
import requests
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 1. 尝试导入同目录下的数据库模块
# ==========================================
try:
    from database_v2 import HierarchicalVectorDatabase 
    DB_MODULE_AVAILABLE = True
except ImportError:
    print(f"[Server] ⚠️ 导入 database 模块失败，将降级运行。")
    DB_MODULE_AVAILABLE = False
    VectorDatabase = None

# ==========================================
# 2. 全局配置
# ==========================================

# 【核心修改】直接写死这个冷门端口 28888
VLLM_PORT = 28888 
CHAT_API_URL = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
CHAT_MODEL_NAME = "/home/models/Qwen2-7B-Instruct"

# 初始化全局数据库实例
GLOBAL_DB = None
if DB_MODULE_AVAILABLE:
    try:
        print("[Server] 正在初始化向量数据库服务...")
        GLOBAL_DB = HierarchicalVectorDatabase()
        print("[Server] ✅ 向量数据库服务就绪")
    except Exception as e:
        print(f"[Server] ⚠️ 数据库实例初始化异常: {e}")
        GLOBAL_DB = None

# ==========================================
# 3. FastAPI 应用初始化
# ==========================================
app = FastAPI(title="态势报告生成系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRequest(BaseModel):
    user_prompt: str

# ==========================================
# 4. 核心业务接口
# ==========================================

@app.post("/generate_report")
async def generate_report(request: UserRequest):
    print(f"[Server] 接收分析指令: {request.user_prompt}")
    
    try:
        # --- 阶段一：情报检索 ---
        retrieved_docs = []
        if GLOBAL_DB:
            try:
                retrieved_docs = GLOBAL_DB.search_with_context(request.user_prompt, top_k=3)
                """
                retrieved_docs是一个列表，格式如下：
                {
                "matched_chunk": 匹配到的文本片段,
                "score": 相似度分数,
                "file_name": 来源文件名（需要在网页的第二页展示来源）,
                "context": 完整上下文信息，这个会提供给LLM用来生成报告
                }
                """
            except Exception as e:
                print(f"[Server] ⚠️ 检索异常: {e}")
        
        if retrieved_docs:
            print(f"[Server] ✅ 检索完成，召回 {len(retrieved_docs)} 条数据。")
            retrieved_docs_context = [doc["context"] for doc in retrieved_docs]
            context_str = "\n\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs_context)])
        else:
            print("[Server] ⚠️ 未检索到数据，使用模拟数据。")
            context_str = "1. [模拟] 无线电信号异常增强。\n2. [模拟] 气象海况恶劣。\n3. [模拟] 历史同期有演练。"
        
        # --- 阶段二：构建 Prompt ---
        final_prompt = f"""
        你是一位专业的国家安全态势分析员。请根据以下【背景情报】对【用户指令】进行深度分析，撰写一份专业的态势报告。
        
        【背景情报】：
        {context_str}
        
        【用户指令】：
        {request.user_prompt}
        
        【要求】：
        1. 必须基于情报事实进行推断，若情报不足请指出。
        2. 报告结构严格包含：摘要、现状分析、趋势预测。
        3. 使用 Markdown 格式，逻辑清晰，字数控制在 800 字以内。
        """

        # --- 阶段三：大模型推理 ---
        llm_content = ""
        try:
            print(f"[Server] 请求 Chat 模型 (端口 {VLLM_PORT})...")
            resp = requests.post(
                CHAT_API_URL, 
                json={
                    "model": CHAT_MODEL_NAME,
                    "messages": [{"role": "user", "content": final_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }, 
                proxies={"http": None, "https": None},
                timeout=60
            )
            resp.raise_for_status()
            print(resp.json())  # 这行很关键，貌似在等待响应完成，如果不加就会直接响应失败。后续可以考虑优化
            llm_content = resp.json()["choices"][0]["message"]["content"]
            print("[Server] ✅ 报告生成完成。")
            
        except Exception as e:
            print(f"[Server] ⚠️ Chat 模型调用失败: {e}")
            llm_content = f"> **⚠️ 系统提示**：大模型连接异常 (Port {VLLM_PORT})。\n\n**相关情报：**\n{context_str}"

        return {
            "status": "success",
            "original_query": request.user_prompt,
            "retrieved_info": context_str,
            "report_content": llm_content
        }

    except Exception as e:
        print(f"[Server] ❌ 内部错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 5. 静态资源托管
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = "index.html"
    if not os.path.exists(frontend_path):
        return "<h1>错误：未找到 index.html</h1>"
    
    with open(frontend_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(
        r'fetch\s*\(\s*["\']http://localhost:\d+/generate_report["\']', 
        'fetch("/generate_report"', 
        content
    )
    return HTMLResponse(content=content)

# 为了更安全，我们从一个冷门的高位端口开始找
DEFAULT_SERVER_PORT = 29001

def find_free_port(start_port=DEFAULT_SERVER_PORT, max_retries=100):
    port = start_port
    while port < start_port + max_retries:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                try:
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError: pass
        port += 1
    raise RuntimeError("无法分配端口")

if __name__ == "__main__":
    try:
        PORT = find_free_port(DEFAULT_SERVER_PORT)
        print("="*60)
        print(f"🚀 态势报告生成系统启动")
        print(f"🔗 访问地址: http://localhost:{PORT}")
        print(f"🔗 模型端口: {VLLM_PORT} (固定)")
        print("="*60)
        uvicorn.run(app, host="0.0.0.0", port=PORT)
    except Exception as e:
        print(f"❌ 启动失败: {e}")