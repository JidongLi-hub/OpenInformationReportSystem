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
# 1. 导入数据库接口模块
# ==========================================
try:
    # 导入本地 database 模块中的向量数据库类
    from database import VectorDatabase
    DB_MODULE_AVAILABLE = True
    print("[Server] ✅ 成功加载 database 模块")
except ImportError as e:
    print(f"[Server] ⚠️ 加载 database 模块失败: {e}")
    print("💡 提示: 请检查运行目录下是否存在 database.py 及其依赖库 (pymilvus, openai)。")
    DB_MODULE_AVAILABLE = False
    VectorDatabase = None

# ==========================================
# 2. 全局服务配置
# ==========================================

# 聊天模型 API 地址 (vLLM 服务端口 8002)
CHAT_API_URL = "http://localhost:8002/v1/chat/completions"
CHAT_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

# 初始化全局数据库实例
GLOBAL_DB = None
if DB_MODULE_AVAILABLE:
    try:
        print("[Server] 正在初始化向量数据库服务...")
        # 实例化数据库对象 (使用默认配置连接本地 Milvus)
        GLOBAL_DB = VectorDatabase()
        print("[Server] ✅ 向量数据库服务就绪")
    except Exception as e:
        print(f"[Server] ⚠️ 数据库实例初始化异常: {e}")
        print("💡 系统将降级运行：使用模拟数据响应请求，不影响服务启动。")
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
        # --- 阶段一：执行情报检索 ---
        retrieved_docs = []
        if GLOBAL_DB:
            try:
                print("[Server] 调用向量检索接口...")
                # 执行语义检索，获取 Top-3 相关文档片段
                retrieved_docs = GLOBAL_DB.search_embedding(request.user_prompt, top_k=3)
            except Exception as e:
                print(f"[Server] ⚠️ 检索过程发生异常: {e}")
        
        # 处理检索结果 (含降级策略)
        if retrieved_docs:
            print(f"[Server] ✅ 检索完成，召回 {len(retrieved_docs)} 条数据。")
            # 格式化上下文数据，添加序号以便 LLM 引用
            context_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)])
        else:
            print("[Server] ⚠️ 未检索到有效数据或服务不可用，切换至模拟数据模式。")
            context_str = (
                "1. [模拟数据] 监测发现目标海域无线电信号异常增强 15%。\n"
                "2. [模拟数据] 气象数据显示未来 48 小时内将有强对流天气。\n"
                "3. [模拟数据] 历史记录显示该区域常用于年度例行测试。"
            )
        
        # --- 阶段二：构建提示词 (Prompt) ---
        final_prompt = f"""
        你是一个专业的态势分析员。请根据以下【背景情报】对【用户指令】进行深度分析，撰写一份态势报告。
        
        【背景情报】：
        {context_str}
        
        【用户指令】：
        {request.user_prompt}
        
        【要求】：
        1. 必须基于情报事实进行推断。
        2. 报告包含：摘要、现状分析、趋势预测。
        3. 使用 Markdown 格式，字数控制在 600 字以内。
        """

        # --- 阶段三：执行大模型推理 ---
        llm_content = ""
        try:
            print(f"[Server] 请求 Chat 模型推理 (端口 8002)...")
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
            llm_content = resp.json()["choices"][0]["message"]["content"]
            print("[Server] ✅ 报告生成完成。")
            
        except Exception as e:
            print(f"[Server] ⚠️ Chat 模型调用失败: {e}")
            llm_content = f"> **⚠️ 系统提示**：大模型服务连接异常，仅展示检索到的情报。\n\n**相关情报如下：**\n{context_str}"

        return {
            "status": "success",
            "original_query": request.user_prompt,
            "retrieved_info": context_str,
            "report_content": llm_content
        }

    except Exception as e:
        print(f"[Server] ❌ 内部服务错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 5. 静态资源托管与端口管理
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = "index.html"
    if not os.path.exists(frontend_path):
        return "<h1>错误：未找到 index.html</h1><p>请确保前端文件部署在正确目录。</p>"
    
    with open(frontend_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 动态注入后端接口地址，适配当前运行端口
    content = re.sub(
        r'fetch\s*\(\s*["\']http://localhost:\d+/generate_report["\']', 
        'fetch("/generate_report"', 
        content
    )
    return HTMLResponse(content=content)

def find_free_port(start_port=8001, max_retries=100):
    port = start_port
    while port < start_port + max_retries:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                try:
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError: pass
        port += 1
    raise RuntimeError("无法分配可用端口，请检查网络设置。")

if __name__ == "__main__":
    try:
        PORT = find_free_port(8001)
        print("="*60)
        print(f"🚀 态势报告生成系统已启动")
        print(f"🔗 访问地址: http://localhost:{PORT}")
        print("="*60)
        uvicorn.run(app, host="0.0.0.0", port=PORT)
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")