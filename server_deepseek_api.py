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
# 1. 安全配置加载
# ==========================================

def load_api_key():
    """
    尝试从环境变量或 .env 文件中读取 API Key
    """
    # 1. 优先从系统环境变量读取 (适合生产环境)
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key.strip()
    
    # 2. 尝试读取本地 .env 文件 (适合开发环境)
    env_file = ".env"
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 忽略注释和空行，查找 KEY=VALUE 格式
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "DEEPSEEK_API_KEY":
                            return value.strip().strip("'").strip('"') # 去除可能存在的引号
        except Exception as e:
            print(f"[Config] ⚠️ 读取 .env 文件出错: {e}")
            
    return None

# 加载密钥
DEEPSEEK_API_KEY = load_api_key()

# 检查密钥状态
if not DEEPSEEK_API_KEY:
    print("="*60)
    print("❌ 严重错误: 未找到 DEEPSEEK_API_KEY！")
    print("请在同级目录下创建 .env 文件，并写入: DEEPSEEK_API_KEY=sk-xxxx")
    print("="*60)
    # 这里不退出，而是允许程序启动，但在调用时报错，方便调试前端
else:
    print(f"[Config] ✅ 成功加载 API Key (长度: {len(DEEPSEEK_API_KEY)})")

# DeepSeek 官方 API 配置
CHAT_API_URL = "https://api.deepseek.com/chat/completions"
CHAT_MODEL_NAME = "deepseek-chat"

# ==========================================
# 2. 尝试加载本地数据库模块
# ==========================================
try:
    from database import VectorDatabase
    DB_MODULE_AVAILABLE = True
except ImportError:
    print(f"[Server] ⚠️ 未找到 database 模块，系统将以【纯模拟数据】模式运行。")
    DB_MODULE_AVAILABLE = False
    VectorDatabase = None

# 初始化全局数据库实例
GLOBAL_DB = None
if DB_MODULE_AVAILABLE:
    try:
        print("[Server] 正在尝试连接本地向量数据库...")
        GLOBAL_DB = VectorDatabase()
        print("[Server] ✅ 向量数据库连接成功")
    except Exception as e:
        print(f"[Server] ⚠️ 数据库初始化失败: {e}")
        print("💡 系统将自动降级：使用【模拟情报】+【DeepSeek大模型】生成报告。")
        GLOBAL_DB = None

# ==========================================
# 3. FastAPI 应用初始化
# ==========================================
app = FastAPI(title="态势报告生成系统 (DeepSeek版)")

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
    print(f"[Server] 收到分析指令: {request.user_prompt}")
    
    try:
        # --- 阶段一：情报检索 (RAG) ---
        retrieved_docs = []
        if GLOBAL_DB:
            try:
                retrieved_docs = GLOBAL_DB.search_embedding(request.user_prompt, top_k=3)
            except Exception as e:
                print(f"[Server] ⚠️ 向量检索出错: {e}")
        
        if retrieved_docs:
            print(f"[Server] ✅ 本地检索成功，召回 {len(retrieved_docs)} 条情报。")
            context_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)])
        else:
            print("[Server] ⚠️ 使用模拟情报数据构建上下文。")
            context_str = (
                "1. [模拟情报] 监测发现目标区域相关网络热度在过去24小时上升 300%。\n"
                "2. [模拟情报] 外部智库发布报告称，该领域供应链存在潜在中断风险。\n"
                "3. [模拟情报] 历史数据表明，类似事件通常会导致短期市场波动。"
            )
        
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

        # --- 阶段三：调用 DeepSeek API ---
        llm_content = ""
        
        if not DEEPSEEK_API_KEY:
            llm_content = "> ❌ **配置错误**：服务器未配置 API Key，请联系管理员检查 `.env` 文件。"
        else:
            try:
                print(f"[Server] 正在请求 DeepSeek API...")
                start_time = time.time()
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
                }
                
                payload = {
                    "model": CHAT_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": final_prompt}
                    ],
                    "stream": False,
                    "temperature": 1.3
                }

                resp = requests.post(
                    CHAT_API_URL, 
                    json=payload, 
                    headers=headers,
                    timeout=120 
                )
                
                resp.raise_for_status()
                result_json = resp.json()
                
                if "choices" in result_json and len(result_json["choices"]) > 0:
                    llm_content = result_json["choices"][0]["message"]["content"]
                    duration = time.time() - start_time
                    print(f"[Server] ✅ DeepSeek 生成完成 (耗时 {duration:.2f}s)")
                else:
                    print(f"[Server] ❌ API 返回结构异常: {result_json}")
                    llm_content = f"> **API 错误**：返回数据格式无法解析。"

            except Exception as e:
                print(f"[Server] ⚠️ DeepSeek 调用失败: {e}")
                llm_content = f"> **⚠️ 网络错误**：无法连接 DeepSeek API ({str(e)})。\n\n**相关情报：**\n{context_str}"

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

DEFAULT_SERVER_PORT = 29001

def find_free_port(start_port, max_retries=100):
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
        print(f"🚀 态势报告系统 (DeepSeek版) 已启动")
        print(f"🔗 访问地址: http://localhost:{PORT}")
        if DEEPSEEK_API_KEY:
            print(f"🔑 API Key: 已加载 (尾号 {DEEPSEEK_API_KEY[-4:]})")
        else:
            print(f"🔑 API Key: ❌ 未加载 (请检查 .env)")
        print("="*60)
        uvicorn.run(app, host="0.0.0.0", port=PORT)
    except Exception as e:
        print(f"❌ 启动失败: {e}")