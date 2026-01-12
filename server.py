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
    from database import VectorDatabase
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
CHAT_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

# 初始化全局数据库实例
GLOBAL_DB = None
if DB_MODULE_AVAILABLE:
    try:
        print("[Server] 正在初始化向量数据库服务...")
        GLOBAL_DB = VectorDatabase()
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

# 更新数据模型，增加 use_rag 开关字段
class UserRequest(BaseModel):
    user_prompt: str
    use_rag: bool = True  # 默认为开启

# ==========================================
# 4. 核心业务接口
# ==========================================

@app.post("/generate_report")
async def generate_report(request: UserRequest):
    print(f"[Server] 收到指令: {request.user_prompt} | RAG状态: {'✅开启' if request.use_rag else '🚫关闭'}")
    
    try:
        # --- 阶段一：情报检索 (根据开关决定是否执行) ---
        retrieved_docs = []
        context_str = ""
        # 标记是否真的获取到了有效情报
        has_valid_context = False 

        if request.use_rag:
            if GLOBAL_DB:
                try:
                    retrieved_docs = GLOBAL_DB.search_embedding(request.user_prompt, top_k=3)
                except Exception as e:
                    print(f"[Server] ⚠️ 检索异常: {e}")
            
            if retrieved_docs:
                print(f"[Server] ✅ 检索完成，召回 {len(retrieved_docs)} 条数据。")
                context_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)])
                has_valid_context = True
            else:
                # 只有在开启RAG且查不到时，才使用模拟数据兜底
                if GLOBAL_DB is None:
                     print("[Server] ⚠️ 数据库服务不可用，使用模拟数据。")
                     context_str = "1. [模拟] 无线电信号异常增强。\n2. [模拟] 气象海况恶劣。\n3. [模拟] 历史同期有演练。"
                     has_valid_context = True 
                else:
                     context_str = "未检索到相关情报。"
                     has_valid_context = False
        else:
            print("[Server] 🚫 RAG 已手动关闭，不进行检索。")
            context_str = "【系统状态：未接入知识库】"
            has_valid_context = False
        
        # --- 阶段二：构建 Prompt (核心优化) ---
        
        special_instruction = ""
        if not has_valid_context:
            # 动态调整要求：如果有情报，要求基于情报；如果没有，告诉模型直接写报告。
            special_instruction = "由于缺乏背景情报，请直接基于你的通用知识进行逻辑推演。"
        else:
            special_instruction = "请务必结合【背景情报】中的具体数据和事实进行深度分析，避免空谈。"

        final_prompt = f"""
        你是一个专业的态势分析员。请根据以下【背景情报】和【用户指令】，撰写一份专业的态势报告。
        
        【背景情报】：
        {context_str}
        
        【用户指令】：
        {request.user_prompt}
        
        要求：
        1. {special_instruction}
        2. 报告格式包含：摘要、现状分析、趋势预测。
        3. 语气专业、客观。必须基于情报事实进行推断，若情报不足请指出。
        4. 使用 Markdown 格式。
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
            llm_content = resp.json()["choices"][0]["message"]["content"]
            print("[Server] ✅ 报告生成完成。")
            
        except Exception as e:
            print(f"[Server] ⚠️ Chat 模型调用失败: {e}")
            llm_content = f"> **⚠️ 系统提示**：大模型连接异常 (Port {VLLM_PORT})。\n\n**相关情报：**\n{context_str}"

        # --- 阶段四：手动注入警示信息 (UI效果优化) ---
        # 如果没有有效情报，我们在 Python 层面强制拼接一个 Markdown 标题
        # 这样它就会像 Title 一样醒目，且绝对位于正文之前
        if not has_valid_context:
            warning_banner = """# ⚠️ 特别提示：未接入情报库
> **当前报告完全基于模型通用逻辑推演，缺乏具体情报数据支持，请谨慎参考。**

---
"""
            llm_content = warning_banner + llm_content

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
DEFAULT_SERVER_PORT = 28001

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