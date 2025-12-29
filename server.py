import os
import re
import socket
import uvicorn
import requests
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 1. 核心配置与工具函数
# ==========================================

# 指向 vLLM 服务的地址
# ⚠️ 请确保这里是你真实 vLLM 运行的端口 (你之前说是 8002)
VLLM_API_URL = "http://localhost:8002/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

def find_free_port(start_port=8001, max_retries=100):
    """自动寻找空闲端口"""
    port = start_port
    while port < start_port + max_retries:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                try:
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError:
                    pass
        port += 1
    raise RuntimeError("找不到可用的空闲端口！")

# ==========================================
# 2. FastAPI 应用初始化
# ==========================================
app = FastAPI(title="态势报告生成系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. 业务逻辑 (数据库模拟 & 报告生成)
# ==========================================

class UserRequest(BaseModel):
    user_prompt: str

def mock_search_database(query: str):
    """模拟数据库检索"""
    print(f"[后端日志] 正在数据库中检索: {query}") 
    return [
        "1. [情报源A] 监测数据显示，过去48小时内，目标海域的无线电通信频率比平时增加了 15%。",
        "2. [情报源B] 气象部门预报，受季风低压影响，该区域未来三天将出现 4-5 米的巨浪。",
        "3. [情报源C] 历史类似事件回顾：去年同期，周边国家曾在此海域进行过联合演练。"
    ]

@app.post("/generate_report")
async def generate_report(request: UserRequest):
    """生成报告的核心 API"""
    print(f"[后端日志] 收到指令: {request.user_prompt}")
    
    try:
        # RAG 检索
        retrieved_docs = mock_search_database(request.user_prompt)
        context_str = "\n".join(retrieved_docs)
        
        # 组装 Prompt
        final_prompt = f"""
        你是一个专业的态势分析员。请根据以下【背景信息】和【用户指令】，撰写一份专业的态势报告。
        【背景信息】：{context_str}
        【用户指令】：{request.user_prompt}
        要求：Markdown格式，字数500以内，分条列述。
        """

        # 调用 vLLM (带降级处理)
        llm_content = ""
        try:
            print(f"[后端日志] 正在请求 vLLM (端口 8002)... 请耐心等待...")
            resp = requests.post(
                VLLM_API_URL, 
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": final_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }, 
                proxies={"http": None, "https": None},
                # ====================================================
                # 👇【核心修改】将超时时间从 5 秒改为 60 秒
                # 大模型生成需要时间，5秒太短了
                # ====================================================
                timeout=60 
            )
            resp.raise_for_status()
            llm_content = resp.json()["choices"][0]["message"]["content"]
            print("[后端日志] ✅ vLLM 生成成功！")
            
        except requests.exceptions.Timeout:
            print(f"[后端日志] ⚠️ vLLM 响应超时 (>60s) -> 切换到模拟模式")
            llm_content = self._get_fallback_content()
            
        except Exception as e:
            print(f"[后端日志] ⚠️ vLLM 调用出错: {e} -> 切换到模拟模式")
            llm_content = self._get_fallback_content()

        # 如果 llm_content 为空（比如 try 块未完全执行），赋予默认值
        if not llm_content:
             llm_content = self._get_fallback_content()

        return {
            "status": "success",
            "original_query": request.user_prompt,
            "retrieved_info": context_str,
            "report_content": llm_content
        }

    except Exception as e:
        print(f"[错误] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    def _get_fallback_content(self):
        """返回降级用的模拟数据"""
        return """
> **⚠️ 系统提示**：模型服务响应超时或不可用，以下为规则引擎生成的模拟数据。

## 📊 态势分析报告（离线版）
根据检索到的情报（无线电频率增加、恶劣海况），判断当前区域存在**非典型军事活动**特征。
建议持续关注气象窗口期（未来72小时）。
"""

# 为了兼容函数内调用，定义一个独立的 fallback 函数
def _get_fallback_content_standalone():
    return """
> **⚠️ 系统提示**：模型服务响应超时或不可用，以下为规则引擎生成的模拟数据。

## 📊 态势分析报告（离线版）
根据检索到的情报（无线电频率增加、恶劣海况），判断当前区域存在**非典型军事活动**特征。
建议持续关注气象窗口期（未来72小时）。
"""

# 修正 generate_report 内部的调用
@app.post("/generate_report")
async def generate_report_fixed(request: UserRequest):
    print(f"[后端日志] 收到指令: {request.user_prompt}")
    
    try:
        retrieved_docs = mock_search_database(request.user_prompt)
        context_str = "\n".join(retrieved_docs)
        
        final_prompt = f"""
        你是一个专业的态势分析员。请根据以下【背景信息】和【用户指令】，撰写一份专业的态势报告。
        【背景信息】：{context_str}
        【用户指令】：{request.user_prompt}
        要求：Markdown格式，字数500以内，分条列述。
        """

        llm_content = ""
        try:
            print(f"[后端日志] 正在请求 vLLM (端口 8002)... 请耐心等待...")
            resp = requests.post(
                VLLM_API_URL, 
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": final_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }, 
                proxies={"http": None, "https": None},
                # 【修改】超时时间改为 60 秒
                timeout=60 
            )
            resp.raise_for_status()
            llm_content = resp.json()["choices"][0]["message"]["content"]
            print("[后端日志] ✅ vLLM 生成成功！")
            
        except Exception as e:
            print(f"[后端日志] ⚠️ vLLM 调用异常: {e} -> 切换到模拟模式")
            llm_content = _get_fallback_content_standalone()

        return {
            "status": "success",
            "original_query": request.user_prompt,
            "retrieved_info": context_str,
            "report_content": llm_content
        }

    except Exception as e:
        print(f"[错误] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 覆盖之前的路由定义
app.router.routes = [r for r in app.router.routes if r.path != "/generate_report"]
app.post("/generate_report")(generate_report_fixed)


# ==========================================
# 4. 前端托管 (核心黑科技)
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = "index.html"
    if not os.path.exists(frontend_path):
        return "<h1>错误：找不到 index.html 文件</h1><p>请确保 index.html 和 server.py 在同一目录下。</p>"
    
    with open(frontend_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(
        r'fetch\s*\(\s*["\']http://localhost:\d+/generate_report["\']', 
        'fetch("/generate_report"', 
        content
    )
    
    return HTMLResponse(content=content)

# ==========================================
# 5. 启动入口
# ==========================================

if __name__ == "__main__":
    try:
        PORT = find_free_port(8001)
    except Exception as e:
        print(f"❌ {e}")
        exit(1)

    print("="*50)
    print(f"🚀 系统启动成功！")
    print(f"🌍 访问地址: http://localhost:{PORT}")
    print(f"🔌 后端端口: {PORT} (前端已自动集成)")
    print("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)