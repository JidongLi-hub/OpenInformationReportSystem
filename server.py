from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <---【新增】导入CORS库
from pydantic import BaseModel
import requests
import uvicorn

# 初始化 FastAPI 应用
app = FastAPI(title="态势报告生成系统后端")

# ==========================================
# 👇【新增】配置跨域允许 (CORS)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    # 允许所有来源访问（比如 http://localhost:8501）
    # 在生产环境中通常会指定具体的域名，但在开发测试阶段用 "*" 最方便
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法 (GET, POST等)
    allow_headers=["*"],  # 允许所有 HTTP 头
)
# ==========================================

# --- 配置部分 ---
# 指向 vLLM 服务的地址（保留你代码中的 8002 端口）
VLLM_API_URL = "http://localhost:8002/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

# --- 1. 定义数据模型 ---
class UserRequest(BaseModel):
    user_prompt: str

# --- 2. 模拟数据库查询 ---
def mock_search_database(query: str):
    print(f"[后端日志] 正在数据库中检索关键词: {query} ...") 
    return "【检索结果】：近期该区域有频繁的海上活动，且伴随有季风气候影响。多方势力在此进行了常规巡航。"

# --- 3. 核心接口：生成报告 ---
@app.post("/generate_report")
async def generate_report(request: UserRequest):
    print(f"[后端日志] 收到前端请求: {request.user_prompt}")
    
    try:
        # 步骤 A: 去数据库查资料
        retrieved_info = mock_search_database(request.user_prompt)
        
        # 步骤 B: 组装 Prompt
        final_prompt = f"""
        你是一个专业的态势分析员。请根据以下【背景信息】和【用户指令】，撰写一份专业的态势报告。
        
        【背景信息】：
        {retrieved_info}
        
        【用户指令】：
        {request.user_prompt}
        
        要求：
        1. 报告格式包含：摘要、现状分析、趋势预测。
        2. 语气专业、客观。
        3. 字数控制在 500 字以内（测试用）。
        4. 使用 Markdown 格式。
        """

        # 步骤 C: 准备 payload
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        # 步骤 D: 请求 vLLM
        print("[后端日志] 正在请求 vLLM 模型...")
        response = requests.post(
            VLLM_API_URL, 
            json=payload, 
            proxies={"http": None, "https": None}
        )
        response.raise_for_status() 
        
        llm_content = response.json()["choices"][0]["message"]["content"]
        print("[后端日志] 模型生成完毕。")

        # 步骤 E: 返回结果
        return {
            "status": "success",
            "original_query": request.user_prompt,
            "retrieved_info": retrieved_info,
            "report_content": llm_content
        }

    except Exception as e:
        print(f"[错误] 处理请求时发生异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动服务器在 8001 端口
    uvicorn.run(app, host="0.0.0.0", port=8001)