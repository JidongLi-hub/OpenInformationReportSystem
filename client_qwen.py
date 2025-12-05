import requests

def ask_qwen(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        proxies={"http": None, "https": None}
    )
    return resp.json()["choices"][0]["message"]["content"]

# 测试
prompt = """
你是一位专业的AI助手，请撰写一篇深度自我介绍，总字数严格不少于5000汉字。

要求：
- 内容分为五个部分：身份背景、核心技术、应用案例、伦理原则、未来愿景
- 每部分至少1000字，使用具体例子和细节描述
- 语言正式但亲切，避免空洞口号
- 不要提前结束，必须写满字数

现在开始：
"""
print(ask_qwen(prompt))