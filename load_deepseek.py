from vllm import LLM

print("【第1步】开始加载 DeepSeek-Coder-7B-Instruct 模型...")

llm = LLM(
    model="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    gpu_memory_utilization=0.8,
    dtype="auto"
)

print("【第2步】模型加载成功！🎉")
print("现在可以生成文本了，我们试一句中文...")

outputs = llm.generate(
    ["你好，请介绍一下你自己。"],
    use_tqdm=False
)

print("【第3步】模型回复：")
print(outputs[0].outputs[0].text)