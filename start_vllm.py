import subprocess
import os
import sys

# --- 配置区域 ---
MODEL_PATH = "Qwen/Qwen2-7B-Instruct"
SERVED_NAME = "Qwen/Qwen2-7B-Instruct"
# 【核心修改】使用冷门高位端口，避免冲突
PORT = 28888 
MAX_MEMORY_USAGE = 1000

def get_free_gpus():
    """查询空闲显卡"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        free_gpus = []
        for line in result.strip().split('\n'):
            index, memory = line.split(',')
            if int(memory.strip()) < MAX_MEMORY_USAGE:
                free_gpus.append(index.strip())
        return free_gpus
    except Exception as e:
        print(f"❌ 获取显卡状态失败: {e}")
        sys.exit(1)

def main():
    print("🔍 正在扫描服务器显卡状态...")
    free_gpus = get_free_gpus()
    
    if not free_gpus:
        print("❌ 错误: 当前没有空闲的显卡！")
        sys.exit(1)
    
    print(f"✅ 发现空闲显卡: {free_gpus}")
    
    # 策略: 优先凑 2 张卡
    if len(free_gpus) >= 2:
        target_gpus = free_gpus[:2]
        tp_size = 2
        print(f"🚀 策略: 使用 GPU {target_gpus} (双卡并行)")
    else:
        target_gpus = free_gpus[:1]
        tp_size = 1
        print(f"⚠️ 策略: 使用 GPU {target_gpus} (单卡模式)")

    gpu_str = ",".join(target_gpus)
    
    # --- 环境变量设置 ---
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    # 1. 清理代理
    proxies = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
    for p in proxies:
        if p in env: del env[p]
    
    # 2. 强制离线模式
    env["HF_HUB_OFFLINE"] = "1"

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_NAME,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp_size),
        "--port", str(PORT)
    ]
    
    print("-" * 50)
    print(f"🚀 准备在端口 {PORT} 启动模型...")
    print(f"执行命令: CUDA_VISIBLE_DEVICES={gpu_str} ... --port {PORT}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止。")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 服务启动出错: {e}")

if __name__ == "__main__":
    main()