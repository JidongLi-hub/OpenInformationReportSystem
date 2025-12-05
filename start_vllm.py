import subprocess
import os
import sys

# --- 配置区域 ---
MODEL_PATH = "Qwen/Qwen2-7B-Instruct"  # 模型路径或名称
SERVED_NAME = "Qwen/Qwen2-7B-Instruct" # API 服务中的模型显示名称
PORT = 8002                            # 服务端口
MAX_MEMORY_USAGE = 1000                # 显存占用小于此值(MB)视为"空闲"

def get_free_gpus():
    """
    使用 nvidia-smi 查询所有显卡的显存使用情况
    返回空闲显卡的 ID 列表
    """
    try:
        # 执行 nvidia-smi 查询命令
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        
        free_gpus = []
        lines = result.strip().split('\n')
        for line in lines:
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
        print("❌ 错误: 当前没有空闲的显卡！请稍后再试或检查 nvidia-smi。")
        sys.exit(1)
    
    print(f"✅ 发现空闲显卡: {free_gpus}")
    
    # --- 决策逻辑 ---
    target_gpus = []
    tp_size = 1
    
    # 策略: 优先凑 2 张卡做并行，如果不够就用 1 张
    if len(free_gpus) >= 2:
        target_gpus = free_gpus[:2] # 取前两张
        tp_size = 2
        print(f"🚀 策略: 显卡充足，将使用 GPU {target_gpus} 开启双卡并行模式 (TP=2)")
    else:
        target_gpus = free_gpus[:1] # 取第一张
        tp_size = 1
        print(f"⚠️ 策略: 显卡紧张，将使用 GPU {target_gpus} 开启单卡模式 (TP=1)")

    # --- 构造启动命令 ---
    gpu_str = ",".join(target_gpus)
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    # --- 核心修复：清理代理配置 ---
    # vLLM 尝试通过环境变量里的代理联网，但代理可能没开导致 Connection refused。
    # 这里强制移除这些变量，让 vLLM 直连或使用本地缓存。
    proxies = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
    print("-" * 50)
    print("🧹 正在清理环境代理设置 (防止连接被拒绝)...")
    for p in proxies:
        if p in env:
            print(f"   - 移除: {p}")
            del env[p]
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_NAME,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp_size), # 动态设置 TP 参数
        "--port", str(PORT)
    ]
    
    print("-" * 50)
    print(f"执行命令: CUDA_VISIBLE_DEVICES={gpu_str} {' '.join(cmd)}")
    print("-" * 50)
    
    # --- 启动 vLLM ---
    try:
        # 使用修改后的 env (无代理) 启动子进程
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止。")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 服务启动出错: {e}")

if __name__ == "__main__":
    main()