import requests
import json

# 后端服务的地址 (server.py 监听的端口)
SERVER_URL = "http://localhost:8001/generate_report"

def test_backend():
    print(f"📡 正在连接后端: {SERVER_URL} ...")
    
    # 构造测试数据
    test_payload = {
        "user_prompt": "请分析当前的南海局势，重点关注最近的航行自由行动。"
    }
    
    try:
        # 发送 POST 请求
        response = requests.post(
            SERVER_URL,
            json=test_payload,
            proxies={"http": None, "https": None}, # 依然需要忽略系统代理
            timeout=300 # 给足够的时间等待生成
        )
        
        # 检查状态码
        if response.status_code == 200:
            print("\n✅ 后端连接成功！")
            data = response.json()
            
            print("-" * 30)
            print(f"📝 原始指令: {data.get('original_query')}")
            print(f"🔍 检索到的情报: {data.get('retrieved_info')}")
            print("-" * 30)
            print("📄 生成的报告内容预览 (前200字):")
            print(data.get('report_content', '')[:200] + "...")
            print("-" * 30)
            print("测试通过！后端逻辑正常。")
        else:
            print(f"\n❌ 请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到服务器。")
        print("请检查：\n1. server.py 是否正在运行？\n2. 端口是否真的是 8001？")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

if __name__ == "__main__":
    test_backend()