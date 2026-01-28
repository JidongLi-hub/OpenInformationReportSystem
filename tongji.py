import os
from pathlib import Path
from collections import defaultdict

def summarize_by_level2(folder_path, extensions):
    """
    按二级子目录汇总统计文件
    :param folder_path: 根目录路径
    :param extensions: 需要统计的后缀列表，如 [".txt", ".pdf"]
    """
    root = Path(folder_path)
    if not root.is_dir():
        print(f"❌ 路径无效: {folder_path}")
        return

    # 存储数据结构: { "一级/二级": {"count": 0, "size": 0} }
    stats = defaultdict(lambda: {"count": 0, "size": 0})
    extensions = [ext.lower() for ext in extensions]
    
    total_count = 0
    total_size = 0

    # 递归遍历所有文件
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # 计算相对于根目录的路径
            relative_parts = file_path.relative_to(root).parts
            
            # 提取二级分类名称 (例如: "东南亚/越南")
            if len(relative_parts) >= 2:
                category = f"{relative_parts[0]}/{relative_parts[1]}"
            elif len(relative_parts) == 1:
                category = f"{relative_parts[0]} (根目录直属)"
            else:
                continue

            # 统计数据
            file_size = file_path.stat().st_size
            stats[category]["count"] += 1
            stats[category]["size"] += file_size
            
            total_count += 1
            total_size += file_size

    # --- 打印报表 ---
    print(f"\n{'='*60}")
    print(f"📊 二级子目录分类汇总报告")
    print(f"📂 根目录: {root.absolute()}")
    print(f"🏷️ 统计类型: {extensions}")
    print(f"{'='*60}")
    print(f"{'分类目录 (一级/二级)':<35} | {'文件数':<8} | {'占用空间':<10}")
    print(f"{'-'*60}")

    # 按名称排序输出
    for cat in sorted(stats.keys()):
        count = stats[cat]["count"]
        size_mb = stats[cat]["size"] / (1024 * 1024)
        print(f"{cat:<35} | {count:<8} | {size_mb:>8.2f} MB")

    print(f"{'-'*60}")
    print(f"{'【总计】':<35} | {total_count:<8} | {total_size/(1024*1024):>8.2f} MB")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # 指定你的 data2 目录
    root_folder = "datafiles" 
    
    # 统计你关心的三种格式
    summarize_by_level2(root_folder, [".txt", ".pdf", ".epub"])
