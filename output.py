import pandas as pd
from pathlib import Path
from collections import defaultdict

def summarize_to_excel_clean(root_folder, extensions, output_excel="corpus_report.xlsx"):
    """
    按一级、二级子目录汇总统计，输出包含总计和明细的 Excel 表格。
    """
    root = Path(root_folder)
    if not root.is_dir():
        print(f"❌ 错误: 路径无效 {root_folder}")
        return

    # 数据结构: {(一级, 二级): {'文件数量': 0, '总大小(Bytes)': 0}}
    stats = defaultdict(lambda: {'文件数量': 0, '总大小(Bytes)': 0})
    extensions = [ext.lower() for ext in extensions]

    print(f"🔍 正在扫描根目录: {root.absolute()}")

    # 递归遍历所有子文件夹
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # 提取路径层级
            relative_parts = file_path.relative_to(root).parts
            l1 = relative_parts[0] if len(relative_parts) > 0 else "根目录"
            l2 = relative_parts[1] if len(relative_parts) > 1 else "无"
            
            key = (l1, l2)
            stats[key]['文件数量'] += 1
            stats[key]['总大小(Bytes)'] += file_path.stat().st_size

    data_list = []
    for (l1, l2), info in stats.items():
        data_list.append({
            "一级目录": l1,
            "二级目录": l2,
            "文件数量": info['文件数量'],
            "总大小(MB)": round(info['总大小(Bytes)'] / (1024 * 1024), 2)
        })

    if not data_list:
        print("⚠️ 未找到匹配的文件。")
        return

    # 转换为 DataFrame 并按层级排序
    df = pd.DataFrame(data_list)
    df = df.sort_values(by=["一级目录", "二级目录"])
    
    # 计算全局总计
    total_files = df['文件数量'].sum()
    total_size_mb = df['总大小(MB)'].sum()
    total_regions = df['一级目录'].nunique()

    # --- 导出到 Excel ---
    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Sheet 1: 详细统计
            df.to_excel(writer, sheet_name='详细分布统计', index=False)
            
            # Sheet 2: 总体汇总
            summary_df = pd.DataFrame([{
                "项目": "全语料总计",
                "总文件数量": total_files,
                "总大小(MB)": round(total_size_mb, 2),
                "涵盖一级分类数": total_regions
            }])
            summary_df.to_excel(writer, sheet_name='汇总摘要', index=False)

        # --- 控制台输出总数据量 ---
        print("\n" + "="*40)
        print(f"📊 语料库统计摘要")
        print("-" * 40)
        print(f"📁 根目录: {root.name}")
        print(f"📚 总计文件: {total_files:,} 篇")
        print(f"💾 总计大小: {total_size_mb:.2f} MB")
        print(f"🌍 覆盖地区: {total_regions} 个")
        print(f"✅ 报告已保存至: {output_excel}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"❌ 导出 Excel 失败: {e}")

if __name__ == "__main__":
    # 配置你的 data2 路径
    TARGET_FOLDER = "datafiles" 
    TARGET_EXTS = [".txt", ".pdf", ".epub"]
    OUTPUT_FILE = "datafiles/语料库规模统计表.xlsx"

    summarize_to_excel_clean(TARGET_FOLDER, TARGET_EXTS, OUTPUT_FILE)