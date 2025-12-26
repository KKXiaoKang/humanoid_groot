import pandas as pd
import sys

# 读取数据
file_path = '/home/lab/lerobot_groot/lerobot_data/v3_0_dataset/1215_5w_groot_4311_4322_4611_4633_downsample/meta/episodes/chunk-000/file-000.parquet'
df = pd.read_parquet(file_path)

print("=" * 80)
print("数据基本信息:")
print("=" * 80)
print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")

print(f"\n所有列名 ({len(df.columns)} 列):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:3d}. {col}")

print("\n" + "=" * 80)
print("数据前10行预览 (所有列):")
print("=" * 80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
print(df.head(10).to_string())

# 保存为 CSV 文件方便在 Excel 或其他工具中查看
csv_path = file_path.replace('.parquet', '.csv')
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n{'=' * 80}")
print(f"完整数据已保存到 CSV 文件: {csv_path}")
print(f"您可以使用 Excel、LibreOffice 或文本编辑器打开此文件查看所有数据")
print("=" * 80)

# 如果需要查看完整数据，可以取消下面的注释
# print("\n" + "=" * 80)
# print("完整数据 (所有行):")
# print("=" * 80)
# pd.set_option('display.max_rows', None)
# print(df.to_string())