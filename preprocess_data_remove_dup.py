import os
import pandas as pd

def find_matching_files(folder_path, csv_path):
    # 读取 CSV 文件中的文件名列（File Name）
    csv_df = pd.read_csv(csv_path)
    csv_files = set(csv_df['file_name'].astype(str)
        .str.replace('Time', 'T')
        .str.replace('_FIS', '').str.replace('_FIs', '').str.strip())  # 获取 CSV 中的所有文件名，并去掉空白字符

    # 获取文件夹中的所有文件
    folder_files = set(f.replace('.mp4', '') for f in os.listdir(folder_path))  # 直接在列表中使用 replace

    # 找到文件夹中有，但 CSV 文件中没有的文件
    not_in_csv = folder_files - csv_files
    not_in_folder = csv_files - folder_files
    # 打印出来这些没有在 CSV 中的文件
    if not_in_csv:
        print("以下文件在文件夹中存在，但没有在CSV中找到:")
        for file in not_in_csv:
            print(file)
    else:
        print("没有文件在文件夹中存在但没有在CSV中找到。")

    if not_in_folder:
        print("以下文件在CSV中存在，但没有在文件夹中找到:")
        for file in not_in_folder:
            print(file)
    else:
        print("没有文件在CSV中存在但没有在文件夹中找到。")
    # 将不重合的文件从folder中删除
    for file in not_in_csv:
        file_path = os.path.join(folder_path, file + '.mp4')
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
    # 找到文件夹和 CSV 文件中都存在的文件
    common_files = folder_files & csv_files

    # 如果你希望保存这些重合的文件，可以根据需求进行处理
    # 比如保存到一个新的文件夹，或者做其他处理
    print(len(common_files), "个文件在文件夹和CSV中都存在。")

# 示例用法
folder_path = 'data/FIS1121'  # 文件夹路径
csv_path = 'data/coding.csv'  # CSV 文件路径

find_matching_files(folder_path, csv_path)
