import os
import json

def process_json_files(folder_path):
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为JSON文件
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    # 打开并读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 确保数据是一个列表且非空
                    if isinstance(data, list) and len(data) > 0:
                        first_element = data[0]
                        
                        # 提取 Total Acc 和 Total Answer Reject Rate 的值
                        total_acc = first_element.get("Total Acc", [None])[0]  # 获取列表的第一个元素
                        total_answer_reject_rate = first_element.get("Total Answer Reject Rate")
                        
                        # 检查是否成功提取到值
                        if total_acc is not None and total_answer_reject_rate is not None:
                            # 格式化为四位小数并四舍五入
                            total_acc_rounded = round(total_acc, 4)
                            total_answer_reject_rate_rounded = round(total_answer_reject_rate, 4)
                            
                            # 打印结果
                            print(f"File: {file}")
                            print(f"Total Acc, Total Answer Reject Rate")
                            print(f"{total_acc_rounded}, {total_answer_reject_rate_rounded}")
                            print("-" * 30)
                        else:
                            print(f"File: {file} - Missing required keys or invalid format.")
                except json.JSONDecodeError:
                    print(f"File: {file} - Invalid JSON format.")
                except Exception as e:
                    print(f"File: {file} - Error processing file: {e}")
import os
import json

def process_json_files(folder_path):
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        # 按照文件名排序
        sorted_files = sorted(files)
        
        for file in sorted_files:
            # 检查文件是否为JSON文件
            if file.endswith(".json")  and ("Llama-3-1-70B" in file or "R1-distill-llama-8b" in file):
                file_path = os.path.join(root, file)
                try:
                    # 打开并读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 确保数据是一个列表且非空
                    if isinstance(data, list) and len(data) > 0:
                        first_element = data[0]
                        
                        # 提取 Total Acc 和 Total Answer Reject Rate 的值
                        total_acc = first_element.get("Total Acc", [None])[0]  # 获取列表的第一个元素
                        total_answer_reject_rate = first_element.get("Total Answer Reject Rate")
                        
                        # 检查是否成功提取到值
                        if total_acc is not None and total_answer_reject_rate is not None:
                            # 格式化为四位小数（不足四位补零）
                            total_acc_formatted = f"{total_acc:.4f}"
                            total_answer_reject_rate_formatted = f"{total_answer_reject_rate:.4f}"
                            
                            # 打印结果
                            print(f"File: {file}")
                            print(f"Total Acc: {total_acc_formatted}")
                            print(f"Total Answer Reject Rate: {total_answer_reject_rate_formatted}")
                            print(f"{total_acc_formatted}, {total_answer_reject_rate_formatted} ✅")
                            print("-" * 30)
                            print("\n")
                        else:
                            print(f"File: {file} - Missing required keys or invalid format.")
                except json.JSONDecodeError:
                    print(f"File: {file} - Invalid JSON format.")
                except Exception as e:
                    print(f"File: {file} - Error processing file: {e}")


# 示例调用
# folder_path = "/data1/wangyan/BaselineFramework/ExperimentResult/evaluation_result/sub1_different_noise-type/ChainofNote"  # 替换为您的文件夹路径
folder_path = "/data1/wangyan/BaselineFramework/ExperimentResult/evaluation_result/main_different_noise-ration/VanillaRAG"  # 替换为您的文件夹路径
process_json_files(folder_path)