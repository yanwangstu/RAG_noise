import os
import json

def process_json_files(folder_path, filter_backbone):
    # 遍历文件夹中的所有文件
    evaluation_info = []
    for root, dirs, files in os.walk(folder_path):
        # 按照文件名排序
        sorted_files = sorted(files)
        
        for file in sorted_files:
            # 检查文件是否为JSON文件
            if file.endswith(".json") and filter_backbone in file:
                file_path = os.path.join(root, file)
                try:
                    # 打开并读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                        # 提取 Total Acc 和 Total Answer Reject Rate 的值
                        total_acc = data["Total Acc"][0]
                        total_answer_reject_rate = data["Total Answer Reject Rate"][0]
                        
                        # 检查是否成功提取到值
                        if total_acc is not None and total_answer_reject_rate is not None:
                            # 格式化为四位小数（不足四位补零）
                            total_acc_formatted = f"{total_acc:.4f}"
                            total_answer_reject_rate_formatted = f"{total_answer_reject_rate:.4f}"
                            
                            # 打印结果
                            # print(f"File: {file}")
                            # print(f"Total Acc: {total_acc_formatted}")
                            # print(f"Total Answer Reject Rate: {total_answer_reject_rate_formatted}")
                            evaluation_info.append(f"{total_acc_formatted}, {total_answer_reject_rate_formatted} ✅")
                            # print("-" * 30)
                            # print("\n")
                        else:
                            print(f"File: {file} - Missing required keys or invalid format.")
                except json.JSONDecodeError:
                    print(f"File: {file} - Invalid JSON format.")
                except Exception as e:
                    print(f"File: {file} - Error processing file: {e}")
    return evaluation_info



if __name__ == "__main__":

    # settings
    experiment_name = "main"
    variable_name = "noise-ration"
    framework_name="VanillaRAG"
    filter_backbones = ["Llama-3-1-8B", "Qwen2-5-7B", 
                       "Llama-3-1-70B", "Qwen2-5-72B", 
                       "Deepseek-v3", "Deepseek-r1",
                       "R1-distill-llama-8b", "R1-distill-llama-70b"]

    for filter_backbone in filter_backbones:
        folder_path = f"/data1/wangyan/SamplesFilter&Revaluation/evaluation_result_filter/{experiment_name}_different_{variable_name}/{framework_name}"
        info = process_json_files(folder_path, filter_backbone)
        result = "\t".join(map(str, info))
        print(result)
