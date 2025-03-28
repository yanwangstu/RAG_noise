import os
import json

filter_QID_path = "/data1/wangyan/Filter/intersection_result.json"
with open(filter_QID_path, 'r', encoding='utf-8') as file:
    filter_QID = json.load(file)


def filter_acc_rej_recalculate(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        evaluation_result = json.load(file)

    total_acc_score = 0
    total_samples_for_acc = 0
    total_rej_samples = 0
    total_samples_for_rej = 0
    for item in evaluation_result:
        if "QID" in item:
            QID = item["QID"]
            if QID not in filter_QID:
                # None represents the RAG system resoponse failed
                # "Response Failed:" represents the evaluation system resoponse failed
                if type(item["Acc Score"]) == int:
                    total_acc_score += item["Acc Score"]
                    total_samples_for_acc += 1
                if type(item["Answer Reject"]) == bool:
                    total_rej_samples += 1 if item["Answer Reject"] else 0
                    total_samples_for_rej += 1

    total_acc = total_acc_score/(total_samples_for_acc*5)
    total_rej = total_rej_samples/total_samples_for_rej

    filter_evaluation_result = {
        "Total Acc": [
            total_acc,
            total_samples_for_acc
        ],
        "Total Answer Reject Rate": [
            total_rej,
            total_samples_for_rej
        ]
    }
    return filter_evaluation_result


def traverse_and_process(src_dir, dst_dir):
    """
    遍历源目录，处理 JSON 文件并存储到目标目录。

    :param src_dir: 源目录路径
    :param dst_dir: 目标目录路径
    """
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"源目录不存在: {src_dir}")
        return

    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 使用 os.walk 遍历源目录
    for root, dirs, files in os.walk(src_dir):
        # 计算相对路径，用于在目标目录中创建相同的目录结构
        relative_path = os.path.relpath(root, src_dir)
        target_path = os.path.join(dst_dir, relative_path)

        # 创建目标目录结构
        os.makedirs(target_path, exist_ok=True)

        # 遍历文件并处理
        for file_name in files:
            # 只处理 .json 文件
            if not file_name.endswith('.json'):
                continue

            src_file_path = os.path.join(root, file_name)
            dst_file_path = os.path.join(target_path, file_name)

          
            processed_data = filter_acc_rej_recalculate(src_file_path)

            # 将处理后的 JSON 数据写入目标文件
            with open(dst_file_path, 'w', encoding='utf-8') as dst_file:
                json.dump(processed_data, dst_file, ensure_ascii=False, indent=4)
                print(f"已处理并保存: {dst_file_path}")


# 示例用法
if __name__ == "__main__":
    evaluation_result_dic = "/data1/wangyan/BaselineFramework/ExperimentResult/evaluation_result/"
    filter_result_dic = "/data1/wangyan/Filter/evaluation_result_filter"
    
    traverse_and_process(src_dir=evaluation_result_dic, 
                         dst_dir=filter_result_dic)