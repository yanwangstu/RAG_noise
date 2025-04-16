import os
import re
import json
import argparse


filtered_QID = "/data1/wangyan/SamplesFilter&Revaluation/filtered_QID.json"
with open(filtered_QID, "r") as file:
    filtered_QID = json.load(file)


def calculate_token_len_distribution(docs_lens: list[dict],
                                     evaluation_results: list[dict], 
                                     interval=300):
    # 初始化分布统计字典
    # distribution = {"0-99": {samples_num: 0, total_acc_scores: [10, 200], total_rej_samples: [10, 200]}, 
    #                 "100-199": ..., 
    #                 ...}
    evaluation_results = evaluation_results[1:]

    distribution = {}
    for i in range(6):
        lower_bound = i * interval
        upper_bound = lower_bound + interval - 1
        if lower_bound == 1500:
            upper_bound = "1500+"
        interval_key = f"{lower_bound}-{upper_bound}"
        distribution[f"{lower_bound}-{upper_bound}"] = {"samples_num": 0, 
                                    "total_acc_scores": [0, 0], 
                                    "total_rej_samples": [0, 0]}
    


    docs_lens_dic = {sample["QID"]: sample["Token Len"] for sample in docs_lens}
    acc_scores_dic = {sample["QID"]: sample["Acc Score"] for sample in evaluation_results}
    rej_samples_dic = {sample["QID"]: sample["Answer Reject"] for sample in evaluation_results}
    
    # 遍历数据，统计每个 Token Len 所属区间
    for QID in docs_lens_dic:
        if QID in filtered_QID:
            continue
        if type(acc_scores_dic[QID]) != int or type(rej_samples_dic[QID]) != bool:
            continue

        token_len = docs_lens_dic[QID]
        
        # 计算所属区间
        lower_bound = (token_len // interval) * interval
        upper_bound = lower_bound + interval - 1
        if lower_bound >= 1500:
            lower_bound = 1500
            upper_bound = "1500+"
        interval_key = f"{lower_bound}-{upper_bound}"
        
        # 更新分布统计
        if interval_key not in distribution:
            distribution[interval_key] = {"samples_num": 0, 
                                          "total_acc_scores": [0, 0], 
                                          "total_rej_samples": [0, 0]}

        distribution[interval_key]["samples_num"] += 1

        distribution[interval_key]["total_acc_scores"][0] += acc_scores_dic[QID]
        distribution[interval_key]["total_acc_scores"][1] += 1

        if rej_samples_dic[QID] == True:
            distribution[interval_key]["total_rej_samples"][0] += 1
        distribution[interval_key]["total_rej_samples"][1] += 1

    return distribution


if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--src_dir", type=str)
    args = parser.parse_args()

    backbone = args.backbone
    src_dir = args.src_dir
    
    # 使用 os.listdir 获取文件列表，并按文件名排序
    file_names = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".json")],  # 过滤出 JSON 文件
        key=lambda x: int(re.search(r'\d+', x).group())  # 提取文件名中的第一个数字
    )
    
    print("Token Len Distribution Analysis: \n")
    # 遍历排序后的文件
    for file_name in file_names:
        # print("-" * 40)  # 分隔符，区分不同文件的结果
        # print(f"处理文件：{file_name}")
        file_path = os.path.join(src_dir, file_name)
        
        # 读取 JSON 数据
        with open(file_path, "r") as file:
            input_docs_lens = json.load(file)

        key=re.search(r'\d+', file_name).group()
        
        evaluation_file = f"/data1/wangyan/BaselineFrameworkAnalysis/ExperimentResult/evaluation_result/main_different_noise-ratio
        /VanillaRAG/main_noise-ratio-{key}_VanillaRAG_{backbone}.json"
        with open(evaluation_file, "r") as file:
            evaluation_results = json.load(file)
        
        # 计算分布
        distribution = calculate_token_len_distribution(input_docs_lens, evaluation_results)
        # 计算总样本数
        total_count  = sum(entry["samples_num"] for entry in distribution.values())
        
        # 按区间下界排序
        sorted_distribution = sorted(
            distribution.items(),
            key=lambda x: int(x[0].split("-")[0])  # 提取区间的下界并转换为整数
        )
        
        # 打印排序后的结果及百分比
        for interval, info in sorted_distribution:
            percentage = (info["samples_num"] / total_count) * 100  # 计算百分比
            # print(f"{interval}")
            # print(f"{info['samples_num']} ({percentage:.2f}%)") 
            # print(f"Acc Score: {0.2*info['total_acc_scores'][0] / info['total_acc_scores'][1]:.4f}")
            # print(f"Answer Reject: {info['total_rej_samples'][0] / info['total_rej_samples'][1]:.4f}\n")
            # print(f"{info['samples_num']} ({percentage:.2f}%)") 
            # print(f"Acc Score: {0.2*info['total_acc_scores'][0] / info['total_acc_scores'][1]:.4f}")
            # print(f"Answer Reject: {info['total_rej_samples'][0] / info['total_rej_samples'][1]:.4f}\n")

            # print(f"{interval}")
            if info['samples_num'] != 0:
                print(f"{percentage:.2f}%\t{0.2*info['total_acc_scores'][0] / info['total_acc_scores'][1]:.4f}, {info['total_rej_samples'][0] / info['total_rej_samples'][1]:.4f}") 
            else:
                print(f"0%\tNone\tNone")

            # print(f"")
            # print(f"Answer Reject: {info['total_rej_samples'][0] / info['total_rej_samples'][1]:.4f}\n")
            # print(f"{info['samples_num']} ({percentage:.2f}%)") 
            # print(f"Acc Score: {0.2*info['total_acc_scores'][0] / info['total_acc_scores'][1]:.4f}")
            # print(f"Answer Reject: {info['total_rej_samples'][0] / info['total_rej_samples'][1]:.4f}\n")
        