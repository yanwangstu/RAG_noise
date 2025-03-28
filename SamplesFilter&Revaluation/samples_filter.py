import json
import random
random.seed(42)

# 文件路径
file_1 = "/data1/wangyan/BaselineFramework/ExperimentResult/evaluation_result/main_different_noise-ration/VanillaRAG/main_noise-ration-07_VanillaRAG_Llama-3-1-8B.json"
file_2 = "/data1/wangyan/BaselineFramework/ExperimentResult/evaluation_result/main_different_noise-ration/VanillaRAG/main_noise-ration-07_VanillaRAG_Qwen2-5-7B.json"

# 打开第一个文件并提取符合条件的 QID
with open(file_1, "r") as file:
    data_1 = json.load(file)
acc_1 = set()
for sample in data_1:
    if "QID" in sample and type(sample["Acc Score"])==int and sample["Acc Score"]>=3:
        acc_1.add(sample["QID"])  # 使用 add() 添加到集合

# 打开第二个文件并提取符合条件的 QID
with open(file_2, "r") as file:
    data_2 = json.load(file)
acc_2 = set()
for sample in data_2:
    if "QID" in sample and type(sample["Acc Score"])==int and sample["Acc Score"]>=3:
        acc_2.add(sample["QID"])  # 使用 add() 添加到集合

# 求intersection
intersection = acc_1 & acc_2


# 输出结果
print("origin len_intersection:", len(intersection))
print("origin intersection:", intersection)

intersection = list(intersection)
random.shuffle(intersection)
intersection = intersection[:510]

# 输出结果
print("len_intersection:", len(intersection))
print("intersection:", intersection)


# 将intersection保存到 JSON 文件
output_file = "/data1/wangyan/Filter/filtered_QID.json"
with open(output_file, "w") as file:
    json.dump(intersection, file, indent=4)  # 转换为列表并美化输出
print(f"intersection has been saved to file: {output_file}")