import json
import random
from typing import Dict


def testbed_construction(
        input_file: str, 
        output_file: str, 
        sample_counts: Dict[str, int]
        ) -> None:
    """
    从JSON文件中读取数据,按指定数量随机抽取各类文档,并保存到新文件
    
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    :param sample_counts: 各类文档的抽取数量字典，格式如下：
    {
        "Golden Documents": 2,
        "Distracting Documents": 3,
        "Inconsequential Documents": 1,
        "Low Quality Documents": 2,
        "Transfer Documents": 1,
        "Irrelevant Documents": 0
    }
    """
    # load origin sampels
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_data = []
    successed_counstruction = 0

    for sample in data:
        if successed_counstruction == 10:
           break
        try:
            new_sample = {}
            new_sample["QID"] = sample["QID"]
            new_sample["Question"] = sample["Question"]
            new_sample["Answers"] = sample["Answers"]

            # process each doc_type
            for doc_type, count in sample_counts.items():
                if count != 0:
                    # fetch documents with doc_type in list
                    documents = sample.get(doc_type)
                    # randomly sample specific count of documents
                    sampled_documents = random.sample(documents, count)
                    new_sample[doc_type] = sampled_documents

            processed_data.append(new_sample)
            successed_counstruction += 1
        except Exception as e:
            print(f"error occures during the process：{str(e)}")

    # write processed data
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)
    
    print(f"successed_counstruction samples: {successed_counstruction}")

    return


# usage example
if __name__ == "__main__":
    sample_config = {
        "Golden Documents": 1,
        "Distracting Documents": 2,
        "Inconsequential Documents": 2,
        "Low Quality Documents": 2,
        "Irrelevant Documents": 1
    }

    testbed_construction(
        input_file="dataset_add_irrelevant/level_1_6000/level_1_6000.json",
        output_file="testbed/example.json",
        sample_counts=sample_config
    )