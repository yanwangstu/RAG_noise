import os
import json
from modelscope import AutoTokenizer


model_path = "/datanfs2/zyx/model_cache/LLM-Research/Meta-Llama-3___1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# calculate the token length of the docs
def docs_token_len_calculate(docs: list[str]) -> int:
    encoded = tokenizer(docs, return_tensors=None, padding=False, truncation=False)
    token_len = sum(len(ids) for ids in encoded["input_ids"])
    return token_len


# calculate the token length of the docs in batch (the whole file)
def docs_token_len_calculate_batch(src_file_path: str) -> list[dict]:
    with open(src_file_path, 'r', encoding='utf-8') as src_file:
        data = json.load(src_file)

    processed_data = []
    for sample in data:
        docs = sample["Retrieval Documents"]
        token_len = docs_token_len_calculate(docs)
        print(f"Current Process QID: {sample['QID']}")
        processed_data.append({
            "QID": sample["QID"],
            "Token Len": token_len
        })
    return processed_data


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
            print(f"\n开始处理: {dst_file_path}")
          
            processed_data = docs_token_len_calculate_batch(src_file_path)

            # 将处理后的 JSON 数据写入目标文件
            with open(dst_file_path, 'w', encoding='utf-8') as dst_file:
                json.dump(processed_data, dst_file, ensure_ascii=False, indent=4)
                print(f"已处理并保存: {dst_file_path}")


if __name__ == "__main__":
    testbed_main_dir = "/data1/wangyan/BaselineFramework/testbed/main_different_noise-ratio"
    dst_dir = "/data1/wangyan/DocsLenDistributionAnalysis/main_different_noise-ratio_token-len"
    traverse_and_process(testbed_main_dir, dst_dir)
