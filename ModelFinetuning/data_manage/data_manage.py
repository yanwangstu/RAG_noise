import os
import json
import random
import numpy as np
import torch
from modelscope import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset

TOTAL_DOCS_NUM = 5
NOISE_DOCS_NUMS = [0, 2, 4, 5]

random.seed(42)


def golden_noise_docs_scratch(sample: dict, noise_docs_num: int)-> list[dict]:
    """
    return info
    [ 
        {'doc': 'sample doc 1', 'label': 0}, 
        {'doc': 'sample doc 2', 'label': 1}, 
        {'doc': 'sample doc 3', 'label': 2},
        {'doc': 'sample doc 4', 'label': 3},
        {'doc': 'sample doc 5', 'label': 4}
    ]
    """
    golden_docs_num = TOTAL_DOCS_NUM-noise_docs_num
    if golden_docs_num != 0:
        golden_docs = [{'doc': text, 'label': 0} for text in sample["Golden Documents"]]
        if len(golden_docs) < golden_docs_num:
            return None
        random.shuffle(golden_docs)
        scratched_golden_docs = golden_docs[:golden_docs_num]
    else:
        scratched_golden_docs = []

    if noise_docs_num != 0:
        mapping_table = {
            1: "Distracting Documents",
            2: "Low Quality Documents",
            3: "Inconsequential Documents",
            4: "Irrelevant Documents"
        }
        noise_docs = []
        for lable, noise_type in mapping_table.items():
            noise_docs += [{'doc': text, 'label': lable} for text in sample[noise_type]]
        if len(noise_docs) < noise_docs_num:
            return None
        random.shuffle(noise_docs)
        scratched_noise_docs = noise_docs[:noise_docs_num]
    else:
        scratched_noise_docs = []

    scratch_docs = scratched_golden_docs + scratched_noise_docs
    random.shuffle(scratch_docs) 

    return scratch_docs


def message_generate_RAG(question: str, docs: list[str])-> list[dict]:
    task_descripition = """Task Description: 
    1. Answer the given Question based on the Retrieval Documents, do NOT add any explanations when giving the response.
    2. If you cannot answer with certainty due to insufficient information, you MUST respond verbatim:  \"I cannot answer the question.\"
    """
    
    formatted_question = f"Question: {question}"
    formatted_docs = "Retrieval Documents:\n" + "\n".join(
            [f"[DOC]{i + 1}. {doc}" for i, doc in enumerate(docs)])
    
    messages = [
            {"role": "system", "content": task_descripition},
            {"role": "user", "content": formatted_question},
            {"role": "user", "content": formatted_docs},
        ]
    return messages


class DataManage:
    def __init__(self, 
                 model_name: str,
                 model_path: str, 
                 dataset_path: str,
                 dataset_type: str, # train/dev
                 cache_path: str):
        
        self.model_name = model_name
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.dataset_type = dataset_type

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # add special token into tokenizer
        self.doc_token = "[DOC]"
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.doc_token]})
        self.doc_token_id: int
        self.doc_token_id = self.tokenizer.convert_tokens_to_ids(self.doc_token)

    def load_origin_data_and_reorganize(self)-> None:
        """
        load origin data and reorganize it to store into dic "train_cache"
        each reorganized samples contains 5 groups of retrieval docs, with different noise ration
        (noise ration = 0.0, 0.3, 0.6, 0.8, 1.0)

        an example of reorganized sample:
        {
            'subID': 0
            'Question': 'sample query',
            'Answers': ['xxx'],
            'retrieval_groups': [
                [ 
                    {'doc': 'sample doc 1', 'label': 0}, 
                    {'doc': 'sample doc 2', 'label': 1}, 
                    {'doc': 'sample doc 3', 'label': 2},
                    {'doc': 'sample doc 4', 'label': 3},
                    {'doc': 'sample doc 5', 'label': 4}
                ],

                [ 
                    {'doc': 'sample doc 1', 'label': 1}, 
                    {'doc': 'sample doc 2', 'label': 0}, 
                    {'doc': 'sample doc 3', 'label': 0},
                    {'doc': 'sample doc 4', 'label': 0},
                    {'doc': 'sample doc 5', 'label': 0}
                ],
                ...
            ]
        }

        docs_mapping_table
        {
            0: "Golden Documents",
            1: "Distracting Documents",
            2: "Low Quality Documents",
            3: "Inconsequential Documents",
            4: "Irrelevant Documents"
        }
        """
        # load dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)

        reorganized_dataset = []
        noise_docs_nums_len = len(NOISE_DOCS_NUMS)
       
        # reorganize dataset
        for sample in dataset:
            reorganized_sample = {
                'subID': sample['subID'],
                'Question': sample['Question'],
                'Answers': sample['Answers']
            }

            retrieval_groups = []
            for noise_docs_num in NOISE_DOCS_NUMS:
                retrieval_group = golden_noise_docs_scratch(sample, noise_docs_num)
                if retrieval_group is None:
                    break
                retrieval_groups.append(retrieval_group)
            if len(retrieval_groups) != noise_docs_nums_len:
                continue
            reorganized_sample['retrieval_groups'] = retrieval_groups
            reorganized_dataset.append(reorganized_sample)
        

        # store reorganize dataset
        with open(os.path.join(self.cache_path, f"{self.model_name}_{self.dataset_type}_reorganize.json"), 'w') as file:
            json.dump(reorganized_dataset, file, indent=4, ensure_ascii=False)
        return
    
    def tokenize_text(self, text: str):
        token_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens, token_ids
        
    def combine_reorganized_dataset_text(self)-> None:
        with open(os.path.join(self.cache_path, f"{self.model_name}_{self.dataset_type}_reorganize.json"), 'r') as file:
            reorganized_dataset = json.load(file)

        combined_dataset = []
        for sample in reorganized_dataset:
            output_text = ", ".join(sample['Answers'])
            retrieval_groups = sample['retrieval_groups']
            input_text_list = []
            output_class_num_list = []

            out_of_length = False
            for group in retrieval_groups:
                docs_list = [doc['doc'] for doc in group]
                input_dict = message_generate_RAG(sample['Question'], docs_list)
                input_text = self.tokenizer.apply_chat_template(
                    input_dict,
                    tokenize=False,
                    add_generation_prompt=True
                )
                input_text_length = len(input_text.split())

                if input_text_length >= 2000:
                    out_of_length = True
                    break
                input_text_list.append(input_text)

                output_class_num = [doc['label'] for doc in group]
                output_class_num_list.append(output_class_num)

            if out_of_length == False:
                combine_sample = {
                    "input_text_list": input_text_list, 
                    "output_text": output_text, 
                    "output_class_num_list": output_class_num_list}
                combined_dataset.append(combine_sample)

        # store combined dataset
        with open(os.path.join(self.cache_path, f"{self.model_name}_{self.dataset_type}_combine_text.json"), 'w') as file:
            json.dump(combined_dataset, file, indent=4, ensure_ascii=False)
    
        return
    
    def collate_fn(self, batch: dict)-> dict:
        """
        批处理函数，用于将多个样本组合成一个批次。
        :param batch: 一批样本，每个样本是一个dict。
        :return: 批次化的张量。
        """

        # 获取输入文本和输出文本
        # Shape: 2dim (batch_size, len(NOISE_DOCS_NUMS)=4 groups)
        input_text_list_batch = [sample['input_text_list'] for sample in batch]
        # Shape: 1dim (batch_size)
        output_text_batch = [sample['output_text'] for sample in batch]
        # Shape: 3dim (batch_size, len(NOISE_DOCS_NUMS)=4 groups, TOTAL_DOCS_NUM=5)
        output_class_num_list_batch = [sample['output_class_num_list'] for sample in batch]

        return {
            "input_text_list_batch": input_text_list_batch,
            "output_text_batch": output_text_batch,
            "output_class_num_list_batch": torch.tensor(output_class_num_list_batch)
        }
    
    def dataloader(self, batch_size, shuffle):
        with open(os.path.join(self.cache_path, f"{self.model_name}_{self.dataset_type}_combine_text.json"), 'r') as file:
            combined_dataset = json.load(file)
        
        # 将 list[dict] 转换为 Dataset
        dataset = Dataset.from_list(combined_dataset)

        # 构造 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

        return dataloader
        

# usage example
if __name__ == "__main__":
    MODEL_PATH_DICT = {"Qwen2.5-7B": "/datanfs2/zyx/model_cache/qwen/Qwen2___5-7B-Instruct", 
                       "Llama-3.1-8B": "/datanfs2/zyx/model_cache/LLM-Research/Meta-Llama-3___1-8B-Instruct"}

    model_names = ["Qwen2.5-7B", "Llama-3.1-8B"]
    types = ["train", "dev"]

    for model_name in model_names:
        for type in types:

            model_path = MODEL_PATH_DICT[model_name]
            dataset_path = f"/data1/wangyan/ModelFinetuning/dataset/origin_dataset/{type}.json"
            dataset_cache_dir_path = "/data1/wangyan/ModelFinetuning/dataset/dataset_cache"

            dataloader = DataManage(model_name, model_path, dataset_path, type, dataset_cache_dir_path)
            
            # running only once for specific [MODEL + Dataset(train/dev)]
            dataloader.load_origin_data_and_reorganize()
            dataloader.combine_reorganized_dataset_text()
            # dataloader.dataloader(batch_size=1, shuffle=False)

            print(f"DataManage for {model_name} {type} done!")
