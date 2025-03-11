"""
add irrelevant documents into the [dir newData] (our reconstructed data)
then store the finetuned dataset in [dir finetunedData]
"""
import argparse

import os
import json
import random
from tqdm import tqdm


def get_all_json_file_paths(directory):
    # construct a blank list to store the path of .json file
    json_file_paths = []
    # use os.walk to tranverse the dir and its subdir
    for root, dirs, files in os.walk(directory):
        for file in files:
            # check whether the file is ended with .json
            if file.endswith(".json"):
                # concate the filename and current dir path to get the full path
                full_path = os.path.join(root, file)
                # add the full path into the list
                json_file_paths.append(full_path)
    return json_file_paths


def load_all_files(filePathsRead):
    file_contents = {}
    for file_path in filePathsRead:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"load file: {file_path}")
            file_contents[file_path] = json.load(file)
    print("load_all_files: all files have been loaded.")
    return file_contents


def irrelevant_documents_generation(file_contents: dict, currentFileRead: str, irrelevant_documents_num: int) -> list[str]:
    irrelevant_documents = []
    CANDIDATE_CATEGORY = ["Golden Documents", "Distracting Documents", "Inconsequential Documents",
                          "Low Quality Documents"]

    local_filePathsRead = list(file_contents.keys())
    local_filePathsRead.remove(currentFileRead)

    for _ in range(irrelevant_documents_num):
        random_file = random.choice(local_filePathsRead)
        data = file_contents[random_file]
        random_sample = random.choice(data)
        random_category = random.choice(CANDIDATE_CATEGORY)
        document = random.choice(random_sample[random_category])
        irrelevant_documents.append(document)

    return irrelevant_documents


def data_finetuning(filePathsRead: str, filePathWrite: str, irrelevant_documents_num: int) -> None:
    # Cache all read file contents at the beginning
    file_contents = load_all_files(filePathsRead)
    print("preprossed files used to costruct finetuned dataset(add irrelevant documents)", filePathsRead)

    new_data = []
    id_count = 0
    file_count = 0
    for fileRead in filePathsRead:
        data = file_contents[fileRead]
        data_num = len(data)
        pbar = tqdm(total=data_num, desc=f"Samples in {fileRead} already Processed")

        for i in range(data_num):
            irrelevant_documents = irrelevant_documents_generation(file_contents, fileRead, irrelevant_documents_num)
            filtered_QID_data = {k: v for k, v in data[i].items() if k != "QID"}
            new_dict = {
                "QID": id_count,
                **filtered_QID_data,
                "Irrelevant Documents": irrelevant_documents,
            }
            new_data.append(new_dict)
            pbar.update(1)
            id_count += 1
            if id_count%6000 == 0:
                json.dump(new_data, open(filePathWrite.format(ID=file_count), "w", encoding='utf-8'), indent=4, ensure_ascii=False)
                new_data = []
                file_count += 1

    if id_count%6000 != 0:
        json.dump(new_data, open(filePathWrite.format(ID=file_count), "w", encoding='utf-8'), indent=4, ensure_ascii=False)
        
    print(f"Total data num: {id_count}")


if __name__ == "__main__":

    # read files list
    FILE_PATHS_READ = [f'dataset/nq-train-{i:02d}.json' for i in range(35)]
    # write files template
    FILE_PATH_WRITE = 'dataset_add_irrelevant/level_3_50000/level_3_50000_{ID}.json'
    IRRELEVANT_DOCUMENTS_NUM = 7
    data_finetuning(FILE_PATHS_READ, FILE_PATH_WRITE, IRRELEVANT_DOCUMENTS_NUM)
