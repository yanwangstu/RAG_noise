"""
split train dataset(about 4,5000) and dev dataset(about 500) for finetuning
split from: DatasetProcess/dataset_add_irrelevant/level_2_20000/level_2_20000_2.json
"""
import os
import json

if __name__ == "__main__":
    origin_json_file_path = "/data1/wangyan/DatasetProcess/dataset_add_irrelevant/level_2_20000/level_2_20000_2.json"
    dataset_dir_path = "/data1/wangyan/ModelFinetuning/dataset"

    # load json file
    with open(origin_json_file_path, "r") as f:
        data = json.load(f)

    # shuffle data
    import random
    random.seed(42)
    random.shuffle(data)

    # split data into train and dev
    train_data = data[:5400]
    dev_data = data[5400:]

    # add new QID for train and dev data
    for i in range(len(train_data)):
        train_data[i] = {
            "subID": i,
            **train_data[i],
        }
    for i in range(len(dev_data)):
        dev_data[i] = {
            "subID": i,
            **dev_data[i],
        }

    # save train and dev data to json files
    with open(os.path.join(dataset_dir_path, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(dataset_dir_path, "dev.json"), "w") as f:
        json.dump(dev_data, f, indent=4, ensure_ascii=False)