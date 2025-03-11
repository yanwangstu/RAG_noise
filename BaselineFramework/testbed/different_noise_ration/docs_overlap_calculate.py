import json
import random
from typing import Dict


def calculate(input_file: str) -> None:
    # load origin sampels
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for sample in data:
        
        documents = sample["Retrieval Documents"]

        new = set(documents)

        m = len(documents)-len(new)
        if m !=0:
            print(sample["QID"])
            print("overlap", len(documents)-len(new))

    return


# usage example
if __name__ == "__main__":

    rations = ["00", "01", "03", "05", "07", "09", "10"]

    for ration in rations:
        print(ration)
        input_file=f"main_noise_ration_{ration}.json"
        calculate(input_file=input_file)