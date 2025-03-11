import json
import random
from typing import Dict

# constant
RANDOM_SEED = 42
# the total word count of docs in a sample should be lower than MAX_DOCS_WORDS
MAX_DOCS_WORDS = 2000
random.seed(RANDOM_SEED)


def filter_QID_search(input_files: list) -> set:
    docs_types = ["Golden Retrieves",
                  "Noise Retrieves"]
    filter_QID = set()

    for input_file in input_files:
        # load data
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        for sample in data:
            total_docs_words = 0

            # calculate the word count of all docs in each sample
            for doc_type in docs_types:
                docs = sample[doc_type]
                for doc in docs:
                    words = doc.split()
                    total_docs_words += len(words)

            if total_docs_words > MAX_DOCS_WORDS:
                filter_QID.add(sample["QID"])
    
    return filter_QID


def testbed_construction_shuffle(
        input_file: str, 
        output_file: str, 
        filtered_QID: set
        ) -> None:
    # load origin sampels
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    new_data = []

    for sample in data:
        if sample["QID"] not in filtered_QID:
            new_sample = {}

            new_sample["QID"] = sample["QID"]
            new_sample["Question"] = sample["Question"]
            new_sample["Answers"] = sample["Answers"]
            new_sample["URL"] = sample["URL"]
        
            documents = sample["Golden Retrieves"] + sample["Noise Retrieves"]
            random.shuffle(documents)
            new_sample["Retrieval Documents"] = documents

            new_data.append(new_sample)

    # write processed data
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    return


# usage example
if __name__ == "__main__":

    rations = ["00", "01", "03", "05", "07", "09", "10"]
    input_files = [f"main_n_{ration}_10.json" for ration in rations]
    output_files = [f"main_noise-ration-{ration}.json" for ration in rations]

    filter_QID = filter_QID_search(input_files)
    print("filter_QID length: ", len(filter_QID))
    print("filter_QID: ", filter_QID)

    for input_file,  output_file in zip(input_files, output_files):
        testbed_construction_shuffle(
            input_file=input_file,
            output_file=output_file,
            filtered_QID=filter_QID
            )
        print("success: ", output_file)