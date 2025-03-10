# load testbed data, then use different framework to get the output
import sys
sys.path.append("..")
import json
import random
from Framework.VanillaRAG import *


if __name__ == "__main__":

    # load testbed data
    testbed_path = "../DatasetProcess/testbed/example.json"
    with open(testbed_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
    
    input_samples = []
    random.seed(64)
    for sample in samples:
        doc_types = ["Golden Documents", 
                    "Distracting Documents", 
                    "Inconsequential Documents", 
                    "Low Quality Documents", 
                    "Irrelevant Documents"]
        docs = []
        for doc_type in doc_types:
            docs += sample.get(doc_type, [])
        random.shuffle(docs)
        input_samples.append({"QID": sample["QID"], 
                              "Question": sample["Question"],
                              "Docs": docs})

    # use different framework to get the output and write into the result

    # use Llama-3.1-8B, use API = False
    instance = VanillaRAG("Llama-3.1-8B", False)

    result = []
    for input_sample in input_samples:
        # prepare the question and docs
        question = input_sample["Question"]
        docs = input_sample["Docs"]
        
        # LLM generation
        output = instance.inference(question, docs)
        result.append({"QID": input_sample["QID"], "Output Answer": output})

    result_record_path = "ExperimentResult/example.json"
    with open(result_record_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)