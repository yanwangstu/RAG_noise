# load testbed data
# then use different framework and backbone to get the result and store it
import sys
sys.path.append("..")
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock
from Framework.NoRAG import *
from Framework.VanillaRAG import *
from Framework.CoNRAG import *
from Framework.SKR import *
from Framework.DRAGIN import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


write_lock = Lock()

def serial_run(input_samples: list[dict],
               framework: str,
               generation_instance: ):
    for input_sample in input_samples:
        # prepare the question and docs
        question = input_sample["Question"]
        docs = input_sample["Retrieval Documents"]
        
        # LLM generation
        if framework != "NoRAG":
            output = generation_instance.inference(question, docs)
        else:
            output = generation_instance.inference(question)
        
        # result record
        result.append({"QID": input_sample["QID"]} | output)

        pbar.update(1)

        # write the experiment result during the experiment
        if input_sample["QID"]%50 == 0:
            with open(result_record_path, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=4)

        # write the experiment result
        with open(result_record_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)



# use specific framework to get the output and write into the result
def experiment(framework: str, 
               model_name: str, 
               useAPI: bool, 
               testbed_path: str,
               result_record_name: str
               ):
    
    framework_classes = {
        "NoRAG": NoRAG,
        "VanillaRAG": VanillaRAG,
        "ChainofNote": CoNRAG,
        "DRAGIN": DRAGIN,
        "SKR": SKR
    }

    # load input_samples
    with open(testbed_path, 'r', encoding='utf-8') as file:
        input_samples = json.load(file)

    # Get the class and instantiated it
    if framework not in framework_classes:
        raise ValueError(f"Unsupported framework: {framework}")
    RAGClass = framework_classes[framework]
    generation_instance = RAGClass(model_name, useAPI)

    # generate the result
    result = []
    result_record_path = f"ExperimentResult/{result_record_name}"
    pbar = tqdm(total=len(input_samples), desc="Result Generation")

    # useAPI is False -- Local serial operation
    if useAPI is False:
        for input_sample in input_samples:
            # prepare the question and docs
            question = input_sample["Question"]
            docs = input_sample["Retrieval Documents"]
            
            # LLM generation
            if framework != "NoRAG":
                output = generation_instance.inference(question, docs)
            else:
                output = generation_instance.inference(question)
            
            # result record
            result.append({"QID": input_sample["QID"]} | output)

            pbar.update(1)

            # write the experiment result during the experiment
            if input_sample["QID"]%50 == 0:
                with open(result_record_path, 'w', encoding='utf-8') as file:
                    json.dump(result, file, ensure_ascii=False, indent=4)

        # write the experiment result
        with open(result_record_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

        return

    # useAPI is True -- Remote parallel operation
    else:
        return



if __name__ == "__main__":
    
    # experiments for different-noise-ration

    # hyperparameter setting
    framework_name = "VanillaRAG"
    model_name = "Llama-3.1-8B"
    useAPI = False
    noise_ration = "03"
    GPU_device = "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_device
    testbed_path = f"testbed/different_noise_ration/main_noise-ration-{noise_ration}.json"
    result_record_name = f"main_noise-ration-{noise_ration}_{framework_name}_{model_name.replace(".", "-")}.json"

    print("Settings: ")
    print("framework_name: ", framework_name)
    print("model_name: ", model_name)
    print("useAPI: ", useAPI)
    print("testbed_path: ", testbed_path)
    print("result_record_name: ", result_record_name)
    print("\n")

    experiment(framework_name, 
               model_name, 
               useAPI, 
               testbed_path, 
               result_record_name)

"""
    model_name,   useAPI

    Llama-3.1-8B, False
    Qwen2.5-7B,   False
    Gemma2-9B,    False (Not yet enabled)

    Qwen2.5-72B,  True
    Deepseek-v3,  True
    Llama-3.1-70B,True (Not yet enabled)

    ** DRAGIN can only use useAPI == False models
"""

"""
    framework_name:
    "NoRAG", "VanillaRAG", "ChainofNote":, "DRAGIN":, "SKR"
"""