# load testbed data
# then use different framework and backbone to get the result and store it
import sys
sys.path.append("..")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from tqdm import tqdm
from Framework.NoRAG import *
from Framework.VanillaRAG import *
from Framework.CoNRAG import *
from Framework.SKR import *
from Framework.DRAGIN import *


# use specific framework to get the output and write into the result
def experiment(framework: str, 
               model_name: str, 
               device: str, 
               useAPI: bool, 
               testbed_path: str,
               result_record_path: str
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
    generation_instance = RAGClass(model_name, device, useAPI)

    # generate the result
    result = []
    pbar = tqdm(total=len(input_samples), 
                desc="Result Generation", 
                mininterval=10, 
                maxinterval=15)
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
        if input_sample["QID"]%20 == 0:
            with open(result_record_path, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=4)

    # write the experiment result
    with open(result_record_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":

    # hyperparameter setting
    experiment_name = "main"
    variable_name = "noise-ration"
    framework_name = "ChainofNote"
    model_name = "Llama-3.1-8B"
    useAPI = False
    variables = ["xx", "01", "03", "05", "07", "09", "10"]
    device = "cuda:3"

    # start
    for variable in variables:

        parse_model_name = model_name.replace(".", "-")
        testbed_path = f"testbed/{experiment_name}_different_{variable_name}/main_{variable_name}-{variable}.json"
        result_record_path = f"ExperimentResult/model_output/{experiment_name}_different_{variable_name}/{framework_name}/{experiment_name}_{variable_name}-{variable}_{framework_name}_{parse_model_name}.json"
    
        print("\n")
        print("Settings: ")
        print("framework_name: ", framework_name)
        print("model_name: ", model_name)
        print("useAPI: ", useAPI)
        print("testbed_path: ", testbed_path)
        print("result_record_path: ", result_record_path)
        print("\n")

        experiment(framework_name, 
                model_name, 
                device,
                useAPI, 
                testbed_path, 
                result_record_path)
    
    """
    # experiments for different-noise-ration

        # hyperparameter setting
        framework_name = "VanillaRAG"
        model_name = "Llama-3.1-8B"
        useAPI = False
        noise_ration = "10"
        device = "cuda:0"

        parse_model_name = model_name.replace(".", "-")
        testbed_path = f"testbed/different_noise_ration/main_noise-ration-{noise_ration}.json"
        result_record_name = f"main_noise-ration-{noise_ration}_{framework_name}_{parse_model_name}.json"

        print("\n")
        print("Settings: ")
        print("framework_name: ", framework_name)
        print("model_name: ", model_name)
        print("useAPI: ", useAPI)
        print("testbed_path: ", testbed_path)
        print("result_record_name: ", result_record_name)
        print("\n")

        experiment(framework_name, 
                model_name, 
                device,
                useAPI, 
                testbed_path, 
                result_record_name)
    """

"""
    model_name,   useAPI

    Llama-3.1-8B,       False
    Qwen2.5-7B,         False
    Gemma2-9B,          False (Not yet enabled)

    Qwen2.5-72B,        True
    Deepseek-v3,        True
    Deepseek-v3-ppin,   True
    Llama-3.1-70B,      True  (Not yet enabled)

    ** DRAGIN can only use useAPI == False models
"""

"""
    framework_name:
    "NoRAG", "VanillaRAG", "ChainofNote":, "DRAGIN":, "SKR"
"""