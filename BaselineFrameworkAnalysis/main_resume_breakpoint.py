# load testbed data
# then use different framework and backbone to get the result and store it
import sys
sys.path.append("..")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import time
import argparse
from tqdm import tqdm
from Framework.NoRAG import *
from Framework.VanillaRAG import *
from Framework.CoNRAG import *
from Framework.SKR import *
from Framework.DRAGIN import *
from datetime import datetime, timedelta


# use specific framework to get the output and write into the result
def experiment_resume_breakpoint(framework: str, 
               model_name: str, 
               device: str, 
               useAPI: bool, 
               testbed_path: str,
               result_record_path: str,
               last_write_QID: int
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

    # reload the previous result
    with open(result_record_path, 'r', encoding='utf-8') as file:
        result = json.load(file)

    # generate the result
    pbar = tqdm(total=len(input_samples), 
                desc="Result Generation",
                mininterval=10, 
                maxinterval=15)
    pbar.update(len(result))

    already_processed_QID = [item["QID"] for item in result]
    already_processed_answer = {item["QID"]: item["Output Answer"] for item in result}

    for input_sample in input_samples:
        if input_sample["QID"] > last_write_QID:
            if input_sample["QID"] in already_processed_QID:
                # ifresponse failed, then retry
                if "Response Failed: Error code: 429" not in already_processed_answer[input_sample["QID"]]:
                    pbar.update(1)
                    continue
                else:
                    # delete the previous response result
                    result = [item for item in result if item.get("QID") != input_sample["QID"]]

            # prepare the question and docs
            question = input_sample["Question"]
            docs = input_sample["Retrieval Documents"]

            try:
                # LLM generation
                if framework != "NoRAG":
                    output = generation_instance.inference(question, docs)
                else:
                    output = generation_instance.inference(question)
            except Exception as e:
                print(f"\n\nResponse Failed: {str(e)}")
                output = {"Output Answer": f"Response Failed: {str(e)}"}
                time.sleep(60)
            
            # result record
            result.append({"QID": input_sample["QID"]} | output)
            pbar.update(1)

            # write the experiment result during the experiment
            if int(pbar.n)%20 == 0:
                with open(result_record_path, 'w', encoding='utf-8') as file:
                    json.dump(result, file, ensure_ascii=False, indent=4)

    # write the experiment result
    with open(result_record_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":

    # hyperparameter parse
    parser = argparse.ArgumentParser()
    # add hyperparameter
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--variable_name", type=str, required=True)
    parser.add_argument("--framework_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--last_write_qid", type=int, required=True)
    parser.add_argument("--use_api", type=str, required=True)
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()

    # hyperparameter setting
    experiment_name = args.experiment_name
    variable_name = args.variable_name
    framework_name = args.framework_name
    model_name = args.model_name
    last_write_QID = args.last_write_qid
    if args.use_api == "True":
        useAPI = True
    if args.use_api == "False":
        useAPI = False
    variable = args.variable
    if args.device == "None":
        device = None
    else:
        device = args.device

    # start
    parse_model_name = model_name.replace(".", "-")
    parse_model_name = parse_model_name.strip("-zyx")
    parse_model_name = parse_model_name.strip("-hjh")
    parse_model_name = parse_model_name.strip("-wy")

    testbed_path = f"testbed/{experiment_name}_different_{variable_name}/{experiment_name}_{variable_name}-{variable}.json"
    result_record_path = f"ExperimentResult/model_output/{experiment_name}_different_{variable_name}/{framework_name}/{experiment_name}_{variable_name}-{variable}_{framework_name}_{parse_model_name}.json"

    now = datetime.now()
    formatted_start_time = now.strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    pid = os.getpid()

    print("\n")
    print("------Break Point Restart------")
    print("Settings")
    print("framework_name: ", framework_name)
    print("model_name: ", model_name)
    print("useAPI: ", useAPI)
    print("GPU Device: ", device)
    print("testbed_path: ", testbed_path)
    print("result_record_path: ", result_record_path)
    print("PID: ", pid)
    print("Last Write QID: ", last_write_QID)
    print("Start Time: ", formatted_start_time)
    print("\n")

    experiment_resume_breakpoint(framework_name, 
           model_name, 
           device,
           useAPI, 
           testbed_path, 
           result_record_path,
           last_write_QID)
    
    end_time = time.time()
    now = datetime.now()
    formatted_end_time = now.strftime("%Y-%m-%d %H:%M:%S")
    delta = timedelta(seconds=end_time-start_time)
    print("\n")
    print("End Time: ", formatted_end_time)
    print("Time Cost: ", delta)
    print("\n")


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
