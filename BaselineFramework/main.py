# load testbed data
# then use different framework and backbone to get the result and store it
import sys
sys.path.append("..")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import time
import asyncio
import argparse
from tqdm import tqdm
from Framework.NoRAG import *
from Framework.VanillaRAG import *
from Framework.CoNRAG import *
from Framework.SKR import *
from Framework.DRAGIN import *
from datetime import datetime, timedelta
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor


# hyperparameter parse
parser = argparse.ArgumentParser()
# add hyperparameter
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--variable_name", type=str, required=True)
parser.add_argument("--framework_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--use_api", type=str, required=True)
parser.add_argument("--rpm_limit", type=int, required=True)
parser.add_argument("--variables", nargs="+", required=True)
parser.add_argument("--device", type=str, required=True)
args = parser.parse_args()
# hyperparameter setting
experiment_name = args.experiment_name
variable_name = args.variable_name
variables = args.variables
framework_name = args.framework_name
model_name = args.model_name
if args.use_api == "True":
    useAPI = True
if args.use_api == "False":
    useAPI = False
if args.device == "None":
    device = None
else:
    device = args.device
RPM_LIMIT = args.rpm_limit

executor = ThreadPoolExecutor()
# when RPM or TPM limit exceed error occures
# sleep 1 min
error_occurred = False


@sleep_and_retry
@limits(calls=RPM_LIMIT, period=60)
async def async_inference(framework: str, 
                          generation_instance,
                          question: str, 
                          docs: dict[str]|None,
                          pbar,
                          condition
                          )->str:
    global error_occurred
    loop = asyncio.get_event_loop()

    async with condition:
        # when RPM or TPM limit exceed error occures
        # sleep 1 min
        while error_occurred:
            await condition.wait()

        if framework != "NoRAG":
            output = await loop.run_in_executor(executor, generation_instance.inference, question, docs)
        else:
            output = await loop.run_in_executor(executor, generation_instance.inference, question)
        pbar.update(1)

        if "Response Failed: Error code: 429" in output["Output Answer"]:
            # Annotate error cooures
            error_occurred = True
            # Notify all pending tasks
            condition.notify_all()
            await asyncio.sleep(60)  # Wait 60s
            print("\nSleep 60s\n")
            error_occurred = False  # Reset error status
            # Notify again that all pending tasks can continue
            condition.notify_all()
        return output


async def async_inference_batch(input_samples: list[dict], 
                                framework: str,
                                generation_instance,
                                pbar):
    results = []
    tasks = []
    condition = asyncio.Condition()

    for input_sample in input_samples:
        # prepare the question and docs
        question = input_sample["Question"]
        docs = input_sample["Retrieval Documents"]
        
        # creating an asynchronous task
        task = asyncio.create_task(async_inference(
            framework, 
            generation_instance,
            question,
            docs if framework != "NoRAG" else None,
            pbar,
            condition
            ))
        tasks.append(task)
    
    # LLM generation
    results_of_tasks = await asyncio.gather(*tasks)

    for input_sample, output in zip(input_samples, results_of_tasks):
        results.append({"QID": input_sample["QID"]} | output)
    
    return results


# use specific framework to get the output and write into the result
def experiment(framework: str, 
               model_name: str, 
               device: str|None, 
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
    results = []
    pbar = tqdm(total=len(input_samples), 
                desc="Result Generation", 
                mininterval=10, 
                maxinterval=15)
    
    #  if useAPI is True
    if useAPI is True:
        batch_size = 80
        for i in range(0, len(input_samples), batch_size):
            sub_input_samples = input_samples[i: i+batch_size]
            sub_results = asyncio.run(async_inference_batch(
                sub_input_samples, 
                framework,
                generation_instance,
                pbar))
            results += sub_results
            
            # write the experiment result
            with open(result_record_path, 'w', encoding='utf-8') as file:
                json.dump(results, file, ensure_ascii=False, indent=4)

    # if useAPI is False
    else:
        for input_sample in input_samples:
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
            # result record
            results.append({"QID": input_sample["QID"]} | output)
            pbar.update(1)

            # write the experiment result during the experiment
            if int(pbar.n)%20 == 0:
                with open(result_record_path, 'w', encoding='utf-8') as file:
                    json.dump(results, file, ensure_ascii=False, indent=4)

        # write the experiment result
        with open(result_record_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":

    print("------Running Total Setting------")
    print("Experiment Name", experiment_name)
    print("Variable Name", variable_name)
    print("Variable List", variables)
    print("Framework Name: ", framework_name)
    print("Model Name: ", model_name)
    print("useAPI: ", useAPI)
    print("GPU Device: ", device)
    print("---------------------------------\n")



    # start running
    parse_model_name = model_name.replace(".", "-")
    parse_model_name = parse_model_name.strip("-zyx")
    parse_model_name = parse_model_name.strip("-hjh")
    parse_model_name = parse_model_name.strip("-wy")
    for variable in variables:

        testbed_path = f"testbed/{experiment_name}_different_{variable_name}/{experiment_name}_{variable_name}-{variable}.json"
        result_record_path = f"ExperimentResult/model_output/{experiment_name}_different_{variable_name}/{framework_name}/{experiment_name}_{variable_name}-{variable}_{framework_name}_{parse_model_name}.json"
        
        
        now = datetime.now()
        formatted_start_time = now.strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        pid = os.getpid()

        print("\n")
        print("------Running Start------")
        print("Settings")
        print("framework_name: ", framework_name)
        print("model_name: ", model_name)
        print("useAPI: ", useAPI)
        print("GPU Device: ", device)
        print("testbed_path: ", testbed_path)
        print("result_record_path: ", result_record_path)
        print("Start Time: ", formatted_start_time)
        print("PID :", pid)
        print("\n")

        experiment(framework_name, 
                model_name, 
                device,
                useAPI, 
                testbed_path, 
                result_record_path)
        
        end_time = time.time()
        now = datetime.now()
        formatted_end_time = now.strftime("%Y-%m-%d %H:%M:%S")
        delta = timedelta(seconds=end_time-start_time)
        print("\n")
        print("End Time: ", formatted_end_time)
        print("Time Cost: ", delta)
        print("\n")

        if useAPI is True:
            time.sleep(60)


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
    "NoRAG", "VanillaRAG", "ChainofNote", "DRAGIN", "SKR"
"""