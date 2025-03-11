# evaluate the experiment result based on the correct answer
import sys
sys.path.append("..")
from PoisonousMushroom.GeneratorAPI import *
import json
import tqdm
from typing import Tuple


# calculate the answer reject rate
# and identify the answer reject of each sample
def answer_reject_calculate(llm_answers: dict, 
                            correct_answers: dict
                            ) -> Tuple[float, dict]:
    
    total_samples = len(correct_answers)
    answer_reject_samples = 0
    answer_reject_dict = {}
    for QID in correct_answers:
        llm_answer = llm_answers[QID]
        if "I cannot answer" in llm_answer:
            answer_reject_samples += 1
            answer_reject_dict[QID] = True
        else:
            answer_reject_dict[QID] = False
    
    answer_reject_rate = answer_reject_samples/total_samples
    return answer_reject_rate, answer_reject_dict


# generate the message for LLM to identify the accuracy of each sample
def message_generate(question: str, 
                     llm_answer: str, 
                     correct_answer: str
                     ) -> list[dict]:
    
    task_descripition = """Task Description: 
    1. 
    2. 
    """
    user_content = f"""Question: {question}
Correct Answer: {correct_answer}
Candidate Answer: {llm_answer}
"""
    
    messages = [
            {"role": "system", "content": task_descripition},
            {"role": "user", "content": user_content},
        ]
    return messages

# calculate the accuracy
# and identify the accuracy of each sample
def accuracy_calculate(question: str, 
                       llm_answers: dict, 
                       correct_answers: dict
                       ) -> Tuple[float, dict]:
    
    # 补全 并行调用deepseek-v3
    accuracy = 0
    accuracy_dict = correct_answers
    
    return accuracy, accuracy_dict


def evaluation(llm_answers_path: str,
               correct_answers_path: str,
               evaluator_write_path: str
               ) -> None:
    
    # load llm_answers
    with open(llm_answers_path, 'r', encoding='utf-8') as file:
        llm_answers = json.load(file)
    llm_answers = {item['QID']: item['Output Answer'] 
                   for item in llm_answers}

    # load correct_answers and questions
    with open(correct_answers_path, 'r', encoding='utf-8') as file:
        correct_answers = json.load(file)

    questions = {item['QID']: item["Question"]
                for item in correct_answers}
    correct_answers = {item['QID']: item["Answers"] 
                    for item in correct_answers}
    
    answer_reject_rate, answer_reject_dict = answer_reject_calculate(llm_answers, correct_answers)
    accuracy, accuracy_dict = accuracy_calculate(questions, llm_answers, correct_answers)

    overall_evaluation = [{"Total acc": accuracy, 
                          "Total answer_reject_rate": answer_reject_rate}]

    each_sample_evaluation = [{"QID": QID,
                               "Answer Reject": answer_reject_dict[QID],
                               "Acc": accuracy_dict[QID]} 
                               for QID in answer_reject_dict]

    with open(evaluator_write_path, 'w', encoding='utf-8') as file:
        json.dump(overall_evaluation+each_sample_evaluation, file, ensure_ascii=False, indent=4)
    
    return


if __name__ == "__main__":
    # set hyperparameters
    llm_answers_path = "ExperimentResult/main_noise-ration-00_VanillaRAG_Llama-3-1-8B.json"
    correct_answers_path = "testbed/different_noise_ration/main_noise-ration-00.json"
    evaluator_write_path = "ExperimentResult/evaluation_result/main_noise-ration-00_VanillaRAG_Llama-3-1-8B.json"
    
    # start evaluation
    evaluation(llm_answers_path, correct_answers_path, evaluator_write_path)
