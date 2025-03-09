import json
import argparse
from unitProcess import questionAdjust
from unitProcess import documentGeneration
from unitProcess import answerAdjust
from unitProcess import spaceDelete

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

write_lock = Lock()
def save_json(new_data,filePathWrite):
    try:
        with write_lock:
             json.dump(new_data, open(filePathWrite, "w", encoding='utf-8'), indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"File lock acquisition failed: {e}")
    except Exception as e:
        print(f"Failed to write to file: {e}")

def call_API(data):
    try:
        data = json.loads(data)
        document_tokens = data['document_tokens']
        # long_answer_candidates_range is 2d
        long_answer_candidates_range = [[item["start_token"], item["end_token"]]
                                                    for item in data['long_answer_candidates']]
        # long_answer_range is 1d
        long_answer_range = [data["annotations"][0]["long_answer"]["start_token"],
                                data["annotations"][0]["long_answer"]["end_token"]]
        # short_answer_range is 2d
        short_answers_range = [[item["start_token"], item["end_token"]]
                                for item in data["annotations"][0]["short_answers"]]
        if short_answers_range != []:
            # generate the question
            origin_question = data["question_text"]
            question = questionAdjust.question_adjust(origin_question)

            # generate the answer 1d list
            answers = [answerAdjust.answer_adjust(document_tokens, answer_range)
                        for answer_range in short_answers_range]

            # generate the golden documents
            golden_document = documentGeneration.document_generation(document_tokens, long_answer_range)
            golden_documents = documentGeneration.golden_documents_generation(question, golden_document, answers)
            golden_documents.append(golden_document)

            # generate the distracting_documents
            distracting_documents = documentGeneration.distracting_documents_generation(question, golden_document, answers)

            # generate counterfactual and inconsequential documents
            transfer_documents, counterfactual_documents, inconsequential_documents = documentGeneration.counterfactual_inconsequential_documents_generation(
                question, document_tokens, long_answer_candidates_range, long_answer_range)
            
            if counterfactual_documents is not None:
                # write the generated samples into a new document
                jsonobj = {
                    "Question": question,
                    "Answers": answers,
                    "URL": data['document_url'],
                    "Golden Documents": golden_documents,
                    "Distracting Documents": distracting_documents,
                    "Inconsequential Documents": inconsequential_documents,
                    "Low Quality Documents": counterfactual_documents,
                    "Transfer Documents": transfer_documents
                }
                return jsonobj
    except Exception as e:
        print(f"\nFailed to generation: {str(e)}\n")
        return None
    
        
def data_process(filePathRead, filePathWrite):
    # Load all Line data
    with open(filePathRead, 'r') as fileRead:
        lines = fileRead.read().splitlines()
    pbar = tqdm(total=len(lines), desc=f"Parallel Data Processing")

    new_data_file = []
    # Set the maximum number of parallels
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_list = []
        for data in lines:
            future = executor.submit(call_API, data)
            future_list.append(future)

        for future in as_completed(future_list):
            try:
                new_data = future.result()
                if new_data is not None:
                    new_data_file.append(new_data)
                    if len(new_data_file) % 10 == 0:
                        save_json(new_data_file,filePathWrite)
            except Exception as e:
                print(f"\nFailed to question {json.loads(data)['question_text']}: {str(e)}\n")
                continue
            finally:
                pbar.update(1)

    save_json(new_data_file,filePathWrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=str, required=True, help='Number to replace "04" in file paths')
    args = parser.parse_args()

    filePathRead = f'NQ_dataset/v1.0/train/nq-train-{args.num}.jsonl'
    filePathWrite = f'newData/nq-train-{args.num}.json'

    # ==================================================== #
    # filePathRead = '/data1/wangyan/Dataset Process/originData/nq-train-sample.jsonl'
    # filePathWrite = '/data1/wangyan/Dataset Process/newData/nq-train-sample.json'
    data_process(filePathRead, filePathWrite)