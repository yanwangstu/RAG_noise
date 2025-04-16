"""
construct the Question, Answer and Retrieval Database
"""


import sys
sys.path.append('/data1/wangyan/DatasetProcess')
import json
import argparse
from unitProcess import questionAdjust
from unitProcess import documentGeneration
from unitProcess import answerAdjust
from unitProcess import spaceDelete

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock


def dataGeneration(data):
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
    
    origin_question = data["question_text"]
    question = questionAdjust.question_adjust(origin_question)
    # generate the answer 1d list
    answers = [answerAdjust.answer_adjust(document_tokens, answer_range)
               for answer_range in short_answers_range]
    
    if len(answers) == 0:
        return None

    # generate the golden documents
    golden_document = documentGeneration.document_generation(document_tokens, long_answer_range)


    if len(golden_document) == 0:
        return None
    
    # generate noise documents
    noise_documents = []
    for doc_range in long_answer_candidates_range:
        if long_answer_range != doc_range:             
            doc = documentGeneration.document_generation(document_tokens, doc_range)
            noise_documents.append(doc)
    
    if len(noise_documents) < 100:
        return None
    
    documents = {0: golden_document}
    documents.update({i+1: noise_document for i, noise_document in enumerate(noise_documents)})
    
    jsonobj = {
        "Question": question,
        "Answers": answers,
        "URL": data['document_url'],
        "Docs Num": len(noise_documents)+1,
        "Docs": documents
    }
    
    return jsonobj
    
    
        
def data_process(filePathRead, filePathWrite):
    # Load all Line data
    new_data_file = []
    pbar = tqdm(total=200, desc=f"Data Processing")
    count = 0

    with open(filePathRead, 'r', encoding='utf-8') as file:
        for _, line in enumerate(file):
            new_data = dataGeneration(line)
            if new_data!=None:
                new_data = {
                    "QID": count, 
                    **new_data
                    }
                new_data_file.append(new_data)
                pbar.update(1)
                count += 1
            if count == 200:
                break

    with open(filePathWrite, 'w') as fileWrite:
        json.dump(new_data_file, fileWrite, indent=4)


if __name__ == "__main__":

    filePathRead = '/data1/wangyan/DatasetProcess/NQ_dataset/v1.0/train/nq-train-30.jsonl'
    filePathWrite = 'new_data.json'
    data_process(filePathRead, filePathWrite)