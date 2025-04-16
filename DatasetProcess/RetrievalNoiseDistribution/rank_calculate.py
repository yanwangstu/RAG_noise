"""
calculate the retrieval rank of BM25 and e5

based on "new_data.json"
which contains Question, Answer and Retrieval Database
"""

import json
import time
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class Retriever:
    def __init__(self):
        # load e5 model
        self.e5 = SentenceTransformer('intfloat/e5-small')

    def rank_e5(self,
                question: str,
                docs: list[str]
                ) -> list[str]:
        # generate embedding for question and docs
        question_embedding = self.e5.encode(question, convert_to_tensor=True).cpu().numpy()
        docs_embeddings = self.e5.encode(docs, convert_to_tensor=True).cpu().numpy()

        # calculate cos_sim
        cos_sim = cosine_similarity(question_embedding.reshape(1, -1), docs_embeddings).flatten()
        ranked_indices = cos_sim.argsort()[::-1]
        return ranked_indices.tolist()

    def rank_bm25(self,
                  question: str,
                  docs: list[str]
                  ) -> list[str]:
        tokenized_docs = [re.findall(r'\w+', doc.lower()) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_question = re.findall(r'\w+', question.lower())
        scores = bm25.get_scores(tokenized_question)

        ranked_indices = scores.argsort()[::-1]
        return ranked_indices.tolist()


# calculate the docs through BM25 and e5
if __name__ == "__main__":
    data_path_read = '/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/new_data.json'
    data_path_write = '/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/rank_result.json'

    with open(data_path_read, 'r', encoding='utf-8') as readfile:
        data = json.load(readfile)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("load model start")

    retrieval = Retriever()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("load model completed")
    pbar = tqdm(total=len(data))

    print("start ranking")
    rank_results = []
    for item in data:
        question = item['Question']
        docs = list(item['Docs'].values())
        e5_rank = retrieval.rank_e5(question, docs)
        bm25_rank = retrieval.rank_bm25(question, docs)
        rank_result = {
            'QID': item['QID'],
            'e5_rank': e5_rank,
            'bm25_rank': bm25_rank
        }
        rank_results.append(rank_result)
        with open(data_path_write, 'w', encoding='utf-8') as writefile:
            json.dump(rank_results, writefile, ensure_ascii=False, indent=4)
        pbar.update(1)

    with open(data_path_write, 'w', encoding='utf-8') as writefile:
        json.dump(rank_results, writefile, ensure_ascii=False, indent=4)
    
    print("\n", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("write completed")
