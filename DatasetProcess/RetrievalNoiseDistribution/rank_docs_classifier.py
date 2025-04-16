"""
classify the top 40 documents in the retrieval rank of BM25 or e5

based on "new_data.json"
which contains Question, Answer and Retrieval Database
"""



import sys
sys.path.append("/data1/wangyan/PoisonousMushroom")
import json
from GeneratorAPI import GeneratorAPI


# task_descripition
task_descripition = '''
Task Description:
You are given a **Question**, a **Short Answer**, and a **Document**. Your task is to classify the document into one of the following categories:
1. **Golden Document**: The document provides highly relevant and reliable information that directly supports the short answer.
2. **Distracting Document**: The document contains some relevant information but also includes misleading or unrelated content that may cause confusion.
3. **Low Quality Document**: The document is poorly written, contains incomplete or extraneous content (e.g., leftover HTML tags, excessive special symbols like ♠, %, unusual numeric formats like 3,539/sq mi), or lacks sufficient detail to be useful.
4. **Inconsequential Document**: The document contains some semantic relevant words and semantic relevant information (e.g., both the document and the question belong to the same topic) but does not contribute meaningfully to answering the question.
5. **Irrelevant Document**: The document has no connection to the question or the short answer in semantic level.(e.g., the document and the question belong to completely different topics)
Please output only the category number corresponding to the classification.
'''


# user prompt
user_prompt = ('''
Question: {question}
Answer: {answer}
Document: {document}
''')


# use LLM to classify the document
def documents_classifier(question: str, 
                         answer: str,
                         document: str
                         ) -> str:
    # 用于映射分类编号到文档类型
    category_mapping = {
        1: "Golden",
        2: "Distracting",
        3: "Low Quality",
        4: "Inconsequential",
        5: "Irrelevant"
    }

    # 生成用户输入
    user = user_prompt.format(question=question, 
                              answer=answer,
                              document=document)
    
    messages = [
        {"role": "system", "content": task_descripition},
        {"role": "user", "content": user},
    ]

    # 调用 API 生成分类编号
    response = GeneratorAPI(messages, "Deepseek-v3")
    response: str

    try:
        category_num = int(response)
        return category_mapping.get(category_num, f"Invalid Response: {response}")
    except ValueError:
        return f"Invalid Response: {response}"


if __name__ == '__main__':
    rank_result_file = "/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/rank_result.json"
    new_data_file = "/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/new_data.json"
    cateogry_info_file = "/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/top40_category.json"
    
    with open(rank_result_file, "r") as file:
        rank_result = json.load(file)
    with open(new_data_file, "r") as file:
        new_data = json.load(file)
    
    cateogry_info = []
    for rank_info, data in zip(rank_result, new_data):
        QID = rank_info["QID"]
        bm25_rank_top40_indice = rank_info["bm25_rank"][:40]
        e5_rank_top40_indice = rank_info["e5_rank"][:40]
        top40_indice_set = set(bm25_rank_top40_indice + e5_rank_top40_indice)

        question = data["Question"]
        answers = ','.join(data["Answers"]) if len(data["Answers"]) > 1 else data["Answers"][0]
        print(f"\n\n\nQID: {QID}")
        print(f"Question: {question}")
        print(f"len(top40_indice_set): {len(top40_indice_set)}\n")

        top40_category = {"QID": QID}
        for index in top40_indice_set:
            document = data["Docs"][str(index)]
            doc_category = documents_classifier(question, answers, document)
            print(f"index: {index}, category: {doc_category}")
            top40_category[index] = doc_category
        
        cateogry_info.append(top40_category)
        with open(cateogry_info_file, "w") as file:
            json.dump(cateogry_info, file, indent=4, ensure_ascii=False)
