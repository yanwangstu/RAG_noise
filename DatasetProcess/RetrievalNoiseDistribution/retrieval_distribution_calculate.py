"""
analysis the retrieval documents distribution under the follwoing settings:

BM25:    top5, top10, top20
e5:      top5, top10, top20
BM25+e5: top5, top10, top20

(BM25+e5 first find top 35 through BM25, then find top_k through e5)

based on "new_data.json"
which contains Question, Answer and Retrieval Database
"""
import json


rank_result_file = "/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/JSON/rank_result.json"
# RAG_data_file = "/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/new_data.json"
rank_docs_cateogry_info_file = "/data1/wangyan/DatasetProcess/RetrievalNoiseDistribution/JSON/top40_category.json"


def ave_distribution_calculate(category_count_list: list[dict]):
    total_samples = len(category_count_list)
    
    ave_distribution = {
        "Golden": 0,
        "Distracting": 0,
        "Low Quality": 0,
        "Inconsequential": 0,
        "Irrelevant": 0
    }
    
    for category_count in category_count_list:
        total_docs = sum(category_count.values()) - category_count.get("QID", 0)
        if total_docs == 0:
            continue
        
        for key in ave_distribution.keys():
            ave_distribution[key] += category_count[key] / total_docs
    
    for key in ave_distribution.keys():
        ave_distribution[key] /= total_samples
    
    return ave_distribution




# retrival_method = "bm25" or "e5"
def single_retrival_distribution(top_k: int, retrival_method: str):
    # load the file
    with open(rank_result_file, "r") as file:
        rank_result = json.load(file)
    with open(rank_docs_cateogry_info_file, "r") as file:
        rank_docs_cateogry_info = json.load(file)

    category_count_list = []
    for rank, category in zip(rank_result, rank_docs_cateogry_info):
        rank_top_k_indice = rank[f"{retrival_method}_rank"][:top_k]
        category_count = {
        "QID": rank["QID"],
        "Golden": 0,
        "Distracting": 0,
        "Low Quality": 0,
        "Inconsequential": 0,
        "Irrelevant": 0
        }
        for index in rank_top_k_indice:
            category_name = category[str(index)]
            if not "Invalid Response" in category_name:
                category_count[category_name] += 1
        category_count_list.append(category_count)

    return ave_distribution_calculate(category_count_list)


# retrival_method is combine "bm25" and "e5"
def mix_retrival_distribution(top_k: int):
    # load the file
    with open(rank_result_file, "r") as file:
        rank_result = json.load(file)
    with open(rank_docs_cateogry_info_file, "r") as file:
        rank_docs_cateogry_info = json.load(file)

    category_count_list = []
    for rank, category in zip(rank_result, rank_docs_cateogry_info):
        BM25_rank_top_35_indice = rank["bm25_rank"][:35]
        e5_rank_indice = [idx for idx in rank["e5_rank"] if idx in BM25_rank_top_35_indice]
        e5_rank_top_k_indice = e5_rank_indice[:top_k]

        category_count = {
        "QID": rank["QID"],
        "Golden": 0,
        "Distracting": 0,
        "Low Quality": 0,
        "Inconsequential": 0,
        "Irrelevant": 0
        }
        for index in e5_rank_top_k_indice:
            category_name = category[str(index)]
            if not "Invalid Response" in category_name:
                category_count[category_name] += 1
        category_count_list.append(category_count)

    return ave_distribution_calculate(category_count_list)


if __name__ == "__main__":
    # use BM25 and e5
    top_k_list = [5, 10, 20]
    methods = ["bm25", "e5"]
    
    for method in methods:
        for top_k in top_k_list:
            result = single_retrival_distribution(top_k, method)
            print("-"*40)
            print("\n")
            print(f"Retrieval Method:{method}")
            print(f"Top_k: {top_k}")
            for key, value in result.items():
                print(f"{key}: {value:.4f}")
            print("\n")


    # use mix method
    for top_k in top_k_list:
        result = mix_retrival_distribution(top_k)
        print("-"*40)
        print("\n")
        print(f"Retrieval Method: MIX")
        print(f"Top_k: {top_k}")
        for key, value in result.items():
            print(f"{key}: {value:.4f}")
        print("\n")

        


        




    



