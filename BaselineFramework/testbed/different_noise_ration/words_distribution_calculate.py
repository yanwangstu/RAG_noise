import json
import random
from typing import Dict
from collections import defaultdict

# more than 3000
docs_length_over = set()


def calculate(input_file: str) -> None:
    # Define the word count intervals that needs to be counted
    intervals = []
    for start in range(0, 3000, 100):
        intervals.append((start, start + 99))
    intervals.append((3000, float('inf')))

    # store samples inside each word count intervals
    interval_counts = defaultdict(int)
    
    # load data
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for sample in data:
        total_docs_words = 0
        docs = sample["Retrieval Documents"]
        
        # calculate the word count of all docs in each sample
        for doc in docs:
            words = doc.split()
            word_count = len(words)
            total_docs_words += word_count
            
        # find out the word count interval for each sample
        for interval in intervals:
            if interval[0] <= total_docs_words <= interval[1]:
                interval_counts[interval] += 1
                break
    
    # Output interval statistical results
    print("Interval Statistics:")
    for interval in intervals:
        count = interval_counts.get(interval, 0)
        print(f"{interval[0]} - {interval[1]} words: {count} samples")
    print("\n")

# usage example
if __name__ == "__main__":

    rations = ["00", "01", "03", "05", "07", "09", "10"]


    for ration in rations:
        input_file=f"main_noise-ration-{ration}.json"
        print(input_file)
        calculate(input_file=input_file)
    print("docs_length_over", docs_length_over)