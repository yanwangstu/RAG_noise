"""
transform answer (short answer) into correct format.
1) remove the stop words in the end of the answer
2) replace the ``, and '' to " and delete the extra space in " "
3) remove some excess spaces between words and punctuation marks
"""


import re
import spacy
from unitProcess import spaceDelete


# load spaCy English model
nlp = spacy.load("en_core_web_sm")
# 添加/删除自定义停用词
new_stopwords = ["(", ".", ",", ":"]  # 假设我们想添加一些自定义的停用词
numbers_to_remove = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
for word in new_stopwords:
    nlp.vocab[word].is_stop = True  # 将它们标记为停用词
for word in numbers_to_remove:
    nlp.vocab[word].is_stop = False



def stopword_delete(text: str):
    doc = nlp(text)
    stopWordCount = 0
    count = 0
    while True:
        count -= 1
        if doc[count].is_stop:
            stopWordCount += 1
        else:
            break
    # if no stop words in the end of the answer
    if stopWordCount == 0:
        filtered_tokens = [token.text for token in doc]
    # if stop words exists in the end of the answer
    else:
        filtered_tokens = [token.text for token in doc[0: len(doc)-stopWordCount]]
    sentence = " ".join(filtered_tokens)
    # replace the “, and ” to " and delete the extra space in " "
    sentence = re.sub(r'“\s', r'"', sentence)
    sentence = re.sub(r'\s”', r'"', sentence)
    return spaceDelete.delete_space(sentence)


def answer_adjust(document_tokens: list, token_range: list):
    sentence = ''
    start_token = token_range[0]
    end_token = token_range[1]
    for i in range(start_token, end_token + 1):
        token_info = document_tokens[i]
        token = token_info['token']
        # ignore HTML sign
        if not token_info.get('html_token', False):
            sentence += ' ' + token
    sentence = re.sub(r'``', r'“', sentence)
    sentence = re.sub(r'\'\'', r'”', sentence)
    return stopword_delete(sentence)


# Usage Example
if __name__ == '__main__':
    print("nlp.Defaults.stop_words", nlp.Defaults.stop_words)
    # 示例文本
    text = "the Shulchan Aruch"
    print(stopword_delete(text))