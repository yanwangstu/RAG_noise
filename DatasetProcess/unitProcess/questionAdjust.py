"""
transform question into correct format.
1) tune the with all upper-lower case into correct format
2) add "?" in the end of the sentence
3) remove some excess spaces between words and punctuation marks
"""

import spacy
from unitProcess import spaceDelete


# load spaCy English Model
nlp = spacy.load("en_core_web_sm")


# case adjust
def question_adjust_case(text: str):
    doc = nlp(text)
    result = []
    sentence_start = True  # mark whether it is the start of the sentence
    for token in doc:
        if sentence_start:
            # 句首单词首字母大写
            result.append(token.text.capitalize())
            sentence_start = False  # 句子开始后，标记为 False
        elif token.pos_ == 'PROPN':
            # 专有名词首字母大写
            result.append(token.text.capitalize())
        elif token.pos_ == 'NOUN' and token.ent_type_ != '':
            # 实体中的名词首字母大写
            result.append(token.text.capitalize())
        elif token.pos_ == 'PRON' and token.text.lower() == 'i':
            # 将人称代词 "i" 转换为大写的 "I"
            result.append('I')
        else:
            result.append(token.text)
        # if token is the sign of ending, we set the  sentence_start = True
        if token.text in ('.', '!', '?'):
            sentence_start = True
    # 将处理后的单词重新连接成句子
    return ''.join([f" {token}" for token in result])


def question_adjust(text: str):
    question = question_adjust_case(text) + '?'
    return spaceDelete.delete_space(question)


# Usage Example
if __name__ == '__main__':
    text = "how is the head of the church of england"
    text2 = "where is zimbabwe located in the world map?"
    text3 = "who sings the song i don't care i love it"
    "Who sings the song i do n't care i love it?"
    question_adjust_text = question_adjust(text3)
    print(f"Text after the analysis: {question_adjust_text}")