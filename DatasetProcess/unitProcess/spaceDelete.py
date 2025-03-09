"""
remove some excess spaces between words and punctuation marks
eg: from "Who is he ( Bob ) ?" to "Who is he (Bob)?"
"""


import re


def delete_space(text: str):
    # delete the blank before ,.;!?":)-
    text = re.sub(r'\s([/%,.;!?:)\-\'])', r'\1', text)
    # delete the blank after (-
    text = re.sub(r'([/(\-])\s', r'\1', text)
    # delete the blank between "do" and "n't"
    text = re.sub(r"\b(do) (\w't)\b", r"\1\2", text, flags=re.IGNORECASE)
    return text.strip()