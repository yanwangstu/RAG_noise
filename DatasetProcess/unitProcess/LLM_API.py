import os
import json
from openai import OpenAI

'''
# 使用阿里云
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    api_key='sk-775b1e7c8a4d4a10b84508b0d6be1ba1',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
'''

# Set the LLM's API-Key and URL
client = OpenAI(
    # **商汤大装置 API Info**
    base_url="https://api.sensenova.cn/compatible-mode/v1",
    # wangyan-key:	
    api_key = "sk-pijLcCU8uwb2icud9fb3rKI4lheOxWJU",
    # zhangyx-key:  
    # api_key = "sk-ylZHAjPZufNCGXOoWgar5tdzUnio2Lbk",
    # **派欧云 API Info**
    # base_url="https://api.ppinfra.com/v3/openai",
    # 请通过 https://ppinfra.com/settings#key-management 获取 API 密钥。
    # api_key="sk_2rwMr_cIXz02Ss7vi3yELfOP1J2D1DBPotAh6sDkvq8",
)


# model = "qwen-plus"
# model = "qwen-max"
# model = "deepseek-v3"
model = "DeepSeek-V3"
# model = 'DeepSeek-R1'
# model = "deepseek-r1"
# model = "deepseek/deepseek-v3/community"


# input a question, return the answer
def answer(system: str, user: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
        ]
    )
    # print(response.choices[0].message.content)
    # print(response.usage)
    cleaned_text = response.choices[0].message.content.replace("<｜end▁of▁sentence｜>", "")
    return cleaned_text


# input a question, return the JSON answer and usage
def answerJSON(system: str, user: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
        ],
        response_format={
            'type': 'json_object'
        }
    )
    # print(response.choices[0].message.content)
    # print(response.usage)
    cleaned_text = response.choices[0].message.content.replace("<｜end▁of▁sentence｜>", "")
    cleaned_text = cleaned_text.replace("```json", "")
    cleaned_text = cleaned_text.replace("```", "")
    return json.loads(cleaned_text)


if __name__ == "__main__":
    system = 'You are an assistant'
    user = 'what is 1+1?'
    print(answer(system, user))
