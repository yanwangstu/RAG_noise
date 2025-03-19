from openai import OpenAI


# constant
MAX_NEW_TOKENS = 600
MODEL_NAME_DICT = { # 钱多多 https://api2.aigcbest.top/token
                    "Llama-3.1-70B-zyx": 
                   {"base_url": "https://api2.aigcbest.top/v1", 
                    "model_id": "llama-3.1-70b-instruct", 
                    "api_key": "sk-477y5ZJSVHyqrQflyJU00JIzt49D4AK5W2i0R1qvWpR3uT3V"},

                    "Llama-3.1-70B-wy": 
                   {"base_url": "https://api2.aigcbest.top/v1", 
                    "model_id": "llama-3.1-70b-instruct", 
                    "api_key": "sk-TeV2RS0MSKVioCfN8oKZUq1kd4zy7Ue0BDl8Lp34OD6kF0Hc"},


                    # (付费版)阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market/detail/qwen2.5-72b-instruct
                   "Qwen2.5-72B": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "qwen2.5-72b-instruct", 
                    "api_key": "sk-775b1e7c8a4d4a10b84508b0d6be1ba1"},

                    # (付费版)阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "Deepseek-v3": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-v3", 
                    "api_key": "sk-775b1e7c8a4d4a10b84508b0d6be1ba1"},

                    # (付费版)阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "Deepseek-r1": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1", 
                    "api_key": "sk-775b1e7c8a4d4a10b84508b0d6be1ba1"},

                    # (免费版) RPM=60 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "R1-distill-llama-8b-wy": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1-distill-llama-8b", 
                    "api_key": "sk-775b1e7c8a4d4a10b84508b0d6be1ba1"},

                    # (免费版) RPM=60 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "R1-distill-llama-70b-wy": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1-distill-llama-70b", 
                    "api_key": "sk-775b1e7c8a4d4a10b84508b0d6be1ba1"},

                    # (免费版) RPM=60 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "R1-distill-llama-8b-zyx": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1-distill-llama-8b", 
                    "api_key": "sk-1c8027e671284f66a0abea9573b6c122"},

                    # (免费版) RPM=60 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "R1-distill-llama-70b-zyx": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1-distill-llama-70b", 
                    "api_key": "sk-1c8027e671284f66a0abea9573b6c122"},

                    # (免费版) RPM=60 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "R1-distill-llama-8b-hjh": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1-distill-llama-8b", 
                    "api_key": "sk-f59545233c6940718a4f01b8e716eefd"},

                    # (免费版) RPM=60 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market
                   "R1-distill-llama-70b-hjh": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "deepseek-r1-distill-llama-70b", 
                    "api_key": "sk-f59545233c6940718a4f01b8e716eefd"},
                    
                    # (付费版)派欧算力云 https://ppinfra.com
                   "Deepseek-v3-ppin": 
                   {"base_url": "https://api.ppinfra.com/v3/openai", 
                    "model_id": "deepseek/deepseek-v3-turbo", 
                    "api_key": "sk_2rwMr_cIXz02Ss7vi3yELfOP1J2D1DBPotAh6sDkvq8"}}


# generate the output through LLM API
def GeneratorAPI(messages: list[dict], model_name: str) -> str:
    base_url = MODEL_NAME_DICT[model_name]["base_url"]
    model_id = MODEL_NAME_DICT[model_name]["model_id"]
    api_key = MODEL_NAME_DICT[model_name]["api_key"]

    client = OpenAI(base_url=base_url, api_key =api_key)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"\n\nResponse Failed: {str(e)}")
        return f"Response Failed: {str(e)}"



# usage example
if __name__ == "__main__":

    from Generator import Generator

    # prepare the question and docs
    question = "In greek mythology who was the goddess of spring growth?"
    docs = ["In Greek mythology, Aphrodite (/ˌæfrəˈdaɪti/; Greek: Ἀφροδίτη), also called Cytherea (/sɪθəˈriːə/; 'the foam-born'), is the daughter of Zeus and Dione and is the queen of love and beauty. Homer describes her as the enchanting, graceful goddess of desire, who influences the hearts of gods and mortals alike. Aphrodite was married to Hephaestus, the god of fire and craftsmanship. The myth of her birth from the sea foam represents her function as the personification of beauty and love, which blossoms in spring and fades with the changing seasons; hence, she is also associated with spring as well as the allure of nature. Similar myths appear in the Orient, in the cults of female deities like Ishtar, Astarte, and Inanna, and in ancient Mesopotamia."] 

    # Basic RAG message generate
    Gen = Generator(None)
    messages  = Gen.message_generate_baseRAG(question, docs)
    print("input: ", messages)

    # LLM generation
    import time
    start = time.time()
    output = GeneratorAPI(messages, "Llama-3.1-70B")
    end = time.time()
    print("output: ", output)
    print("total_time", end-start)
