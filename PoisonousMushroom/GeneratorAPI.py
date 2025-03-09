from openai import OpenAI


# constant
MAX_NEW_TOKENS = 500
MODEL_NAME_DICT = {"Llama-3.1-70B": 
                   {"base_url": "", 
                    "model_id": "", 
                    "api_key": ""},

                    # 阿里云百炼 https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.27ba516ecCpHkQ&accounttraceid=f54baffa43d5411a8c27405d70adde94qoxh#/model-market/detail/qwen2.5-72b-instruct
                   "Qwen2.5-72B": 
                   {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    "model_id": "qwen2.5-72b-instruct", 
                    "api_key": "sk-775b1e7c8a4d4a10b84508b0d6be1ba1"},

                    # sensenova
                   "Deepseek-v3": 
                   {"base_url": "https://api.sensenova.cn/compatible-mode/v1", 
                    "model_id": "DeepSeek-V3", 
                    "api_key": "sk-pijLcCU8uwb2icud9fb3rKI4lheOxWJU"}}


# generate the output through LLM API
def GeneratorAPI(messages: list[dict], model_name: str) -> str:
    base_url = MODEL_NAME_DICT[model_name]["base_url"]
    model_id = MODEL_NAME_DICT[model_name]["model_id"]
    api_key = MODEL_NAME_DICT[model_name]["api_key"]

    client = OpenAI(base_url=base_url, api_key =api_key)

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
    )

    return response.choices[0].message.content


# usage example
if __name__ == "__main__":
    from Generator import Generator, GeneratorOutput

    # prepare the question and docs
    question = "In greek mythology who was the goddess of spring growth?"
    docs = ["In Greek mythology, Aphrodite (/ˌæfrəˈdaɪti/; Greek: Ἀφροδίτη), also called Cytherea (/sɪθəˈriːə/; 'the foam-born'), is the daughter of Zeus and Dione and is the queen of love and beauty. Homer describes her as the enchanting, graceful goddess of desire, who influences the hearts of gods and mortals alike. Aphrodite was married to Hephaestus, the god of fire and craftsmanship. The myth of her birth from the sea foam represents her function as the personification of beauty and love, which blossoms in spring and fades with the changing seasons; hence, she is also associated with spring as well as the allure of nature. Similar myths appear in the Orient, in the cults of female deities like Ishtar, Astarte, and Inanna, and in ancient Mesopotamia.",
            "In Greek mythology, Artemis (/ˈɑːrtɪmɪs/; Greek: Ἄρτεμις), also called Cynthia (/ˈsɪnθiə/; 'the moon goddess'), is the daughter of Zeus and Leto and is the queen of the hunt and wilderness. Homer describes her as the fierce, independent protector of the natural world, who roams the forests with her bow and arrows. Artemis was associated with Apollo, the god of the sun and prophecy. The myth of her birth on the island of Delos represents her function as the personification of the untamed wilderness, which thrives in spring and recedes in winter; hence, she is also associated with spring as well as the vitality of nature. Similar myths appear in the Orient, in the cults of hunting deities like Diana, Skadi, and Arduinna, and in ancient Anatolia.",
            "In Greek mythology, Hestia (/ˈhɛstiə/; Greek: Ἑστία), also called Vesta (/ˈvɛstə/; 'the hearth goddess'), is the daughter of Cronus and Rhea and is the queen of the hearth and home. Homer describes her as the gentle, protective guardian of domestic life, who maintains the sacred fire. Hestia was associated with Zeus, the king of the gods. The myth of her eternal virginity represents her function as the personification of the hearth, which warms in spring and sustains through the seasons; hence, she is also associated with spring as well as the comfort of home. Similar myths appear in the Orient, in the cults of hearth deities like Vesta, Brigid, and Hestia, and in ancient Rome."
            ] 

    # Basic RAG message generate
    Gen = Generator(None)
    messages  = Gen.message_generate_baseRAG(question, docs)
    print("input: ", messages)

    # LLM generation
    output = GeneratorAPI(messages, "Qwen2.5-72B")
    print("output: ", output)
