import sys
sys.path.append("/data1/wangyan/PoisonousMushroom")
from Generator import Generator, GeneratorOutput
from GeneratorAPI import GeneratorAPI


class NoRAG:
    def __init__(self, model_name: str, device: str|None, useAPI: bool):
        self.useAPI = useAPI
        self.model_name = model_name
        if not useAPI:
            self.generator = Generator(model_name, device)
        else:
            self.generator = Generator(None)

    def inference(self, question: str) -> dict:
        messages = self.generator.message_generate_base(question)

        if not self.useAPI:
            output = self.generator.generator(messages, False)
            output: GeneratorOutput
            return {"Output Answer": output.output_text}
        
        else:
            output = GeneratorAPI(messages, self.model_name)
            output: str
            return {"Output Answer": output}


# usage example
if __name__ == "__main__":

    # use Llama-3.1-8B, use API = False
    instance = NoRAG("Llama-3.1-8B", False)

    # use Qwen2.5-72B, use API = True
    # instance = NoRAG("Qwen2.5-72B", True)

    # prepare the question
    question = "In greek mythology who was the goddess of spring growth?"
    
    # LLM generation
    output = instance.inference(question)
    print(output)
