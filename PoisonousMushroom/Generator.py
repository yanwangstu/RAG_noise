from dataclasses import dataclass
from torch import FloatTensor, LongTensor, Tensor
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerateDecoderOnlyOutput
import torch


# constant
MAX_NEW_TOKENS = 600
MIN_NEW_TOKENS = 1
MODEL_PATH_DICT = {"Qwen2.5-7B": "/datanfs2/zyx/model_cache/qwen/Qwen2___5-7B-Instruct", 
                   "Llama-3.1-8B": "/datanfs2/zyx/model_cache/LLM-Research/Meta-Llama-3___1-8B-Instruct",
                   "DeepSeek-R1-Distill-Llama-8B": "/datanfs2/zyx/model_cache/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}


@dataclass
class GeneratorOutput:
    """
    output_attention: tuple[FloatTensor*5_dim]
    (num_layer, batch_size, num_heads, generated_length, sequence_length)

    output_scores: tuple[FloatTensor*2_dim]
    (new_len_tokens, batch_size, vocab_size)
    """
    output_text: str
    output_tokens: list[str] | None
    output_attention: tuple[tuple[FloatTensor]] | None
    output_scores: tuple[FloatTensor] | None


class Generator:
    def __init__(self, model_name: str, device: str|None=None):
        """
        when the model_name is None, 
        only use the message_generate function and use API to get the response
        """
        if model_name is not None:
            # set GPU device
            self.device = device

            # choose the model path through model name
            model_path = MODEL_PATH_DICT[model_name]

            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="bfloat16",
                device_map=self.device,
                attn_implementation="eager")
            self.model.config.pad_token_id = self.model.config.eos_token_id

    # SKR -- message generate: Check if adding information is necessary
    def message_generate_additionINFO(self, question: str):
        task_descripition = """Task Description: 
        Do you need additional information to answer this question?
        if you need, please answer: \"Yes, I need.\"
        if you don't need, please answer: \"No, I don’t need.\"
        Do not answer questions and explain reasons.
        """

        formatted_question = f"Question: {question}"
        
        messages = [
                {"role": "system", "content": task_descripition},
                {"role": "user", "content": formatted_question},
            ]
        return messages

            
    # No RAG QA -- input message generate
    def message_generate_base(self, question: str) -> list[dict]:
        task_descripition = """Task Description: 
        1. Answer the given Question Directly, do NOT add any explanations when giving the response.
        2. If you cannot answer with certainty due to insufficient information, you MUST respond verbatim:  \"I cannot answer the question.\"
        """

        formatted_question = f"Question: {question}"
        
        messages = [
                {"role": "system", "content": task_descripition},
                {"role": "user", "content": formatted_question},
            ]
        return messages

    # RAG QA (without Chain-of-Note) -- input message generate
    def message_generate_baseRAG(self, question: str, docs: list[str]) -> list[dict]:
        task_descripition = """Task Description: 
        1. Answer the given Question based on the Retrieval Documents, do NOT add any explanations when giving the response.
        2. If you cannot answer with certainty due to insufficient information, you MUST respond verbatim:  \"I cannot answer the question.\"
        """
        
        formatted_question = f"Question: {question}"
        formatted_docs = "Retrieval Documents:\n" + "\n".join(
                [f"{i + 1}. {doc}" for i, doc in enumerate(docs)])
        
        messages = [
                {"role": "system", "content": task_descripition},
                {"role": "user", "content": formatted_question},
                {"role": "user", "content": formatted_docs},
            ]
        return messages

    # Chain-of-Note RAG QA -- input message generate
    def message_generate_CoNRAG(self, question: str, docs: list[str]) -> list[dict]:
        task_descripition = """Task Description:
        1. Read the given Question and Retrieval Documents to gather relevant information.
        2. Write reading notes summarizing the key points from these passages.
        3. Discuss the relevance of the given question and Wikipedia passages.
        4. If some passages are relevant to the given question, provide a brief answer based on the passages.
        5. If no passage is relevant, directly provide answer without considering the passages.
        6. If you cannot answer with certainty due to insufficient information, you MUST respond verbatim:  \"I cannot answer the question.\"
        """

        formatted_question = f"Question: {question}"
        formatted_docs = "Retrieval Documents:\n" + "\n".join(
                [f"{i + 1}. {doc}" for i, doc in enumerate(docs)])
        
        messages = [
                {"role": "system", "content": task_descripition},
                {"role": "user", "content": formatted_question},
                {"role": "user", "content": formatted_docs},
            ]
        return messages

    # generate the output based on the message
    # if return_info is True, addition info beyond output_text can be extracted
    def generator(
            self,
            messages: list[dict],
            return_info: bool
    ) -> GeneratorOutput:

        # transfer messages into input_text
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # transfer input_text into input_token_id_tensor & attention mask
        # eg: {'input_ids': tensor([[128000, 271, ...]]), 'attention_mask': tensor([[1, 1, ...]])}
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        input_ids = model_inputs.input_ids
        input_ids_length = input_ids.shape[1]
        # print("input_ids_length", input_ids_length)

        # generate output_text directly
        if not return_info:
            # generate output_token_id_tensor through input_token_id_tensor
            outputs_ids = self.model.generate(
                **model_inputs,
                # new_tokens excludes the input token, only generated tokens is taken into consideration
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.eos_token_id,
                min_new_tokens=MIN_NEW_TOKENS,
                use_cache=True
            )[0, input_ids_length:]
            outputs_ids: LongTensor

            # transfer output_token_id_tensor into output_text
            output_text = self.tokenizer.decode(
                outputs_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return GeneratorOutput(
                output_text=output_text,
                output_tokens=None,
                output_attention=None,
                output_scores=None
            )

        # generate output_text and other info
        else:
            outputs = self.model.generate(
                **model_inputs,
                return_dict_in_generate=True,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.eos_token_id,
                min_new_tokens=MIN_NEW_TOKENS,
                output_scores=True,
                use_cache=False
            )
            outputs: GenerateDecoderOnlyOutput

            # transfer output_token_id_tensor into output_token (1_dim tensor)
            outputs_ids = outputs.sequences[0, input_ids_length:]
            output_tokens = self.tokenizer.convert_ids_to_tokens(outputs_ids.tolist())

            # transfer output_token_id_tensor into output_text
            output_text = self.tokenizer.decode(
                outputs_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # get output scores of each token
            outputs_scores = outputs.scores

            # get the output attention through re-input the output into the LLM
            outputs_attention = self.model(outputs.sequences[:, input_ids_length:], output_attentions=True).attentions
            # outputs_attention = [layer_attn[-1] for layer_attn in outputs.attentions]

            torch.cuda.empty_cache()

            return GeneratorOutput(
                output_text=output_text,
                output_tokens=output_tokens,
                output_attention=outputs_attention,
                output_scores=outputs_scores
            )



# usage example
if __name__ == "__main__":
    # prepare the question and docs
    question = "In greek mythology who was the goddess of spring growth?"
    docs = ["In Greek mythology, Aphrodite (/ˌæfrəˈdaɪti/; Greek: Ἀφροδίτη), also called Cytherea (/sɪθəˈriːə/; 'the foam-born'), is the daughter of Zeus and Dione and is the queen of love and beauty. Homer describes her as the enchanting, graceful goddess of desire, who influences the hearts of gods and mortals alike. Aphrodite was married to Hephaestus, the god of fire and craftsmanship. The myth of her birth from the sea foam represents her function as the personification of beauty and love, which blossoms in spring and fades with the changing seasons; hence, she is also associated with spring as well as the allure of nature. Similar myths appear in the Orient, in the cults of female deities like Ishtar, Astarte, and Inanna, and in ancient Mesopotamia.",
            "In Greek mythology, Artemis (/ˈɑːrtɪmɪs/; Greek: Ἄρτεμις), also called Cynthia (/ˈsɪnθiə/; 'the moon goddess'), is the daughter of Zeus and Leto and is the queen of the hunt and wilderness. Homer describes her as the fierce, independent protector of the natural world, who roams the forests with her bow and arrows. Artemis was associated with Apollo, the god of the sun and prophecy. The myth of her birth on the island of Delos represents her function as the personification of the untamed wilderness, which thrives in spring and recedes in winter; hence, she is also associated with spring as well as the vitality of nature. Similar myths appear in the Orient, in the cults of hunting deities like Diana, Skadi, and Arduinna, and in ancient Anatolia.",
            "In Greek mythology, Hestia (/ˈhɛstiə/; Greek: Ἑστία), also called Vesta (/ˈvɛstə/; 'the hearth goddess'), is the daughter of Cronus and Rhea and is the queen of the hearth and home. Homer describes her as the gentle, protective guardian of domestic life, who maintains the sacred fire. Hestia was associated with Zeus, the king of the gods. The myth of her eternal virginity represents her function as the personification of the hearth, which warms in spring and sustains through the seasons; hence, she is also associated with spring as well as the comfort of home. Similar myths appear in the Orient, in the cults of hearth deities like Vesta, Brigid, and Hestia, and in ancient Rome."
            ]

    # load "Llama-3.1-8B" or "Qwen2.5-7B"
    Gen = Generator("Llama-3.1-8B", "cuda:0")

    # Basic RAG message generate
    messages  = Gen.message_generate_baseRAG(question, docs)
    # print("input: ", messages)

    # LLM generation
    output = Gen.generator(messages, False)
    print("output: ", output.output_text)
