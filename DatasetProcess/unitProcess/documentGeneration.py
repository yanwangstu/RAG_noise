"""
1. transform document (document_tokens: list, token_range: list) into correct format.
    1) combine the tokens into sentence
    2) replace the ``, and '' to " and delete the extra space in " "
    3) remove some excess spaces between words and punctuation marks

2. generate documents with different categories
    1) distracting document
    2) counterfactual and inconsequential documents
"""


import re
import json
import random
from unitProcess import spaceDelete
from unitProcess import LLM_API


# combine the tokens into sentence  through document_tokens (list) and token_range ([a, b])
def document_generation(document_tokens: list, token_range: list):
    sentence = ''
    start_token = token_range[0]
    end_token = token_range[1]
    for i in range(start_token, end_token + 1):
        token_info = document_tokens[i]
        token = token_info['token']
        # ignore HTML sign
        if not token_info.get('html_token', False):
            sentence += ' ' + token
    # replace the ``, and '' to " and delete the extra space in " "
    sentence = re.sub(r'``\s', r'"', sentence)
    sentence = re.sub(r'\s\'\'', r'"', sentence)
    # delete excess space and return
    return spaceDelete.delete_space(sentence)


# user prompt for generating Golden Documents and Distracting Documents
user_promptGD = ('''
Question: {question}
Short Answer: {answer}
Golden Paragraph: {golden_document}
''')


# system prompt for generating Golden Documents
system_promptG = '''
Task requirements:
Given a Question, a Short Answer and a Golden Paragraph, generate 10 new Golden Paragraphs (g0-g9) based on the following Principle:
a) You can delete some part of the given Golden Paragraph and add some new details, but you must retain the correct answer in the generate new Golden Paragraphs.
b) Reorganize the sentence structure of the entire paragraph (difference>80%), and change the active and passive voice/modifier position/rhetorical structure...
c) Ensure paragraph lengths match original through: ± 15% word count variation from Golden Paragraph.
d) Output strictly in the following example format, including all necessary quotes and escape characters.

Example Input:
Question: where is zimbabwe located in the world map?
Short Answer: in southern Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa, Botswana, Zambia and Mozambique.
Golden Paragraph: Zimbabwe, officially the Republic of Zimbabwe, is a landlocked country located in southern Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa, Botswana, Zambia and Mozambique. The capital and largest city is Harare. A country of roughly 16 million people, Zimbabwe has 16 official languages, with English, Shona, and Ndebele the most commonly used.

Example JSON Output (new Golden Paragraphs):
{
    "g0": "Zimbabwe, a landlocked nation in southern Africa, lies between the Zambezi and Limpopo Rivers. It shares borders with South Africa, Botswana, Zambia, and Mozambique.",
    "g1": "Located in southern Africa, Zimbabwe is bordered by South Africa, Botswana, Zambia, and Mozambique. This landlocked country spans the region between the Zambezi and Limpopo Rivers. Harare serves as its capital, and the nation boasts 16 official languages, with English, Shona, and Ndebele being widely spoken.",
    "g2": "Between the Zambezi and Limpopo Rivers lies Zimbabwe, a landlocked country in southern Africa with a population of approximately 16 million. Its neighbors include South Africa, Botswana, Zambia, and Mozambique. The capital and largest city of Zimbabwe is Harare. Zimbabwe has 16 official languages, with English, Shona, and Ndebele the most commonly used.",
    "g3": "In southern Africa, Zimbabwe is situated between the Zambezi and Limpopo Rivers. It is bordered by South Africa to the south, Botswana to the west, Zambia to the north, and Mozambique to the east. The capital city is Harare, and the country has 16 official languages, including English, Shona, and Ndebele.",
    "g4": "Zimbabwe is a landlocked country in southern Africa, positioned between the Zambezi and Limpopo Rivers. It is surrounded by South Africa, Botswana, Zambia, and Mozambique. The nation's capital is Harare, and its linguistic diversity includes English, Shona, and Ndebele as prominent languages.",
    "g5": "Surrounded by South Africa, Botswana, Zambia, and Mozambique, Zimbabwe is a landlocked country in southern Africa. It stretches between the Zambezi and Limpopo Rivers. Harare is the capital of Zimbabwe."
}
'''


def golden_documents_generation(question: str, golden_document: str, answer: list):
    # use prompt to generate golden_document
    user_input = user_promptGD.format(question=question, golden_document=golden_document, answer=answer)
    response = LLM_API.answerJSON(system_promptG, user_input)
    golden_documents = list(response.values())
    return golden_documents


# system prompt for generating Distracting Documents
system_promptD = '''
Task Requirements:
1. Given a Question, a Short Answer and a Golden Paragraph, generate 10 Distracting Paragraphs through these stages:
    a) Core Entity Substitution:
       - Identify question-critical entities in Short Answer
       - Replace with same-domain but incorrect entities using:
           • Theoretical: Keep discipline but change theory (e.g., "relativity→quantum mechanics") 
           • Event-based: Maintain event type but shift spatiotemporal coordinates (e.g., "2019→2023")
    b) Semantic Restructuring (>85% textual divergence):
       - Toggle active/passive voice
       - Each generated Distracting Paragraphs must be different in structure and sentence pattern.
       - Reconfigure rhetorical patterns (e.g., convert causal chains to parallel structures)
       - Reposition modifiers (e.g., transform prepositional phrases to relative clauses)
    c) Logical Consistency Verification:
       1. Ensure new entities contextually align with:
           - Geographic/chronological parameters
           - Domain-specific terminology
           - Quantitative relationships
       2. Preserve non-critical authentic details from Golden Paragraph for Distracting Documents
       3. Eliminate:
           - Cross-dimensional substitutions (animal→architecture terms)
           - Numerical contradictions (e.g., mismatched magnitude)

2. Output Format:
   - Generate 10 Distracting Paragraphs (d0-d9)
   - Each must contain question-related erroneous core entities
   - Maintain original JSON structure with proper escaping

3. Critical Avoidances:
   - Cross-category entity swaps
   - Internal chronological conflicts
   - Commonsense violations (e.g., "fish climb trees")

Example Input:
Question: where is zimbabwe located in the world map?
Short Answer: in southern Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa, Botswana, Zambia and Mozambique.
Golden Paragraph: Zimbabwe, officially the Republic of Zimbabwe, is a landlocked country located in southern Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa, Botswana, Zambia and Mozambique. The capital and largest city is Harare. A country of roughly 16 million people, Zimbabwe has 16 official languages, with English, Shona, and Ndebele the most commonly used.

Example JSON Output (Distracting Paragraph):
{
    "d0": "Zimbabwe, known officially as the Republic of Zimbabwe, is a country located in East Africa, positioned near the Nile River and surrounded by nations such as Kenya, Ethiopia, and Somalia. The nation is home to approximately 16 million residents, with its capital city being Harare. Among its 16 recognized languages, English, Shona, and Ndebele are the most widely spoken.",
    "d1": "The Republic of Zimbabwe is a coastal nation situated in West Africa, bordered by Senegal, Mali, and Niger. With a population of around 16 million people, Zimbabwe's official languages include French, Wolof, and Fulani. Its capital city, Harare, is renowned for its vibrant cultural heritage.",
    "d2": "Zimbabwe is a landlocked country found in North Africa, nestled between the Sahara Desert and the Mediterranean Sea. It shares borders with Egypt, Libya, and Algeria. Home to roughly 16 million inhabitants, the nation recognizes Arabic, Berber, and Hausa as its primary languages, while its capital remains Harare.",
    "d3": "Located in Central America, Zimbabwe is a tropical country bordered by Mexico, Guatemala, and Belize. With a population of approximately 16 million, the nation's official languages are Spanish, Mayan, and English. Harare serves as the capital city, known for its bustling markets and diverse culture.",
    "d4": "Zimbabwe, officially called the Republic of Zimbabwe, is an island nation in Southeast Asia, surrounded by Thailand, Vietnam, and Cambodia. The country has a population of about 16 million people and recognizes Khmer, Thai, and Vietnamese as its official languages. Its capital, Harare, is a major hub for trade and tourism.",
    "d5": "A country in Eastern Europe, Zimbabwe is bordered by Poland, Ukraine, and Romania. With a population nearing 16 million, the nation's primary languages include Polish, Ukrainian, and Romanian. Harare, the capital city, is famous for its historic architecture and rich traditions.",
    "d6": "Zimbabwe is a remote territory in the Arctic region, neighboring Greenland, Canada, and Iceland. Despite its harsh climate, the country supports a population of around 16 million. Its official languages are Inuit, Danish, and Icelandic, while Harare stands as the administrative center of this icy landscape.",
    "d7": "Positioned in South Asia's Deccan Plateau, Zimbabwe shares borders with India and Sri Lanka. The Godavari River irrigates its agricultural heartland, sustaining 16 million residents. Telugu and Tamil join English as official languages, with Harare established as capital in 1956.",
    "d8": "As a Middle Eastern nation bordering Jordan and Syria, Zimbabwe's semi-arid landscape supports 16 million inhabitants. Arabic and Kurdish prevail in official communications, while Harare's ancient citadel attracts archaeological interest. Water resource management remains a critical policy focus.",
    "d9": "Zimbabwe occupies Scandinavia's southern peninsula, adjoining Sweden and Norway. The Baltic Sea forms its eastern boundary, with Swedish and Norwegian used in government proceedings. Harare's urban population accounts for 75% of the nation's 16 million residents, concentrated in coastal cities."
}
'''


def distracting_documents_generation(question: str, golden_document: str, answers: list[str]):
    # use prompt to generate distracting_document
    user_input = user_promptGD.format(question=question, golden_document=golden_document, answer=answers)
    response = LLM_API.answerJSON(system_promptD, user_input)
    # ensure the distracting documents cannot answer the question
    distracting_documents = [item for item in list(response.values()) if item not in answers[0]]
    return distracting_documents


# system prompt for generating Counterfactual Documents
system_promptC = '''
Task requirements:
1. Given 7 paragraphs and semantically related Question, please generate a fine-tuned paragraph for each paragraph, by modifying relevant entities or adding relevant expressions to make it contain incorrect knowledge, while ensuring that they are still semantically relevant to the question.
2. Each generated paragraph must meet the following requirements:
   - Contains incorrect information, but should not violate common sense (for example: do not make obvious mistakes such as "birds swim in water").
   - No internal logical errors are allowed (for example: "The 2019 AFC Asian Cup was held in the United Arab Emirates, and Japan was announced as the host country on March 9, 2015" is self-contradictory).
   - Keep the paragraph fluent and grammatically correct.
   - Try not to include negative sentences in the output content.
3. The generated paragraph needs to be semantically relevant to the question, but should not include sentences that can directly answer the question.
4. Please output strictly in the following example format, including all necessary quotes and escape characters.

EXAMPLE INPUT:
{
    "Question": "Where is the capital of France?",
    "t0": "Paris, known for its famous landmarks like the Eiffel Tower.",
    "t1": "Paris is located in the northern part of France, and it's a major cultural and economic center.",
    "t2": "The population of Paris is over 2 million people, and it's one of the most visited cities in the world.",
    "t3": "Paris, is renowned for its art museums, including the Louvre and the Musée d'Orsay.",
    "t4": "France, officially the French Republic, is a country located primarily in Western Europe. Its overseas regions and territories include French Guiana in South America, Saint Pierre and Miquelon in the North Atlantic.",
    "t5": "France reached its political and military zenith in the early 19th century under Napoleon Bonaparte, subjugating part of continental Europe and establishing the First French Empire.",
    "t6": "France retains its centuries-long status as a global centre of art, science, and philosophy. It hosts the fourth-largest number of UNESCO World Heritage Sites and is the world's leading tourist destination, having received 100 million foreign visitors in 2023."
}

EXAMPLE JSON OUTPUT:
{
    "c0": "Lyon, known for its famous landmarks like the Eiffel Tower.",
    "c1": "Paris, located in the southern region of France, has been known for its recent economic decline.",
    "c2": "The population of Paris is estimated to be over 5 million, significantly larger than its actual population.",
    "c3": "The Louvre Museum, located in Paris, is a relatively new museum, built only in the 1990s.",
    "c4": "France, officially the French Republic, is a country located primarily in Eastern Europe. Its overseas regions and territories include French Guiana in South America, Saint Pierre and Miquelon in the North Atlantic.",
    "c5": "France reached its political and military zenith in the early 18th century under Napoleon Bonaparte, subjugating part of continental Europe and establishing the First French Empire.",
    "c6": "France retains its centuries-long status as a global centre of art, science, and philosophy. It hosts the largest number of UNESCO World Heritage Sites and is the world's leading tourist destination, having received 100 million foreign visitors in 2023.",
}
'''


# Generate counterfactual and inconsequential documents based on golden documents context
def counterfactual_inconsequential_documents_generation(
    question: str,
    document_tokens: list, 
    documents_range: list[list], 
    golden_documents_range: list
    )-> tuple[list[str] | None, list[str] | None, list[str] | None]:
    
    # Constants configuration
    MIN_CHARACTERS = 130
    TOTAL_DOCUMENTS_NEED = 14
    TRANSFER_DOCS_NEED = 7
    DIRECTIONS = (-1, 1)  # Left/Right search directions

    try:
        golden_index = documents_range.index(golden_documents_range)
    except ValueError:
        return None, None, None

    max_index = len(documents_range) - 1
    candidate_documents = []
    iteration = 1

    # Context document collection
    while len(candidate_documents) < TOTAL_DOCUMENTS_NEED:
        found_new = False
        
        for direction in DIRECTIONS:
            current_index = golden_index + direction * iteration
            if 0 <= current_index <= max_index:
                prev_index = current_index - direction
                if (direction == -1 and documents_range[current_index][1] <= documents_range[prev_index][0]) or \
                   (direction == 1 and documents_range[current_index][0] >= documents_range[prev_index][1]):
                    
                    doc = document_generation(document_tokens, documents_range[current_index])
                    if len(doc) >= MIN_CHARACTERS:
                        candidate_documents.append(doc)
                        found_new = True

        # Early termination check
        if not found_new and iteration > max(golden_index, max_index - golden_index):
            return None, None, None
        
        iteration += 1

    # Document processing
    random.shuffle(candidate_documents)
    transfer_docs = candidate_documents[:TRANSFER_DOCS_NEED]
    inconsequential_docs = candidate_documents[TRANSFER_DOCS_NEED:TOTAL_DOCUMENTS_NEED]

    # API call preparation
    user_input = {
        "Question": question,
        **{f"t{i}": doc for i, doc in enumerate(transfer_docs)}
    }
    response = LLM_API.answerJSON(system_promptC, json.dumps(user_input, ensure_ascii=False))
    counterfactual_documents = list(response.values())
    
    return transfer_docs, counterfactual_documents, inconsequential_docs
