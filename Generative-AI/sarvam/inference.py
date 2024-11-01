## DataBase creation
from pypdf import PdfReader

reader = PdfReader("./data/iesc111.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=128, chunk_overlap=0
)

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = encoder.encode(token_split_texts)

import faiss

vector_dimension = vectors.shape[1]

index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

def get_prompt(query,top_k = 15, rank_k = 3):
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]


    PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    import numpy as np

    search_text = 'where is your office?'
    search_vector = encoder.encode(search_text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    distances, ann = index.search(_vector, k=top_k)

    from ragatouille import RAGPretrainedModel

    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    retrieved_context = [token_split_texts[idx] for idx in ann[0]]

    ranked_retrieved_context = RERANKER.rerank(search_text, retrieved_context,k=rank_k)

    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc['content'] for i, doc in enumerate(ranked_retrieved_context)]
    )

    final_prompt = PROMPT_TEMPLATE.format(
        question=search_text, context=context
    )

    return final_prompt

## Transformer Model Build
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
MODEL_NAME = "aisquared/dlite-v2-124m"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
    # , quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

## Query classification Question/Statement
import os
open_api_key="sk-M4nOjWlPG7P8VWJdwnm1T3BbdjsfbdkjjkjdhKLe6gbx1dm4CzZNc"
os.environ["OPENAI_API_KEY"]=open_api_key

from langchain.chat_models import ChatOpenAI

llm=ChatOpenAI(model_name='gpt-3.5-turbo')

from fastapi import FastAPI
# import uvicorn
from pydantic import BaseModel

class query_data(BaseModel):
    query: str


app = FastAPI()

@app.post('/predict')
def result(data: query_data):
    sentence = 'How does the sound travel'
    prompt = f"""
    Classify the given sentence as question if the sentence is interrogative otherwise classify as statement.
    For example
    Sentence: What is the capital of Russia
    Answer: question

    Sentence: Sun rises in the east
    Answer: statement

    Sentence: {sentence}
    Answer:
    """
    query_type = llm.predict(prompt) ##

    import joblib
    nb_model = joblib.load('./models/multinb.joblib')
    subject = nb_model.predict(sentence)[0]
    if query_type == 'question' and subject == 'Physics':
        final_prompt = get_prompt(sentence)
    else: final_prompt = sentence

    final_response = LLM(final_prompt)[0]["generated_text"]

    ## Add voice to Bot
    ### Using google text2speech
    from gtts import gTTS

    language='en'
    tts=gTTS(text=final_response,lang=language,slow=False, tld='co.in')
    tts.save("response.mp3")

    ### Using sarvam's API

    import requests

    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
        "inputs": [final_response],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    # headers = {"Content-Type": "application/json"}
    headers = {'API-Subscription-Key':'<your-api-key>'}

    response = requests.request("POST", url, json=payload, headers=headers)

    return {"predictions": f'{final_response}'}

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)