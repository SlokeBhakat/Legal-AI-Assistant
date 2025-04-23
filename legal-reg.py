

from datasets import load_dataset

ds = load_dataset("hugsid/legal-contracts")


ds["train"].save_to_disk("content/legal-contracts")



from datasets import load_from_disk


dataset = load_from_disk("content/legal-contracts")


print(dataset)
print(dataset[0])



dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 50)



import re

def clean_text(text):

    text = re.sub(r'Page \d+ of \d+', '', text)


    text = text.encode('ascii', 'ignore').decode()

    text = re.sub(r'\n\s*\n', '\n\n', text)


    text = re.sub(r'[ \t]+', ' ', text)


    text = re.sub(r'(?<=\n)([A-Z\s]{3,})(?=\n)', lambda m: m.group(0).title(), text)


    return text.strip()




cleaned_dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})
cleaned_dataset



from datasets import DatasetDict

split_dataset = cleaned_dataset.train_test_split(test_size=0.1)


def chunk_text(batch, chunk_size=512):
    chunked = []

    for text in batch["text"]:
        words = text.split()
        chunks = [
            {"text": " ".join(words[i:i + chunk_size])}
            for i in range(0, len(words), chunk_size)
        ]
        chunked.extend(chunks)

    return {"text": [c["text"] for c in chunked]}


chunked_dataset = split_dataset["train"].map(
    chunk_text,
    batched=True,
    remove_columns=["text"]
)



import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_passages(batch):
    batch["embedding"] = model.encode(batch["text"], show_progress_bar=False).tolist()
    return batch

embedded_dataset = chunked_dataset.map(embed_passages, batched=True)






import hnswlib
import numpy as np


embeddings = np.array(embedded_dataset["embedding"])
texts = embedded_dataset["text"]


dim = embeddings.shape[1]
num_elements = len(embeddings)

p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=100, M=16)

p.add_items(embeddings)


p.set_ef(50)


id_to_text = {i: texts[i] for i in range(num_elements)}


def search(query, top_k=5):
    query_embedding = model.encode([query])
    labels, distances = p.knn_query(query_embedding, k=top_k)
    return [(id_to_text[i], 1 - dist) for i, dist in zip(labels[0], distances[0])]


results = search("arbitration or dispute resolution clause", top_k=3)


for idx, (text, score) in enumerate(results):
    print(f"\n🔹 Result {idx+1} (Score: {score:.4f}):\n{text[:500]}...")








from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate


import os
os.environ["GROQ_API_KEY"] = "groq_api_key"


llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")


def format_context(docs):
    return "\n\n".join([f"{i+1}. {doc}" for i, (doc, _) in enumerate(docs)])


def ask_legal_bot(question):
    relevant_docs = search(question, top_k=3)
    context = format_context(relevant_docs)

    prompt = f"""You are a legal assistant. Use the following legal documents to answer the question:

{context}

Q: {question}
A:"""

    messages = [
        SystemMessage(content="You are a helpful legal assistant."),
        HumanMessage(content=prompt)
    ]

    response = llm(messages)
    return response.content


ask_legal_bot("What happens if the borrower defaults?")



ask_legal_bot("Can either party terminate the contract early?")


ask_legal_bot("What penalties are there for breach of contract?")


