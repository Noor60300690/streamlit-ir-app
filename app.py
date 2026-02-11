import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
embeddings = np.load("embeddings.npy")

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

st.title("Information Retrieval System")

query = st.text_input("Enter your search query:")

if query:
    query_embedding = np.random.rand(512).astype(np.float32)  # dummy query embedding
    results = retrieve_top_k(query_embedding, embeddings)

    for doc, score in results:
        st.write(doc.strip(), "Score:", float(score))
