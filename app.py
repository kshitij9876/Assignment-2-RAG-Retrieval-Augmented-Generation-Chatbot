from flask import Flask, request, jsonify
import mysql.connector
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Sample corpus
corpus = """Retrieval-Augmented Generation (RAG) is a technique that grants generative artificial intelligence models information retrieval capabilities. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information to augment information drawn from its own vast, static training data. This allows LLMs to use domain-specific and/or updated information.
Use cases include providing chatbot access to internal company data or giving factual information only from an authoritative source.
Process: The RAG process is made up of four key stages. First, all the data must be prepared and indexed for use by the LLM. Thereafter, each query consists of a retrieval, augmentation, and generation phase.
Indexing: Typically, the data to be referenced is converted into LLM embeddings, numerical representations in the form of large vectors. RAG can be used on unstructured (usually text), semi-structured, or structured data (for example knowledge graphs). These embeddings are then stored in a vector database to allow for document retrieval.
Overview of RAG process, combining external documents and user input into an LLM prompt to get tailored output
Retrieval: Given a user query, a document retriever is first called to select the most relevant documents that will be used to augment the query.This comparison can be done using a variety of methods, which depend in part on the type of indexing used.
Augmentation:The model feeds this relevant retrieved information into the LLM via prompt engineering of the user's original query.Newer implementations (as of 2023) can also incorporate specific augmentation modules with abilities such as expanding queries into multiple domains and using memory and self-improvement to learn from previous retrievals.
Generation:Finally, the LLM can generate output based on both the query and the retrieved documents.Some models incorporate extra steps to improve output, such as the re-ranking of retrieved information, context selection, and fine-tuning"""

# Function to clean and preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    return text.strip()

# Function to chunk the corpus
def chunk_corpus(corpus, chunk_size=200):
    sentences = sent_tokenize(corpus)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        current_chunk_size += len(sentence.split())
        if current_chunk_size <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = len(sentence.split())
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [preprocess_text(chunk) for chunk in chunks]

chunks = chunk_corpus(corpus)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed each chunk
embeddings = model.encode(chunks)

# Set up FAISS index
d = embeddings.shape[1]  # Dimensionality of the embeddings
index = faiss.IndexFlatL2(d)  # FAISS index for L2 distance
index.add(np.array(embeddings, dtype=np.float32))

def retrieve_relevant_chunks(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    return [chunks[i] for i in indices[0]]

# Load the model and tokenizer
model_gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("distilgpt2")

# Function to generate an answer based on retrieved chunks
def generate_answer(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer_gpt2.encode(input_text, return_tensors="pt")
    outputs = model_gpt2.generate(inputs, max_length=600, num_return_sequences=1)
    return tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)

# Initialize Flask app
app = Flask(__name__)

# MySQL setup
conn = mysql.connector.connect(host="localhost", user="root", password="ayushk192", database="chatbot")
cursor = conn.cursor()

# Create table for chat history (if it doesn't exist)
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME,
                    role VARCHAR(255),
                    content TEXT)''')

# /chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    retrieved_chunks = retrieve_relevant_chunks(user_query)
    answer = generate_answer(user_query, retrieved_chunks)
    
    # Save chat history
    timestamp = datetime.now()
    cursor.execute("INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)", 
                   (timestamp, 'user', user_query))
    cursor.execute("INSERT INTO chat_history (timestamp, role, content) VALUES (%s, %s, %s)", 
                   (timestamp, 'system', answer))
    conn.commit()

    return jsonify({"answer": answer, "retrieved_chunks": retrieved_chunks})

# /history endpoint
@app.route('/history', methods=['GET'])
def history():
    cursor.execute("SELECT * FROM chat_history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    return jsonify(rows)

if __name__ == '__main__':
    app.run(debug=True)