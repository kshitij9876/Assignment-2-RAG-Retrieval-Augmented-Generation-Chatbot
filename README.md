# RAG-Retrieval-Augmented-Generation-Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that combines semantic search with a generative language model. It uses FAISS for vector search, sentence-transformers for embeddings, and distilgpt2 for generating responses. Chat history is stored in a MySQL database and the system is served via a Flask API.

### Installation and Setup

1. Clone the Repository
  ```bash
  git clone <URL>
  cd rag-chatbot
  ```

2. Install Dependencies

  Ensure you have Python 3.8+ installed. Then, install the required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

3. Set Up MySQL
   
  **Install MySQL:**
  
  Download and install MySQL from MySQL Downloads.
  
  **Create a Database:**
  
  Log in to MySQL and create a database:
   ```bash
   CREATE DATABASE chatbot;
   ```

  **Create the Table:**

  Run the following SQL query to create the chat_history table:
   ```bash
  CREATE TABLE chat_history (
      id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp DATETIME,
      role VARCHAR(255),
      content TEXT
  );
  ```

  **Set Up Credentials:**
  
    Update the database credentials in the Flask app code. Locate the mysql.connector.connect call and replace:
     ```bash
    conn = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="chatbot"
    )
    ```

5. Run the Flask Server

  Start the Flask app:
  ```bash
  python app.py
  ```

6. Test the Endpoints

   Run the test.py to test the endpoints using Flask API. Write your desired query.
   
