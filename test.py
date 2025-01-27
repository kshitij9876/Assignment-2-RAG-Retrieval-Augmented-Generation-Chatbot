import requests

# Define the URL for the chat endpoint and history endpoint
chat_url = "http://127.0.0.1:5000/chat"
history_url = "http://127.0.0.1:5000/history"

# Define the payload with the user query
payload = {
    "query": "What is RAG ?"
}

# Send the POST request to the Flask API
chat_response = requests.post(chat_url, json=payload)

# Check if the response is successful
if chat_response.status_code == 200:
    print("Response from /chat:")
    print(chat_response.json())  # Print the response (answer and retrieved chunks)
else:
    print(f"Failed to get a response. Status code: {chat_response.status_code}")

# Send the GET request to the Flask API for chat history
history_response = requests.get(history_url)

# Check if the response is successful
if history_response.status_code == 200:
    print("\nChat history:")
    print(history_response.json())  # Print the chat history
else:
    print(f"Failed to get chat history. Status code: {history_response.status_code}")