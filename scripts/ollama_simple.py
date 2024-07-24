import os
import pdfplumber  # For extracting text from PDF files
import requests
from typing import List

# Load and process documents from a directory
def load_documents(directory: str) -> List[str]:
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                documents.append(text)
        elif filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Query the model using HTTP API
def query_model_with_context(prompt: str, context: List[str]) -> str:
    combined_context = "\n\n".join(context)
    full_prompt = f"Context:\n{combined_context}\n\nQuestion: {prompt}"
    
    url = 'http://localhost:11434/api/generate'  # Replace with your actual endpoint
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'llama2-uncensored',
        'prompt': full_prompt,
        'max_new_tokens': 256,
        'temperature': 0.01,
        'stream':False 
    }
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('response','')
    else:
        return f"Error: {response.status_code}, {response.response}"

# Example usage
def main():
    # Directory containing your documents
    input_directory = '../input_files'

    # Load documents
    documents = load_documents(input_directory)

    # Combine documents into a single context
    combined_context = "\n\n".join(documents)

    # Query example
    prompt = "List the attribute categories present in the NCISS attribute definitions"
    answer = query_model_with_context(prompt, [combined_context])
    
    print(answer)

if __name__ == "__main__":
    main()
