import requests
import json

# Define the URL of the LLaMA server's /generate endpoint
LLAMA_SERVER_URL = 'http://localhost:11434/api/generate'  # Replace with your server URL

def interact_with_llama_server(text):
    # Define the input data as a dictionary
    input_data = {
        'model': 'llama2-uncensored',
        'prompt':text
    }

    try:
        # Send POST request to the LLaMA server's /generate endpoint
        response = requests.post(LLAMA_SERVER_URL, json=input_data, stream=True)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Initialize an empty result string to accumulate response values
            final_result = ''
            result = ''
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    # Append the current chunk to the result string
                    result += chunk.decode('utf-8')
                    
                    # Attempt to parse JSON from the accumulated result
                    try:
                        data = json.loads(result)
                        response_value = data.get('response')
                        if response_value:
                            #print(f'LLaMA Server Response (Partial):\n{response_value}')
                            # Clear result after successfully parsing a response
                            final_result += response_value
                            result = ''
                    
                    except json.JSONDecodeError:
                        # Continue accumulating chunks until a complete JSON object is received
                        continue
            
            print(final_result)
            
        else:
            print(f'Error: Failed to interact with LLaMA server. Status code: {response.status_code}')
            print(response.text)  # Print response text for debugging
    
    except requests.exceptions.RequestException as e:
        print(f'Error: Failed to connect to LLaMA server: {e}')

# Example usage
if __name__ == '__main__':
    input_text = "This is a test input for the LLaMA server."
    interact_with_llama_server(input_text)
