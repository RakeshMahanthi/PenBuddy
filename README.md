# PenBuddy
This project aims to build/utilize an LLM model to assist pentesters in writing exploits. 

### Installation

1. Download ollama tool from their website. 

2. Pull the llama2-uncensored model. After the model downloads, you will enter into a command prompt where you can interact with the model. Exit from it. 
`ollama run llama2-uncensored`

3. The ollama server runs on port 11434. It exposes various API endpoints with which we can interact with the model. 

4. In scripts/ directory, you will find the ollama-client.py which when run interacts with the llama2-uncensored model and gives the result. Please update the prompt in the code according to your wish. 


