import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from paperqa import Doc
import os
from langchain_core.documents import Document

document_directory = '../input_files'

# Load documents from the input_files folder
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if filename.endswith('.txt'):
                with open(file_path, 'r') as file:
                    documents.append(file.read())
            elif filename.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    text = ''
                    for page_num in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page_num].extract_text()
                    documents.append(Document(page_content=text,metadata={"source": filename}))
    return documents

# URL processing
def process_input(doucment_directory, question):
    model_local = Ollama(model="llama3")
    
    #Load the documents in the input directory
    documents = load_documents_from_folder('../input_files')
    
    #Initialize text splitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)

    #Split the documents
    doc_splits = text_splitter.split_documents(documents)
    
    # Create a metadata for each chunk
    #metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    #Convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.OllamaEmbeddings(model='llama3'),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    #Perform the RAG
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Streamlit app layout
st.set_page_config(page_title="Document QA with Local LLaMA3 Server", layout="wide")

st.title("Document QA with Local LLaMA3 Server")

st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #dcdcdc;
    }
    .main {
        background-color: #2e2e2e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .title {
        color: #ffffff;
        font-size: 2em;
        margin-bottom: 20px;
    }
    .text-input {
        margin-bottom: 20px;
    }
    .stTextInput>div>input {
        background-color: #333;
        color: #ddd;
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .answer {
        font-size: 1.2em;
        color: #f1f1f1;
        border: 1px solid #444;
        padding: 10px;
        border-radius: 5px;
        background-color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Container for the main content
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Question input with a unique key
    question = st.text_input("Enter your question:", key="question_input", placeholder="Type your question here...", help="Ask a question about the documents.")
    
    # Submit button with a unique key
    if st.button("Submit", key="submit_button", help="Submit your question"):
        if question:
            answer = process_input(document_directory, question)
            st.markdown(f'<div class="answer">Answer: {answer}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a question.")
    
    st.markdown('</div>', unsafe_allow_html=True)