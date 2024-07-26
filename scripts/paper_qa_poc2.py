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

# Main function to test the functionality
def main():
    st.title("Document QA with Local LLaMA2 Server")
    
    document_directory = './input_files'
    question = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        answer = process_input(document_directory, question)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()