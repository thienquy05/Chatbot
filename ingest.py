import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # Load the OpenAI API key from .env 
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OpenAI API key not found. Please check .env file again")
    
    # Ensure the OpenAI API key is available via environment variable (the OpenAIEmbeddings implementation
    # reads the key from the environment rather than accepting an openai_api_key parameter)
    os.environ["OPENAI_API_KEY"] = openai_key

    # Load all text files in the /data folder
    print("Loading documents")
    loader = DirectoryLoader("data")
    documents = loader.load()
    print(len(documents))
    # Split text into smaller chunks for better embedding quality
    print("Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,   # Roughly 500 characters per chunk
        chunk_overlap = 50  # small overlap to keep context connected
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings for each chunk using OpenAI's text-embedding model
    print("Creating embeddings and saving to Chroma database")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Store embeddings locally using Chroma
    db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

    # Save the vector database to use it later in main.py
    db.persist()
    
    print("Ingestion complete! Your data has been embedded and stored locally")

if __name__ == "__main__":
    main()

