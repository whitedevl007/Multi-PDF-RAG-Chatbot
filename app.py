import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Initialize Hugging Face InferenceClient
huggingface_client = InferenceClient(
    model="bigscience/bloomz-560m",  # Replace with the desired Hugging Face model
    token=os.getenv("HUGGINGFACE_API_KEY"),
)


# Function to query Hugging Face models
def query_huggingface_model(prompt):
    try:
        response = huggingface_client.text_generation(prompt, max_new_tokens=200)
        return response.get("generated_text", "Sorry, no response generated.")
    except Exception as e:
        return f"Error querying Hugging Face model: {str(e)}"


# Function to read PDF content
def pdf_read(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into manageable chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


# Function to create and save FAISS vector store
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


# Function to handle user input and retrieve answers
def user_input(user_question):
    if not os.path.exists("faiss_db"):
        raise FileNotFoundError(
            "FAISS vector store is missing. Please upload and process PDFs first."
        )

    # Load existing FAISS index
    new_db = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    retriever = new_db.as_retriever()
    retrieval_tool = create_retriever_tool(
        retriever, "pdf_extractor", "Tool to extract answers from the PDF content."
    )

    # Combine retriever result with Hugging Face model
    context = retriever.get_relevant_documents(user_question)
    combined_context = "\n".join([doc.page_content for doc in context])
    prompt = f"Context: {combined_context}\n\nQuestion: {user_question}\n\nAnswer:"
    return query_huggingface_model(prompt)


# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using RAG (Hugging Face)")

    # User question input
    user_question = st.text_input("Ask a question about the uploaded PDF(s):")

    if user_question:
        with st.spinner("Retrieving answer..."):
            try:
                response = user_input(user_question)
                st.success("Response: " + response)
            except FileNotFoundError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

    # Sidebar for file upload
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", accept_multiple_files=True, type=["pdf"]
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = pdf_read(pdf_docs)
                        text_chunks = get_chunks(raw_text)
                        vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
