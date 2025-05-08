import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_vectorstore(text,hf_key):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        api_key=hf_key,
        api_url = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large-instruct"
        )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    final_docs = text_splitter.create_documents([text])
    print(embeddings)
    vectors = FAISS.from_documents(final_docs,embeddings)
    return vectors

def create_conversational_rag(vectorstore, groq_key):
    llm = ChatGroq(model="Llama3-70b-8192",groq_api_key= groq_key)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    retriever = vectorstore.as_retriever()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    return qa_chain

st.title("PDF-based Conversational RAG App with Memory")
st.write("Upload any PDF file and ask questions to the LLM")

st.sidebar.write("Enter API Keys")
hf_key = st.sidebar.text_input("Enter HuggingFace API Key", type = "password")
groq_key = st.sidebar.text_input("Enter Groq PI Key", type = "password")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file and hf_key and groq_key and st.session_state.qa_chain is None:
    with st.spinner("Reading and indexing PDF content..."):
        text = extract_text_from_pdf(uploaded_file)
        vectorstore = get_vectorstore(text,hf_key)
        st.session_state.qa_chain = create_conversational_rag(vectorstore,groq_key)
        st.success("Knowledge base ready from PDF!")

question = st.text_input("Ask a question about the PDF:")
if question and st.session_state.qa_chain:
    response = st.session_state.qa_chain.run(question)
    st.session_state.chat_history.append(("User", question))
    st.session_state.chat_history.append(("Assistant", response))
    st.write("Answer:", response)

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")