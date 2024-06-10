# pip install streamlit langchain chromadb langchain-community PyMuPDF
import re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile

embeddings = OllamaEmbeddings(model="nomic-embed-text")
st.set_page_config(layout="wide", page_title="🦜🔗 Information Extraction and Translation App")
st.title("🦜🔗 Information Report and Translation App")

llm = Ollama(model="llama3", verbose=False, temperature=0)
kolllm = Ollama(model="EEVE-Korean-10.8B", verbose=False, temperature=0)

@st.cache_resource
def read_file(file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file_name.getbuffer())
        file_path = tf.name
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(len(documents))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    return docs

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    return text.strip()

st.markdown("### 주의사항 :  LLM이 pdf 파일을 읽고 요약해드립니다")
st.markdown("#### PDF 업로드 ▼ ")
uploaded_file = st.file_uploader('pdfuploader', accept_multiple_files=False, type="pdf")

if uploaded_file is not None:
    # Clear previous documents from vectorstore
    if 'vectorstore' in st.session_state:
        coll = st.session_state.vectorstore.get()
        ids_to_del = coll['ids']
        st.session_state.vectorstore._collection.delete(ids_to_del)
        del st.session_state.vectorstore

    txt_input = read_file(uploaded_file)
    # Initialize vectorstore and insert documents
    st.session_state.vectorstore = Chroma.from_documents(documents=txt_input, embedding=embeddings)

st.info("Prompt: Organize documents neatly using tables and bullets. Please refrain from creating anything unrelated to the summary.")
st.info("Chunk size: 2000, Chunk overlap: 400")

## Main UI
if uploaded_file is not None:
    qachain = RetrievalQA.from_chain_type(llm, retriever=st.session_state.vectorstore.as_retriever())
    map_prompt_template = """Organize documents neatly using tables and bullets.\n
    Please refrain from creating anything unrelated to the summary.\n
    Please summarize the information you wrote in 100 characters or less.\n
    {text}
    Please summarize it nicely."""

    with st.form("query_form", clear_on_submit=True):
        query1, query2, query3, query4 = st.columns(4)
        queries = [query1.text_input("Query 1"), query2.text_input("Query 2"), query3.text_input("Query 3"), query4.text_input("Query 4")]
        submitted_query = st.form_submit_button("Query the Document")

    if submitted_query:
        for query in queries:
            if query:
                with st.spinner(f"Retrieving and summarizing information for '{query}'..."):
                    relevant_docs = st.session_state.vectorstore.similarity_search(query, k=3)
                    unique_docs = list(set(doc.page_content for doc in relevant_docs))
                    cleaned_text = "\n\n".join(clean_text(doc) for doc in unique_docs)
                    prompt_text = map_prompt_template.format(text=cleaned_text)
                    summary = llm.invoke(prompt_text)
                    st.session_state.summary_result = f"### {query}\n{summary}\n\n"
                    st.markdown(st.session_state.summary_result)

    ## Display ChromaDB Data for Inspection
    if st.button("Show ChromaDB Data"):
        with st.expander("ChromaDB Stored Documents"):
            docs = st.session_state.vectorstore.similarity_search("", k=len(txt_input))  # Retrieve all documents
            for doc in docs:
                st.write(doc.metadata)
                st.write(doc.page_content)
