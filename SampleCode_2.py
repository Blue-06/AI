import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os
import httpx

#-----tiktoken setup----#
tiktoken_cache_dir = "token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# ---- HTTP CLIENT ----
client = httpx.Client(verify=False)

# ---- LLM SETUP ----
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-somethinf",
    http_client=client
)

# ---- EMBEDDINGS SETUP ----

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-eSZtXTNitDmAGrD7Ct4liQ",
    http_client=client
)

# ---- STREAMLIT UI ----
st.set_page_config(page_title="RAG PDF Summarizer")
st.title("RAG-powered PDF Summarizer")

upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:

    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    try:
        # Step 1: Extract text
        st.info("Extracting text from PDF...")
        raw_text = extract_text(temp_file_path)

        # Step 2: Chunking
        st.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(raw_text)

        # Step 3: Vector DB
        with st.spinner("Indexing document..."):
            vectordb = Chroma.from_texts(
                texts=chunks,
                embedding=embedding_model,
                persist_directory="./chroma_index"
            )

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Step 4: MODERN RAG CHAIN
        prompt = ChatPromptTemplate.from_template("""
        Summarize the document based on the context below:

        {context}
        """)

        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        # Step 5: Run
        summary_prompt = "Summarize this document"

        with st.spinner("Generating summary..."):
            result = rag_chain.invoke({"input": summary_prompt})

        # Output
        st.subheader("Summary")
        st.write(result["answer"])

    finally:
        os.remove(temp_file_path)
