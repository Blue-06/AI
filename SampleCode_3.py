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

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="RAG PDF Assistant", layout="wide")
st.title("RAG PDF Summarizer + Q&A")

os.environ["TIKTOKEN_CACHE_DIR"] = "token"

client = httpx.Client(verify=False)

# ---------------------------
# LLM (your GenAI setup)
# ---------------------------
llm = ChatOpenAI(
    openai_api_base="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-eSZtXTNitDmAGrD7Ct4liQ",
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    openai_api_base="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-eSZtXTNitDmAGrD7Ct4liQ",
    http_client=client
)

# ---------------------------
# SESSION STATE
# ---------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# ---------------------------
# FILE UPLOAD
# ---------------------------
upload_file = st.file_uploader("Upload PDF", type="pdf")

if upload_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(upload_file.read())
        pdf_path = tmp.name

    try:
        # ---------------------------
        # 1. Extract text
        # ---------------------------
        st.info("Extracting PDF text...")
        raw_text = extract_text(pdf_path)

        # ---------------------------
        # 2. Chunking
        # ---------------------------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(raw_text)

        # ---------------------------
        # 3. Vector DB
        # ---------------------------
        st.info("Indexing document...")
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_index"
        )

        st.session_state.vectordb = vectordb

        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        # ---------------------------
        # 4. SUMMARY (run once)
        # ---------------------------
        if st.session_state.summary is None:

            summary_prompt = ChatPromptTemplate.from_template("""
            Summarize the following document clearly and concisely:

            {context}
            """)

            summary_chain = create_stuff_documents_chain(llm, summary_prompt)
            summary_rag = create_retrieval_chain(retriever, summary_chain)

            with st.spinner("Generating summary..."):
                result = summary_rag.invoke({"input": "Summarize this document"})
                st.session_state.summary = result["answer"]

        # Show summary
        st.subheader("Document Summary")
        st.write(st.session_state.summary)

        st.success("PDF ready for Q&A below")

    finally:
        os.remove(pdf_path)

# ---------------------------
# Q&A SECTION
# ---------------------------
if st.session_state.vectordb:

    st.divider()
    st.subheader("💬 Ask Questions from PDF")

    query = st.text_input("Ask anything about the PDF:")

    if query:

        retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})

        qa_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions based only on the document.

        Context:
        {context}

        Question:
        {input}

        Answer clearly and concisely:
        """)

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})

        st.subheader("Answer")
        st.write(response["answer"])

else:
    st.info("Upload a PDF to enable Q&A.")