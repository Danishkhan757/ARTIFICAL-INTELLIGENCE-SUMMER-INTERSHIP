# -*- coding: utf-8 -*-
# --- SQLite Override for ChromaDB on Streamlit Cloud ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import streamlit as st
import os
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- Page Configuration ---
st.set_page_config(
    page_title="Samsung Manual Assistant",
    page_icon="💧",
    layout="centered"
)

st.title("💧 Samsung Washing Machine Assistant")
st.markdown("Ask any questions about your Samsung washing machine's modes, warnings, and recommended actions.")

# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your secret key starting with 'sk-'")
    st.markdown("---")
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload the Samsung HTML Manual", type=["html"])
    st.markdown("*(If no file is uploaded, the app will look for a default `model.html` in your directory.)*")

# --- RAG Pipeline (Cached for Performance) ---
@st.cache_resource(show_spinner=False)
def setup_rag_pipeline(api_key, file_path):
    # Set the API key
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize models
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load and process document using BSHTMLLoader
    loader = BSHTMLLoader(file_path=file_path)
    machine_docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(machine_docs)

    # Create vector database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompt
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:""")

    # Build the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if user_input := st.chat_input("E.g., What should I do for a daily wash?"):

    # 1. Validate API Key
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # 2. Determine File Path
    file_path = "model.html"
    if uploaded_file is not None:
        # Save uploaded file to a temporary location for LangChain to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
    elif not os.path.exists(file_path):
        st.error("❌ Please upload the HTML manual in the sidebar or ensure a file named `model.html` is in your project folder.")
        st.stop()

    # 3. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 4. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            try:
                # Load pipeline and get response
                rag_chain = setup_rag_pipeline(api_key, file_path)
                response = rag_chain.invoke(user_input).content

                # Display and save response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
