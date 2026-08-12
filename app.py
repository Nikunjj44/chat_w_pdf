import os
import streamlit as st
import uuid
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def get_pdf_data(data):
    """
    Extracts each page's text from the uploaded PDFs and concatenates them.
    Returns a single string containing all text from all uploaded PDFs.

    :param data: list of pdf files uploaded through the input
    """
    text = ""
    num_pages = 0
    count = 0
    for pdf_file in data:
        count = count + 1
        reader_obj = PdfReader(pdf_file)
        pages = reader_obj.pages
        for page in pages:
            text = text + page.extract_text()
        num_pages = num_pages + len(pages)

    print(f"Total number of pages extracted = {num_pages}")
    return text


def get_text_chunks(text):
    """
    Uses langchain to create chunks of text.

    :param text: text from all of the pdfs
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text, api_key):
    """
    Creates embeddings of the chunks and stores them in a FAISS vector store.

    :param text: chunks of text
    :param api_key: OpenAI API key (from user input or env)
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    vector_store = FAISS.from_texts(
        texts=text,
        embedding=embeddings
    )
    return vector_store


def create_conversation_chain(vector_store, api_key):
    """
    Builds the full conversational RAG chain using the provided API key.

    :param vector_store: FAISS vector store built from PDF chunks
    :param api_key: OpenAI API key (from user input or env)
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    retriever = vector_store.as_retriever()

    # Re-writing questions using history
    contextualize_ques_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Given the chat history and the latest user question, "
                "rewrite the question so that it is a standalone question. "
                "Do not answer the question."""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_ques_prompt
    )

    # Prompt for answering
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant that answers questions about
            the uploaded PDF documents.

            Use the following retrieved context to answer the question.

            <context>
            {context}
            </context>

            If the answer cannot be found in the provided context,
            say that you don't know.

            Keep your answer clear and concise.

            FORMATTING RULES:
            1. Use Markdown for normal text.
            2. Use LaTeX for all mathematical expressions.
            3. For inline mathematics, use \\( ... \\).
            4. For standalone equations, use:
            $$
            equation
            $$
            5. NEVER use square brackets [ ... ] to represent mathematical equations.
            6. NEVER output raw LaTeX commands such as \\mathbf, \\frac, \\sum,
            \\alpha, etc. outside a math delimiter.
            7. Preserve the mathematical notation from the PDF accurately.
            """
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    ques_ans_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        ques_ans_chain
    )

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversation_chain


store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def resolve_api_key():
    """
    Resolves the OpenAI API key from (in order):
      1. User input in the sidebar (session state)
      2. .env file (local dev)
      3. Streamlit secrets (cloud deploy)
    Returns None if no key is found.
    """
    # 1. User-provided key from sidebar
    if st.session_state.get("user_api_key"):
        return st.session_state.user_api_key

    # 2. .env file (local dev)
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 3. Streamlit secrets (deployed app fallback — optional)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def main():
    # Load env variables (for local dev)
    load_dotenv()

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_api_key" not in st.session_state:
        st.session_state.user_api_key = ""

    # Page setup
    st.set_page_config(page_title="Talk to PDFs", page_icon=":smiley:", layout="centered")

    st.header("Talk to your files :smiley:")
    user_input = st.chat_input("Start a conversation with your PDFs here...")

    with st.sidebar:
        # ============ API KEY SECTION ============
        st.subheader("🔑 OpenAI API Key")

        st.session_state.user_api_key = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            value=st.session_state.user_api_key,
            help="Your key is used only for this session and is never stored on the server.",
            placeholder="sk-..."
        )

        with st.expander("ℹ️  Why do I need this?"):
            st.markdown(
                """
                This is a **demo app** — you bring your own OpenAI API key so
                you can try it out without me paying for your usage 😄

                **How to get one:**
                1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
                2. Click **Create new secret key**
                3. Copy it here (starts with `sk-...`)

                **Cost:** Chatting with a typical PDF costs less than **$0.01**
                using `gpt-4o-mini` + `text-embedding-3-small`.

                🔒 Your key is kept in your browser session only. It is not
                logged, stored, or sent anywhere except OpenAI.
                """
            )

        st.divider()

        # ============ PDF UPLOAD SECTION ============
        st.subheader("📄 Upload your PDFs")
        uploaded_pdfs = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Save", type="primary"):
            api_key = resolve_api_key()

            # Guardrails
            if not api_key:
                st.error("⚠️  Please enter your OpenAI API key first.")
            elif not uploaded_pdfs:
                st.error("⚠️  Please upload at least one PDF.")
            else:
                with st.spinner("Saving... This might take some time :eight-thirty:"):
                    try:
                        pdf_text = get_pdf_data(uploaded_pdfs)
                        chunks = get_text_chunks(pdf_text)
                        st.session_state.vector_store = get_vector_store(chunks, api_key)
                        st.session_state.conversation = create_conversation_chain(
                            st.session_state.vector_store, api_key
                        )
                        st.success("PDF saved successfully! :check_mark_button:")
                    except Exception as e:
                        st.error(f"❌ Something went wrong: {e}")
                        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                            st.info("💡 Double-check that your OpenAI API key is valid.")

    # ============ CHAT HANDLING ============
    if user_input:
        if not resolve_api_key():
            st.warning("⚠️  Please enter your OpenAI API key in the sidebar first.")
        elif "conversation" not in st.session_state:
            st.warning("⚠️  Please upload and save a PDF first.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            try:
                response = st.session_state.conversation.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    }
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": response["answer"]}
                )
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])


if __name__ == "__main__":
    main()