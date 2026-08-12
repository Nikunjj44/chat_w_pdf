import streamlit as st
import uuid
from dotenv import load_dotenv
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
    This function is responsible for extracting each page's text of the pdf and then concatenating them.
    It returns 1 string that contains all text from all of the uploaded pdfs
    
    :param data: list of pdf files uploaded through the inpuy
    """

    text = ""
    num_pages = 0
    count = 0
    for pdf_file in data:
        count = count+1
        reader_obj = PdfReader(pdf_file)
        pages = reader_obj.pages
        for page in pages:
            text = text + page.extract_text()
        num_pages = num_pages + len(pages)
    
    print(f"Total number of pages extracted = {num_pages}")
    return text

def get_text_chunks(text):
    """
    This function makes use of langchain and creates chunks of text
    
    :param text: text from all of the pdfs
    """
    # we can use CharacterTextSplitter also, but we do not
    # because recursive uses multiple splitters whereas CharacterTextSplitter uses only one

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200                         
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text):
    """
    This creates embeddings of the chunks of text passed and then stores them in a vector store
    
    :param text: chunks of text
    """
    # we can use OpenAIEmbeddings - but their benchmarking is not the best and it is paid
    # the vector store FAISS is local, for cloud we can use Pinecone
    # this would be slow as it is doing this on the local CPU, but it requires a GPU

    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )
    vector_store = FAISS.from_texts(
        texts = text,
        embedding = embeddings
    )

    return vector_store

def create_conversation_chain(vector_store):


    ## creating memory
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        temperature = 0
    )


    retriever = vector_store.as_retriever()


    ### re-writing the questions using history
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

    #### Prompt for Answering

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


    ### Create document QA chain

    ques_ans_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    ### Create RAG chain

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        ques_ans_chain
    )

    ### Adding session specific memory

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer"
    )

    return conversation_chain

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


def main():
    # loading env variables
    load_dotenv()

    # initializing streamlit variables
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # setting up page's url
    st.set_page_config(page_title="Talk to PDFs", page_icon=":smiley:", layout="centered")

    st.header("Talk to your files :smiley:")
    user_input = st.chat_input("Start a conversation with your PDFs here...")

    ### Showcasing all of the messages

    # if user_input:
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])

    with st.sidebar:
        st.subheader("Upload your PDFs below")
        uploaded_pdfs = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True)
        if st.button("Save"):
            # adding spinner for processing
            with st.spinner("Saving... This might take some time :eight-thirty:"):
                pdf_text = get_pdf_data(uploaded_pdfs)

                # Now we split all of the text and break it down into chunks in order to feed it into the model
                chunks = get_text_chunks(pdf_text)

                # create the vector store
                st.session_state.vector_store = get_vector_store(chunks)

                # creates conversation chain
                st.session_state.conversation = create_conversation_chain(st.session_state.vector_store)

                st.success("PDF saved successfully! :check_mark_button:")

    if user_input:
        if "conversation" not in st.session_state:
            st.warning("Please upload and save PDF first.")
        else:

            st.session_state.messages.append(
                            {"role": "user", "content": user_input}
                        )

            response = st.session_state.conversation.invoke(
                {"input" : user_input},
                config = {
                            "configurable": {
                                "session_id": st.session_state.session_id
                            }
                        }
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

if __name__ == "__main__":
    main()