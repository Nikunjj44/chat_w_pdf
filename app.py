import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts = text,
        embedding = embeddings
    )

    return vector_store

def main():
    # loading env variables
    load_dotenv()

    # setting up page's url
    st.set_page_config(page_title="Talk to PDFs", page_icon=":smiley:", layout="centered")

    st.header("Talk to your files :smiley:")
    st.text_input("Start a conversation with your PDFs here...")

    with st.sidebar:
        st.subheader("Upload your PDFs below")
        uploaded_pdfs = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True)
        if st.button("Save"):
            # adding spinner for processing
            with st.spinner("Saving... This might take some time :clock:"):
                pdf_text = get_pdf_data(uploaded_pdfs)

                # Now we split all of the text and break it down into chunks in order to feed it into the model
                chunks = get_text_chunks(pdf_text)

                st.write(chunks)

                # create the vector store
                vector_store = get_vector_store(chunks)

                st.write(vector_store)


if __name__ == "__main__":
    main()