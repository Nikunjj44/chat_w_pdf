import streamlit as st
from dotenv import load_dotenv


def main():
    st.set_page_config(page_title="Talk to PDFs", page_icon=":smiley:", layout="centered")

    st.header("Talk to your files :smiley:")
    st.text_input("Start a conversation with your PDFs here...")

    with st.sidebar:
        st.subheader("Upload your PDFs below")
        pdfs = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True)
        st.button("Save")


if __name__ == "__main__":
    main()