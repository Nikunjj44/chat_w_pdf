# Talk to PDFs - Conversational RAG Chatbot
This project implements a conversational chatbot that lets users upload one or more PDF documents and have a context-aware, multi-turn conversation with them. Using a Retrieval-Augmented Generation (RAG) pipeline, the system fetches the most relevant sections of the uploaded PDFs and generates grounded answers, while also remembering the conversation history to resolve follow-up questions naturally.

## TechStack Used

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-4B8BBE?style=for-the-badge&logo=text&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)
![PyPDF2](https://img.shields.io/badge/PyPDF2-CB3837?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## Methodology
1. Model Used -- OpenAI's `gpt-4o-mini` for generation and `text-embedding-3-small` for embeddings.
2. Data pre-processing:
     a) Reading each uploaded PDF using **PyPDF2** and extracting text page-by-page.
     b) Concatenating text from all uploaded PDFs into a single corpus so multiple documents can be queried together.
     c) Splitting the corpus into overlapping chunks using **RecursiveCharacterTextSplitter** with chunk size = 1000 and overlap = 200.
     d) Recursive splitting is preferred over standard character splitting as it tries a priority list of separators (paragraphs → line breaks → spaces → characters), preserving the natural structure of the text.
     e) The 200-character overlap ensures that ideas split across two chunks still retain enough context in both chunks for retrieval to work reliably.
3. Creating vector representations of text chunks using **OpenAI Embeddings**:
     a) Each chunk is converted into a 1536-dimensional dense vector using the `text-embedding-3-small` model.
     b) These vectors are stored locally in a **FAISS** vector store, which enables fast approximate nearest neighbor (ANN) search.
     c) FAISS is used as it runs locally with zero setup; for cloud deployments, this can be swapped with Pinecone, Weaviate, or Chroma.
4. Building the RAG chain using **History-Aware Retriever and Stuff Documents Chain**:
     a) **History-Aware Retriever:** Before retrieval, an LLM call rewrites the user's latest question into a standalone question using the chat history. For example, "What are its limitations?" gets rewritten to "What are the limitations of transformer architecture?" so the retriever knows what to search for.
     b) **Stuff Documents Chain:** The top-k retrieved chunks are "stuffed" into a single prompt as context, along with the original user question and chat history. This prompt is sent to the LLM to generate the final answer.
     c) The system prompt includes a grounding instruction ("if the answer is not in the context, say you don't know") to reduce hallucinations, and formatting rules to preserve LaTeX notation from mathematical PDFs.
     d) Both chains are combined using `create_retrieval_chain`, which wires them into a single runnable that takes a question and returns a grounded answer.
5. Adding conversational memory using **RunnableWithMessageHistory**:
     a) Each browser session is assigned a unique UUID, ensuring multiple concurrent users do not share chat history.
     b) An `InMemoryChatMessageHistory` object stores the messages for each session and is automatically injected into the RAG chain on every invocation.
6. Building Streamlit App

## Streamlit Application

**Landing Page**
<img width="1374" height="899" alt="image" src="https://github.com/user-attachments/assets/02db8490-2eb4-44e4-8dfd-b2d0962150c8" />

**Adding inputs in the sidebar (OpenAI API key and PDFs)**

<img width="290" height="609" alt="image" src="https://github.com/user-attachments/assets/40f5ed82-3c5e-4efa-8b3e-40179a693ab0" />

**Sample Chat Output**

<img width="741" height="823" alt="image" src="https://github.com/user-attachments/assets/07da0050-e470-4ca5-b401-3c4d7b1a74ba" />
<img width="724" height="758" alt="image" src="https://github.com/user-attachments/assets/c617696f-1163-4d36-826d-45ef9846b12c" />

**Examples of some captured edge cases**

The user can see their whole chat history in a scrollable container.
The answer prompts also formats the answers properly and bolds the topics as seen in the showcasing of the contents of one of the files.

<img width="724" height="696" alt="image" src="https://github.com/user-attachments/assets/aaacd28d-209f-4c18-991d-6df4fe0a4c67" />

Mathematical formulas are also displayed accurately.

<img width="724" height="341" alt="image" src="https://github.com/user-attachments/assets/0b297ba7-2e4a-4915-a975-45daebddac1c" />
<img width="726" height="747" alt="image" src="https://github.com/user-attachments/assets/cf52031b-9ce3-4395-8c0c-246f721c760c" />

**Application Link -- Try it out 😄**

https://chat-with-pdf-files.streamlit.app

## Future Scope

Currently the application uses an in-memory FAISS index, meaning the vector store is lost every time the app restarts. This can be improved by migrating to a persistent vector database like Pinecone, Weaviate, or Chroma, which would allow users to upload documents once and query them across sessions. Another improvement would be extending ingestion support beyond PDFs to include DOCX, TXT, HTML, and Markdown files, making the tool useful for a wider variety of document types.

Furthermore, we can have some additional enhancements while displaying the final answers like:
1. Including **source citations** with page numbers and snippets from the PDF, so users can verify where each part of the answer came from.
2. Adding **token streaming** so answers appear word-by-word instead of all at once, matching the responsiveness of modern chatbots.
3. Incorporating **evaluation frameworks** like RAGAS or TruLens to measure answer relevancy and retrieval quality over time.
4. Supporting **open-source embeddings** (e.g., `bge-large-en` via HuggingFace) as an alternative to OpenAI, giving users a fully local, cost-free option.
