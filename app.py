import streamlit as st
from pathlib import Path
from pptx import Presentation

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AeroRAG | Aviation Intelligence",
    page_icon="✈️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "documents"
DB_DIR = PROJECT_DIR / "chroma_db_local"

COLLECTION_NAME = "aviation"


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def load_and_split_documents(pptx_path):

    presentation = Presentation(str(pptx_path))
    documents = []

    for slide_number, slide in enumerate(
        presentation.slides, start=1
    ):

        slide_text = "\n".join(
            shape.text
            for shape in slide.shapes
            if shape.has_text_frame and shape.text.strip()
        )

        if slide_text.strip():

            documents.append(
                Document(
                    page_content=slide_text,
                    metadata={
                        "source": pptx_path.name,
                        "slide": slide_number
                    }
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# VECTOR DATABASE INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def initialize_rag():

    pptx_files = list(DOCS_DIR.glob("*.pptx"))

    if not pptx_files:
        raise FileNotFoundError(
            "No PowerPoint presentation found in documents."
        )

    pptx_path = pptx_files[0]

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    if vector_store._collection.count() == 0:

        chunks = load_and_split_documents(pptx_path)

        if not chunks:
            raise ValueError(
                "No readable text was found in the presentation."
            )

        vector_store.add_documents(chunks)

    return vector_store, pptx_path.name


# --------------------------------------------------
# PAGE INTERFACE
# --------------------------------------------------

st.title("✈️ AeroRAG")
st.subheader("Aviation Knowledge Assistant")

st.write(
    "An AI-powered Retrieval-Augmented Generation "
    "system using LangChain, ChromaDB and Ollama."
)

st.divider()

try:

    with st.spinner("Initializing RAG architecture..."):

        vector_store, document_name = initialize_rag()

    st.success("RAG database initialized successfully.")

    st.caption(f"Knowledge source: {document_name}")

except Exception as error:

    st.error("Unable to initialize the RAG system.")

    st.code(str(error))

    st.info(
        "Make sure Ollama is running and both required "
        "models are installed."
    )

    st.stop()


# --------------------------------------------------
# QUESTION AND ANSWER
# --------------------------------------------------

question = st.text_input(
    "Ask a question about aviation:",
    placeholder="What does a V-n diagram show?"
)

if st.button("Ask AeroRAG", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Retrieving information and generating answer..."
            ):

                results = vector_store.similarity_search(
                    question,
                    k=4
                )

                if not results:

                    st.warning(
                        "No relevant information was found."
                    )

                else:

                    context = "\n\n".join(
                        f"Slide {doc.metadata.get('slide', '?')}:\n"
                        f"{doc.page_content}"
                        for doc in results
                    )

                    llm = ChatOllama(
                        model="qwen2.5:3b",
                        temperature=0
                    )

                    prompt = f"""
You are an aviation knowledge assistant.

Answer using only the presentation excerpts below.
If the information is not available, clearly say so.
Mention the relevant slide number whenever possible.

Presentation excerpts:

{context}

Question: {question}

Answer:
"""

                    response = llm.invoke(prompt)

            st.subheader("Answer")

            st.write(response.content)

            with st.expander("View retrieved presentation excerpts"):

                for index, doc in enumerate(results, start=1):

                    st.markdown(
                        f"**Retrieved document {index} — "
                        f"Slide {doc.metadata.get('slide', '?')}**"
                    )

                    st.write(doc.page_content)

                    st.divider()

        except Exception as error:

            st.error("An error occurred while generating the answer.")

            st.code(str(error))


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with st.sidebar:

    st.title("About AeroRAG")

    st.write(
        "A Retrieval-Augmented Generation architecture "
        "for aviation knowledge retrieval."
    )

    st.markdown("### Technologies")

    st.write(
        "- Python\n"
        "- Streamlit\n"
        "- LangChain\n"
        "- ChromaDB\n"
        "- Ollama\n"
        "- nomic-embed-text\n"
        "- qwen2.5:3b"
    )
