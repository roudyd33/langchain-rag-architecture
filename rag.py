import os
from pathlib import Path
from pptx import Presentation

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "documents"
DB_DIR = PROJECT_DIR / "chroma_db_local"

# Find the PowerPoint
pptx_files = list(DOCS_DIR.glob("*.pptx"))
if not pptx_files:
    raise FileNotFoundError(
        "No PowerPoint (.pptx) file found in documents."
    )

pptx_path = pptx_files[0]
print(f"Reading: {pptx_path.name}")

def load_and_split_documents(pptx_path):
    presentation = Presentation(str(pptx_path))
    documents = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
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
                        "slide": slide_number,
                    },
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    return splitter.split_documents(documents)
# Create embeddings and connect to the local database
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="aviation",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)

if vector_store._collection.count() == 0:
    chunks = load_and_split_documents(pptx_path)
    print(f"Created {len(chunks)} text chunks.")

    if chunks:
        vector_store.add_documents(chunks)
        print("Aviation documents added to the database.")
    else:
        print("No text was found in the PowerPoint.")
else:
    print("Using the existing aviation database.")

print("RAG database ready!")
# Ask a question and retrieve relevant chunks
question = input("\nAsk a question about your presentation: ")

results = vector_store.similarity_search(question, k=4)

if not results:
    print("I couldn't find relevant information in the presentation.")
else:
    context = "\n\n".join(
        f"Slide {doc.metadata.get('slide', '?')}:\n{doc.page_content}"
        for doc in results
    )

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    prompt = f"""
Answer using only the presentation excerpts below.
If the excerpts do not contain the answer, say you cannot find it.
Mention the slide number when possible.

Presentation excerpts:
{context}

Question: {question}
"""

    response = llm.invoke(prompt)
    print("\nAnswer:")

    if isinstance(response.content, str):
        print(response.content)
    elif isinstance(response.content, list):
        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                print(item["text"])