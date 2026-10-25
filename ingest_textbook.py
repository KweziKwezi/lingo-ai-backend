import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from database import add_book
from pptx import Presentation

load_dotenv()

def extract_text_from_pptx(pptx_path: str) -> str:
    """Extract all text from a PowerPoint file."""
    prs = Presentation(pptx_path)
    full_text = []
    
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                full_text.append(shape.text)
    
    return "\n".join(full_text)

def ingest_textbook(file_path: str, title: str, grade: str, subject: str, 
                    user_id: int, type: str = "textbook") -> int:
    """
    Ingest a textbook or slide deck into the vector database.
    Supports PDF and PPTX files.
    """
    print(f"📖 Loading: {title}...")
    
    # 1. Extract text based on file type
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
    elif file_path.endswith('.pptx'):
        full_text = extract_text_from_pptx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and PPTX are supported.")
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\nChapter ", "\nCHAPTER ", "\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(full_text)
    print(f"   Split into {len(chunks)} chunks")
    
    # 3. Add book to database FIRST to get the book_id
    filename = os.path.basename(file_path)
    book_id = add_book(
        title=title,
        grade=grade,
        subject=subject,
        filename=filename,
        file_path=file_path,
        total_chunks=len(chunks),
        user_id=user_id,
        type=type
    )
    print(f"   ✅ Registered with ID: {book_id}")
    
    # 4. Prepare documents with metadata
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        section_num = (i // 5) + 1
        section_label = "Slide" if type == "slides" else "Chapter"
        
        documents.append(chunk)
        metadatas.append({
            "book_id": str(book_id), # CRITICAL: This is what we filter by in main.py!
            "book_title": title,
            "grade": grade,
            "subject": subject,
            "chapter": f"{section_label} {section_num}",
            "type": type,
            "user_id": str(user_id),
            "chunk_index": i
        })
    
    # 5. Add to ChromaDB
    print("   Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db = Chroma(
        persist_directory="./sa_textbook_db", # GLOBAL FOLDER
        embedding_function=embeddings
    )
    
    db.add_texts(
        texts=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully ingested: {title} (ID: {book_id}) into ./sa_textbook_db\n")
    return book_id