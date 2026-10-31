import os
import gc
import sqlite3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from database import add_book
from pptx import Presentation

load_dotenv()

# Configure persistent storage paths
DATA_DIR = "/data" if os.path.exists("/data") else "."
DB_PATH = os.path.join(DATA_DIR, "sa_study_app.db")
CHROMA_DIR = os.path.join(DATA_DIR, "sa_textbook_db")

def ingest_textbook(file_path: str, title: str, grade: str, subject: str, 
                    user_id: int, type: str = "textbook") -> int:
    print(f"📖 Loading: {title}...")
    
    documents = []
    
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\nChapter ", "\nCHAPTER ", "\n\n", "\n", " ", ""])
        documents = text_splitter.split_documents(raw_docs)
    elif file_path.endswith('.pptx'):
        prs = Presentation(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\n", " ", ""])
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_text = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            if slide_text:
                doc = Document(page_content="\n".join(slide_text), metadata={"source": file_path, "page": slide_num, "type": "slide"})
                documents.extend(text_splitter.split_documents([doc]))
    else:
        raise ValueError("Unsupported file format. Only PDF and PPTX are supported.")
    
    print(f"   Split into {len(documents)} chunks")
    
    filename = os.path.basename(file_path)
    book_id = add_book(title=title, grade=grade, subject=subject, filename=filename, file_path=file_path, total_chunks=len(documents), user_id=user_id, type=type)
    print(f"   ✅ Registered with ID: {book_id}")
    
    # Clean old chunks
    print(f"    Cleaning up old chunks for book_id {book_id}...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        db._collection.delete(where={"book_id": str(book_id)})
        print(f"   ✅ Old chunks deleted")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean old chunks: {e}")
    
    final_documents = []
    final_metadatas = []
    for i, doc in enumerate(documents):
        page_num = doc.metadata.get("page", (i // 5) + 1)
        section_label = "Slide" if type == "slides" else "Page"
        final_documents.append(doc.page_content)
        final_metadatas.append({
            "book_id": str(book_id), "book_title": title, "grade": grade, "subject": subject,
            "section": f"{section_label} {page_num}", "type": type, "user_id": str(user_id),
            "chunk_index": i, "page": int(page_num)
        })
    
         # 5. 💾 Save text page-by-page into SQLite for the Reader
    print("   💾 Saving page-by-page text for Reader...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS page_texts (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, page_num INTEGER NOT NULL, text_content TEXT NOT NULL)""")
    cursor.execute("DELETE FROM page_texts WHERE book_id = ?", (book_id,))
    
    for doc in documents:
        raw_page = doc.metadata.get("page", 1)
        is_slide = doc.metadata.get("type") == "slide"
        page_num = raw_page + 1 if not is_slide else raw_page
        
        cursor.execute(
            "INSERT INTO page_texts (book_id, page_num, text_content) VALUES (?, ?, ?)",
            (book_id, int(page_num), doc.page_content)
        )
    conn.commit()
    conn.close()
    print("   ✅ Page texts saved to database")

    # 6. Add to ChromaDB (Highly Optimized for Low Memory)
    print("   Embedding and storing in ChromaDB...")
    
    # Force garbage collection to free up RAM before the heavy AI step
    del raw_docs
    gc.collect()
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # CRITICAL: Process in very small batches to prevent RAM spikes
    batch_size = 10 
    total_batches = (len(documents) // batch_size) + 1
    
    for i in range(0, len(documents), batch_size):
        batch_docs = [d.page_content for d in documents[i:i + batch_size]]
        batch_meta = [d.metadata for d in documents[i:i + batch_size]]
        
        db.add_texts(texts=batch_docs, metadatas=batch_meta)
        print(f"   ✅ Embedded batch {i//batch_size + 1} of {total_batches}")
        
        # Delete batch data and force garbage collection immediately
        del batch_docs
        del batch_meta
        gc.collect()
    
    # Final cleanup
    del documents
    gc.collect()
    
    print(f"✅ Successfully ingested: {title} (ID: {book_id}) into {CHROMA_DIR}\n")
    return book_id