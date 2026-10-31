import os
import sqlite3
import gc
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from database import add_book
from pptx import Presentation

load_dotenv()

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\nChapter ", "\nCHAPTER ", "\n\n", "\n", " ", ""])
        documents = text_splitter.split_documents(raw_docs)
        del raw_docs # Free memory immediately
        gc.collect()
    elif file_path.endswith('.pptx'):
        prs = Presentation(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, separators=["\n", " ", ""])
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
    print(f"   🧹 Cleaning up old chunks for book_id {book_id}...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        db._collection.delete(where={"book_id": str(book_id)})
        print(f"   ✅ Old chunks deleted")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not clean old chunks: {e}")
    
    # Save page-by-page text to SQLite
    print("   💾 Saving page-by-page text for Reader...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS page_texts (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, page_num INTEGER NOT NULL, text_content TEXT NOT NULL)""")
    cursor.execute("DELETE FROM page_texts WHERE book_id = ?", (book_id,))
    for doc in documents:
        raw_page = doc.metadata.get("page", 1)
        is_slide = doc.metadata.get("type") == "slide"
        page_num = raw_page + 1 if not is_slide else raw_page
        cursor.execute("INSERT INTO page_texts (book_id, page_num, text_content) VALUES (?, ?, ?)", (book_id, int(page_num), doc.page_content))
    conn.commit()
    conn.close()
    print("   ✅ Page texts saved to database")

    # Add to ChromaDB (Highly Optimized for 512MB RAM)
    print("   Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # Batch size of 30 is the sweet spot: fast enough to avoid 100s timeout, small enough to avoid OOM
    batch_size = 30
    total_batches = (len(documents) // batch_size) + 1
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        texts = [d.page_content for d in batch_docs]
        
        # CRITICAL FIX: Inject book_id and book_title into the metadata!
        # This is how the AI knows which book these text chunks belong to.
        metas = []
        for d in batch_docs:
            meta = d.metadata.copy()
            meta["book_id"] = str(book_id)
            meta["book_title"] = title
            metas.append(meta)
        
        db.add_texts(texts=texts, metadatas=metas)
        print(f"   ✅ Embedded batch {i//batch_size + 1} of {total_batches}")
        
        # Aggressive memory cleanup after every single batch
        del texts
        del metas
        del batch_docs
        gc.collect()
    
    # Final cleanup
    del documents
    gc.collect()
    
    print(f"✅ Successfully ingested: {title} (ID: {book_id}) into {CHROMA_DIR}\n")
    return book_id