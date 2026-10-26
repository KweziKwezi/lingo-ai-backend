import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from database import add_book
from pptx import Presentation

load_dotenv()

def ingest_textbook(file_path: str, title: str, grade: str, subject: str, 
                    user_id: int, type: str = "textbook") -> int:
    """
    Ingest a textbook or slide deck into the vector database.
    Automatically deletes old chunks for this book before ingesting.
    """
    print(f"📖 Loading: {title}...")
    
    documents = []
    
    # 1. Extract text and preserve page/slide metadata
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\nChapter ", "\nCHAPTER ", "\n\n", "\n", " ", ""]
        )
        
        documents = text_splitter.split_documents(raw_docs)
        
    elif file_path.endswith('.pptx'):
        prs = Presentation(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n", " ", ""]
        )
        
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            
            if slide_text:
                full_slide_text = "\n".join(slide_text)
                doc = Document(
                    page_content=full_slide_text,
                    metadata={"source": file_path, "page": slide_num, "type": "slide"}
                )
                split_docs = text_splitter.split_documents([doc])
                documents.extend(split_docs)
    else:
        raise ValueError("Unsupported file format. Only PDF and PPTX are supported.")
    
    print(f"   Split into {len(documents)} chunks")
    
    # 2. Add book to database FIRST to get the book_id
    filename = os.path.basename(file_path)
    book_id = add_book(
        title=title,
        grade=grade,
        subject=subject,
        filename=filename,
        file_path=file_path,
        total_chunks=len(documents),
        user_id=user_id,
        type=type
    )
    print(f"   ✅ Registered with ID: {book_id}")
    
    # 3. 🔥 CRITICAL: Delete ALL old chunks for this book_id before adding new ones
    print(f"   🧹 Cleaning up old chunks for book_id {book_id}...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory="./sa_textbook_db", embedding_function=embeddings)
        collection = db._collection
        
        # Delete all documents with this book_id
        collection.delete(where={"book_id": str(book_id)})
        print(f"   ✅ Old chunks deleted")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clean old chunks: {e}")
    
    # 4. Prepare final metadata for ChromaDB
    final_documents = []
    final_metadatas = []
    
    for i, doc in enumerate(documents):
        page_num = doc.metadata.get("page", (i // 5) + 1)
        section_label = "Slide" if type == "slides" else "Page"
        
        final_documents.append(doc.page_content)
        final_metadatas.append({
            "book_id": str(book_id),
            "book_title": title,
            "grade": grade,
            "subject": subject,
            "section": f"{section_label} {page_num}",
            "type": type,
            "user_id": str(user_id),
            "chunk_index": i,
            "page": int(page_num)
        })
        # 3.5. Save text page-by-page into SQLite for the Reader
    print("   💾 Saving page-by-page text for Reader...")
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    # Delete old page texts for this book if re-uploading
    cursor.execute("DELETE FROM page_texts WHERE book_id = ?", (book_id,))
    
    for doc in documents:
        page_num = doc.metadata.get("page", 1)
        cursor.execute(
            "INSERT INTO page_texts (book_id, page_num, text_content) VALUES (?, ?, ?)",
            (book_id, int(page_num), doc.page_content)
        )
    conn.commit()
    conn.close()
    # 5. Add to ChromaDB
    print("   Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db = Chroma(
        persist_directory="./sa_textbook_db",
        embedding_function=embeddings
    )
    
    db.add_texts(
        texts=final_documents,
        metadatas=final_metadatas
    )
    
    print(f"✅ Successfully ingested: {title} (ID: {book_id}) into ./sa_textbook_db\n")
    return book_id