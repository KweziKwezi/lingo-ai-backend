import os
import shutil
import json
import re
import sqlite3
from typing import Optional, List
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

# LangChain imports
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Local imports
from hybrid_tutor import hybrid_tutor
from ingest_textbook import ingest_textbook
from database import (
    get_all_books, get_book_by_id, delete_book,
    create_user, get_user_by_email, get_user_by_id
)
from auth import (
    verify_password, get_password_hash, create_access_token,
    get_current_user
)

load_dotenv()

# ==========================================
# DATABASE INITIALIZATION
# ==========================================
def ensure_new_tables():
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, book_id INTEGER NOT NULL, title TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    cursor.execute("CREATE TABLE IF NOT EXISTS quiz_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, questions TEXT NOT NULL, user_answers TEXT NOT NULL, results TEXT NOT NULL, score INTEGER NOT NULL, max_score INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()
ensure_new_tables()

def ensure_chat_tables():
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, book_id INTEGER, title TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (conversation_id) REFERENCES conversations(id))""")
    conn.commit()
    conn.close()
ensure_chat_tables()

def ensure_public_library_schema():
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE books ADD COLUMN is_public BOOLEAN DEFAULT 0")
        cursor.execute("ALTER TABLE books ADD COLUMN admin_user_id INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()
ensure_public_library_schema()
def ensure_past_papers_schema():
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE books ADD COLUMN is_past_paper BOOLEAN DEFAULT 0")
        cursor.execute("ALTER TABLE books ADD COLUMN institution_type TEXT") # 'High School' or 'University'
        cursor.execute("ALTER TABLE books ADD COLUMN institution_name TEXT")
        cursor.execute("ALTER TABLE books ADD COLUMN module_code TEXT")
        cursor.execute("ALTER TABLE books ADD COLUMN year INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass # Columns already exist
    conn.close()

ensure_past_papers_schema()

def ensure_page_text_schema():
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS page_texts (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, page_num INTEGER NOT NULL, text_content TEXT NOT NULL)""")
    conn.commit()
    conn.close()
ensure_page_text_schema()

# Admin Security Check
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@lingoai.co.za")

def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user["email"] != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized as admin")
    return current_user

# Create uploads folder if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app = FastAPI(
    title="SA Study App API",
    description="AI tutor for South African students with authentication",
    version="3.0.0"
)

# ==========================================
# CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str

class QuestionRequest(BaseModel):
    question: str
    language: str
    book_id: Optional[int] = None
    chapter: Optional[str] = None
    conversation_id: Optional[int] = None

class QuestionResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[int] = None

class BookResponse(BaseModel):
    id: int
    title: str
    grade: str
    subject: str
    filename: str
    total_chunks: int
    type: str
    uploaded_at: str

class QuizGenerateRequest(BaseModel):
    book_id: int
    num_questions: int = 5

class QuizGradeRequest(BaseModel):
    questions: list
    user_answers: dict

class SessionCreateRequest(BaseModel):
    book_id: int
    title: str

class QuizAttemptSaveRequest(BaseModel):
    session_id: int
    questions: list
    user_answers: dict
    results: dict
    score: int
    max_score: int

class ConversationCreateRequest(BaseModel):
    book_id: Optional[int] = None
    title: str

class MessageAddRequest(BaseModel):
    conversation_id: int
    role: str
    content: str

class TranslationRequest(BaseModel):
    text: str
    target_language: str

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_json_from_llm(text: str):
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```', '', text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    start_obj = text.find('{')
    end_obj = text.rfind('}')
    if start_obj != -1 and end_obj != -1 and start_obj < end_obj:
        try:
            return json.loads(text[start_obj:end_obj+1])
        except json.JSONDecodeError:
            pass
            
    start_arr = text.find('[')
    end_arr = text.rfind(']')
    if start_arr != -1 and end_arr != -1 and start_arr < end_arr:
        try:
            return json.loads(text[start_arr:end_arr+1])
        except json.JSONDecodeError:
            pass
            
    text = text.replace('None', 'null')
    text = text.replace('True', 'true')
    text = text.replace('False', 'false')
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ [JSON PARSE ERROR] Failed to parse. Raw snippet: {text[:300]}")
        print(f"❌ [JSON PARSE ERROR] Details: {e}")
        raise

# ==========================================
# BASIC & AUTH ENDPOINTS
# ==========================================
@app.get("/")
def home():
    return {"message": "🇿🇦 SA Study App API v3.0", "status": "healthy"}

@app.post("/register", response_model=UserResponse)
def register(user: UserRegister):
    existing_user = get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    password_hash = get_password_hash(user.password)
    user_id = create_user(email=user.email, password_hash=password_hash, name=user.name)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")
    return {"id": user_id, "email": user.email, "name": user.name}

@app.post("/login", response_model=Token)
def login(user: UserLogin):
    db_user = get_user_by_email(user.email)
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": str(db_user["id"])})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "email": current_user["email"], "name": current_user["name"]}

# ==========================================
# LIBRARY & UPLOAD ENDPOINTS
# ==========================================
@app.get("/books", response_model=List[BookResponse])
def list_books(my_books_only: bool = False, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"] if my_books_only else None
    return get_all_books(user_id=user_id)

@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: int, current_user: dict = Depends(get_current_user)):
    book = get_book_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.delete("/books/{book_id}")
def remove_book(book_id: int, current_user: dict = Depends(get_current_user)):
    success = delete_book(book_id, current_user["id"])
    if not success:
        raise HTTPException(status_code=404, detail="Book not found or not owned by you")
    return {"message": "Book removed from library"}

@app.post("/upload")
async def upload_textbook(
    file: UploadFile = File(...),
    title: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    type: str = Form(default="textbook"),
    current_user: dict = Depends(get_current_user)
):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")):
        raise HTTPException(status_code=400, detail="Only PDF and PPTX files are allowed")

    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        book_id = ingest_textbook(file_path=file_path, title=title, grade=grade, subject=subject, user_id=current_user["id"], type=type)
        return {"message": "File successfully uploaded!", "book_id": book_id, "title": title, "type": type}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# ==========================================
# AI TUTOR & QUIZ ENDPOINTS
# ==========================================
@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    try:
        conversation_id = request.conversation_id
        if not conversation_id:
            title = request.question[:50] + "..." if len(request.question) > 50 else request.question
            conversation_id = create_conversation(current_user["id"], request.book_id, title)
        
        add_message(conversation_id, "user", request.question)
        answer, source = hybrid_tutor(question=request.question, target_language=request.language, book_id=request.book_id, chapter=request.chapter)
        add_message(conversation_id, "assistant", answer)
        
        return QuestionResponse(answer=answer, source=source, conversation_id=conversation_id)
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz")
def generate_quiz(req: QuizGenerateRequest, current_user: dict = Depends(get_current_user)):
    try:
        book = get_book_by_id(req.book_id)
        if not book:
            raise HTTPException(status_code=404, detail="Book not found")
        
        b_id = book.get("id") if isinstance(book, dict) else getattr(book, "id", None)
        if not b_id:
            raise HTTPException(status_code=500, detail="Could not determine book ID")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(persist_directory="./sa_textbook_db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8, "filter": {"book_id": str(b_id)}})
        docs = retriever.invoke("main concepts and details")
        
        if len(docs) == 0:
            raise HTTPException(status_code=400, detail="No content found for this book.")

        context = "\n\n".join([doc.page_content for doc in docs])
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

        prompt = f"""You are an expert teacher. Based ONLY on the context below, generate {req.num_questions} questions.
        CRITICAL INSTRUCTION: These questions MUST BE UNIQUE and varied. Focus on different facts, definitions, and concepts.
        The questions MUST be a mix of these three types: "mcq", "short", and "long".
        Context: {context}
        Output ONLY a valid JSON array of objects. Each object must have:
        - "id": integer (1 to {req.num_questions})
        - "type": string ("mcq", "short", or "long")
        - "question": string
        - "options": array of 4 strings (ONLY for "mcq", else null)
        - "correct_answer": integer 0-3 (ONLY for "mcq", else null)
        - "model_answer": string (The correct answer/explanation for grading short/long)
        - "marks": integer (1 for mcq/short, 3 for long)
        CRITICAL: Output ONLY the raw JSON array. Do NOT use markdown formatting."""

        response = llm.invoke(prompt)
        return {"questions": extract_json_from_llm(response.content)}
    except Exception as e:
        print(f"❌ [QUIZ] CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/grade-quiz")
def grade_quiz(req: QuizGradeRequest, current_user: dict = Depends(get_current_user)):
    try:
        llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
        prompt = f"""You are an expert teacher grading a quiz. 
        Questions: {json.dumps(req.questions)}
        Student Answers: {json.dumps(req.user_answers)}
        Grade the student's work. For MCQs, check if the index matches 'correct_answer'. For short/long answers, compare to 'model_answer' and award partial marks.
        CRITICAL: For EVERY question that is incorrect or missing, you MUST include the correct answer in the feedback.
        Output ONLY a valid JSON OBJECT (starting with {{ and ending with }}), NOT a list. It must have this EXACT structure:
        {{
          "total_score": 0,
          "max_score": 0,
          "graded_questions": [
            {{
              "id": 1,
              "is_correct": true,
              "awarded_marks": 1,
              "feedback": "Great job!"
            }}
          ]
        }}
        Do NOT wrap the output in ```json ... ``` markdown blocks."""

        response = llm.invoke(prompt)
        parsed_data = extract_json_from_llm(response.content)
        
        if not isinstance(parsed_data, dict) or "graded_questions" not in parsed_data:
            raise ValueError(f"AI returned invalid grading structure. Got type: {type(parsed_data)}")
        return parsed_data
    except Exception as e:
        print(f"❌ [GRADE] Critical Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")

# ==========================================
# SESSION & CHAT HISTORY ENDPOINTS
# ==========================================
@app.post("/sessions")
def create_new_session(req: SessionCreateRequest, current_user: dict = Depends(get_current_user)):
    return {"session_id": create_session(current_user["id"], req.book_id, req.title), "message": "Session created"}

@app.get("/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)):
    return get_user_sessions(current_user["id"])

@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: int, current_user: dict = Depends(get_current_user)):
    attempts = get_session_quiz_history(session_id)
    for attempt in attempts:
        attempt["questions"] = json.loads(attempt["questions"])
        attempt["user_answers"] = json.loads(attempt["user_answers"])
        attempt["results"] = json.loads(attempt["results"])
    return attempts

@app.post("/sessions/{session_id}/save-attempt")
def save_quiz_attempt_endpoint(session_id: int, req: QuizAttemptSaveRequest, current_user: dict = Depends(get_current_user)):
    save_quiz_attempt(session_id=session_id, questions=json.dumps(req.questions), answers=json.dumps(req.user_answers), results=json.dumps(req.results), score=req.score, max_score=req.max_score)
    return {"message": "Quiz attempt saved"}

@app.post("/conversations")
def create_new_conversation(req: ConversationCreateRequest, current_user: dict = Depends(get_current_user)):
    return {"conversation_id": create_conversation(current_user["id"], req.book_id, req.title), "message": "Conversation created"}

@app.get("/conversations")
def list_conversations(current_user: dict = Depends(get_current_user)):
    return get_user_conversations(current_user["id"])

@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int, current_user: dict = Depends(get_current_user)):
    return get_conversation_messages(conversation_id)

@app.post("/conversations/{conversation_id}/messages")
def add_new_message(conversation_id: int, req: MessageAddRequest, current_user: dict = Depends(get_current_user)):
    add_message(conversation_id, req.role, req.content)
    return {"message": "Message added"}

@app.delete("/conversations/{conversation_id}")
def delete_conv(conversation_id: int, current_user: dict = Depends(get_current_user)):
    if not delete_conversation(conversation_id, current_user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found or not owned by you")
    return {"message": "Conversation deleted"}

# ==========================================
# ADMIN & PUBLIC LIBRARY ENDPOINTS
# ==========================================
@app.post("/admin/upload-library")
async def upload_library_book(
    file: UploadFile = File(...),
    title: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    type: str = Form(default="textbook"),
    current_user: dict = Depends(get_current_admin)
):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")):
        raise HTTPException(status_code=400, detail="Only PDF and PPTX files are allowed")

    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        book_id = ingest_textbook(file_path=file_path, title=title, grade=grade, subject=subject, user_id=current_user["id"], type=type)
        conn = sqlite3.connect("sa_study_app.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE books SET is_public = 1, admin_user_id = ? WHERE id = ?", (current_user["id"], book_id))
        conn.commit()
        conn.close()
        return {"message": "Book successfully added to the Public Library!", "book_id": book_id}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/library")
def get_public_library():
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM books WHERE is_public = 1 ORDER BY uploaded_at DESC")
    books = cursor.fetchall()
    conn.close()
    return [dict(b) for b in books]

@app.delete("/admin/library/{book_id}")
def delete_library_book(book_id: int, current_user: dict = Depends(get_current_admin)):
    """Admin only: Delete a book from the public library, ChromaDB, and disk."""
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT file_path FROM books WHERE id = ? AND is_public = 1", (book_id,))
    book = cursor.fetchone()
    
    if not book:
        conn.close()
        raise HTTPException(status_code=404, detail="Public book not found")
        
    file_path = book[0]
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory="./sa_textbook_db", embedding_function=embeddings)
        collection = db._collection
        collection.delete(where={"book_id": str(book_id)})
    except Exception as e:
        print(f"⚠️ Warning: Could not delete from ChromaDB: {e}")
        
    cursor.execute("DELETE FROM page_texts WHERE book_id = ?", (book_id,))
    cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()
    
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"message": "Book successfully deleted from library"}

@app.get("/library/{book_id}/content")
def get_book_content(book_id: int):
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT is_public FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone()
    conn.close()
    if not book or not book[0]:
        raise HTTPException(status_code=404, detail="Book not found or not public")

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(persist_directory="./sa_textbook_db", embedding_function=embeddings)
        collection = vectorstore._collection
        results = collection.get(where={"book_id": str(book_id)}, include=["documents", "metadatas"])
        
        chunks = []
        if results and results['documents']:
            zipped = list(zip(results['documents'], results['metadatas']))
            def get_sort_key(item):
                meta = item[1]
                for key in ['chunk_index', 'page', 'chunk', 'index', 'chunk_id']:
                    if key in meta:
                        try: return int(meta[key])
                        except (ValueError, TypeError): pass
                return 0
            zipped.sort(key=get_sort_key)
            chunks = [{"text": doc, "metadata": meta} for doc, meta in zipped]
        return {"chunks": chunks}
    except Exception as e:
        print(f"❌ [LIBRARY] Error fetching content: {e}")
        raise HTTPException(status_code=500, detail="Failed to load book content")

@app.get("/library/{book_id}/pdf")
def get_pdf_file(book_id: int):
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, is_public FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone()
    conn.close()
    
    if not book or not book['is_public']:
        raise HTTPException(status_code=404, detail="Book not found")
    if not os.path.exists(book['file_path']):
        raise HTTPException(status_code=404, detail="PDF file missing on server")
    return FileResponse(book['file_path'], media_type="application/pdf")

@app.get("/library/{book_id}/page/{page_num}/text")
def get_page_text(book_id: int, page_num: int):
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT text_content FROM page_texts WHERE book_id = ? AND page_num = ?", (book_id, page_num))
    row = cursor.fetchone()
    conn.close()
    return {"text": row[0]} if row else {"text": ""}
    @app.post("/admin/upload-past-paper")
async def upload_past_paper(
    file: UploadFile = File(...),
    institution_type: str = Form(...), # 'High School' or 'University'
    institution_name: str = Form(...),
    subject_or_module: str = Form(...),
    module_code: str = Form(default=""),
    year: int = Form(...),
    current_user: dict = Depends(get_current_admin)
):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")):
        raise HTTPException(status_code=400, detail="Only PDF and PPTX files are allowed")

    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Ingest it just like a textbook so it gets chunked for translation & AI chat
        book_id = ingest_textbook(
            file_path=file_path, 
            title=f"{subject_or_module} {year}", 
            grade=str(year), 
            subject=subject_or_module, 
            user_id=current_user["id"], 
            type="past_paper"
        )
        
        # Update with past paper specific metadata
        conn = sqlite3.connect("sa_study_app.db")
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE books SET 
            is_past_paper = 1, 
            institution_type = ?, 
            institution_name = ?, 
            module_code = ?, 
            year = ? 
            WHERE id = ?
        """, (institution_type, institution_name, module_code, year, book_id))
        conn.commit()
        conn.close()
        
        return {"message": "Past paper successfully added to the Hub!", "book_id": book_id}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/past-papers")
def get_past_papers(
    institution_type: Optional[str] = None,
    year: Optional[int] = None
):
    """Get past papers with optional filters."""
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM books WHERE is_past_paper = 1"
    params = []
    
    if institution_type:
        query += " AND institution_type = ?"
        params.append(institution_type)
    if year:
        query += " AND year = ?"
        params.append(year)
        
    query += " ORDER BY year DESC, uploaded_at DESC"
    
    cursor.execute(query, params)
    papers = cursor.fetchall()
    conn.close()
    return [dict(p) for p in papers]

@app.get("/past-papers/{book_id}/download")
def download_past_paper(book_id: int):
    """Force download the past paper PDF."""
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, is_past_paper FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone()
    conn.close()
    
    if not book or not book[0]:
        raise HTTPException(status_code=404, detail="Past paper not found")
    if not os.path.exists(book['file_path']):
        raise HTTPException(status_code=404, detail="File missing on server")
        
    # media_type='application/pdf' with filename forces browser to download
    return FileResponse(
        book['file_path'], 
        media_type="application/pdf",
        filename=os.path.basename(book['file_path'])
    )

@app.post("/translate")
def translate_text(req: TranslationRequest, current_user: dict = Depends(get_current_user)):
    """Translate a single chunk of text to save API credits."""
    try:
        llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
        prompt = f"""You are an expert translator. Translate the following text into {req.target_language}. 
        Maintain the original formatting, tone, and educational meaning. 
        Do NOT add any conversational filler, greetings, or explanations. Output ONLY the translated text.
        
        Text to translate:
        {req.text}
        """
        response = llm.invoke(prompt)
        return {"translated_text": response.content.strip()}
    except Exception as e:
        print(f"❌ [TRANSLATE] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# ==========================================
# DATABASE HELPER FUNCTIONS
# ==========================================
def create_session(user_id: int, book_id: int, title: str) -> int:
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (user_id, book_id, title) VALUES (?, ?, ?)", (user_id, book_id, title))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_user_sessions(user_id: int):
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT s.*, b.title as book_title, b.subject, b.grade FROM sessions s JOIN books b ON s.book_id = b.id WHERE s.user_id = ? ORDER BY s.created_at DESC", (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return [dict(row) for row in sessions]

def save_quiz_attempt(session_id: int, questions: str, answers: str, results: str, score: int, max_score: int):
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO quiz_attempts (session_id, questions, user_answers, results, score, max_score) VALUES (?, ?, ?, ?, ?, ?)", (session_id, questions, answers, results, score, max_score))
    conn.commit()
    conn.close()

def get_session_quiz_history(session_id: int):
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM quiz_attempts WHERE session_id = ? ORDER BY created_at DESC", (session_id,))
    attempts = cursor.fetchall()
    conn.close()
    return [dict(row) for row in attempts]

def create_conversation(user_id: int, book_id: int, title: str) -> int:
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_id, book_id, title) VALUES (?, ?, ?)", (user_id, book_id, title))
    conv_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def get_user_conversations(user_id: int):
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT c.*, b.title as book_title FROM conversations c LEFT JOIN books b ON c.book_id = b.id WHERE c.user_id = ? ORDER BY c.updated_at DESC", (user_id,))
    conversations = cursor.fetchall()
    conn.close()
    return [dict(row) for row in conversations]

def get_conversation_messages(conversation_id: int):
    conn = sqlite3.connect("sa_study_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", (conversation_id,))
    messages = cursor.fetchall()
    conn.close()
    return [dict(row) for row in messages]

def add_message(conversation_id: int, role: str, content: str):
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (conversation_id, role, content))
    cursor.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

def delete_conversation(conversation_id: int, user_id: int) -> bool:
    conn = sqlite3.connect("sa_study_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM conversations WHERE id = ?", (conversation_id,))
    conv = cursor.fetchone()
    if not conv or conv[0] != user_id:
        conn.close()
        return False
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return True

# ==========================================
# RUN THE SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting SA Study App API server...")
    print("📖 API Documentation: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)