import os
import shutil
import json
import re
import sqlite3
import traceback
from typing import Optional, List
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from hybrid_tutor import hybrid_tutor
from ingest_textbook import ingest_textbook
from database import (get_all_books, get_book_by_id, delete_book, create_user, get_user_by_email, get_user_by_id)
from auth import (verify_password, get_password_hash, create_access_token, get_current_user)

load_dotenv()

# ==========================================
# PERSISTENT STORAGE PATHS
# ==========================================
DATA_DIR = "/data" if os.path.exists("/data") else "."
DB_PATH = os.path.join(DATA_DIR, "sa_study_app.db")
CHROMA_DIR = os.path.join(DATA_DIR, "sa_textbook_db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# ==========================================
# DATABASE INITIALIZATION
# ==========================================
def run_sql(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()

run_sql("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, book_id INTEGER NOT NULL, title TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
run_sql("CREATE TABLE IF NOT EXISTS quiz_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, questions TEXT NOT NULL, user_answers TEXT NOT NULL, results TEXT NOT NULL, score INTEGER NOT NULL, max_score INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
run_sql("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, book_id INTEGER, title TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
run_sql("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (conversation_id) REFERENCES conversations(id))")
run_sql("ALTER TABLE books ADD COLUMN is_public BOOLEAN DEFAULT 0")
run_sql("ALTER TABLE books ADD COLUMN admin_user_id INTEGER")
run_sql("CREATE TABLE IF NOT EXISTS page_texts (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, page_num INTEGER NOT NULL, text_content TEXT NOT NULL)")
run_sql("ALTER TABLE books ADD COLUMN is_past_paper BOOLEAN DEFAULT 0")
run_sql("ALTER TABLE books ADD COLUMN institution_type TEXT")
run_sql("ALTER TABLE books ADD COLUMN institution_name TEXT")
run_sql("ALTER TABLE books ADD COLUMN module_code TEXT")
run_sql("ALTER TABLE books ADD COLUMN year INTEGER")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@lingoai.co.za")
def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user["email"] != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Not authorized as admin")
    return current_user

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="SA Study App API", description="AI tutor for South African students", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# MODELS
# ==========================================
class UserRegister(BaseModel): email: EmailStr; password: str; name: str
class UserLogin(BaseModel): email: EmailStr; password: str
class Token(BaseModel): access_token: str; token_type: str
class UserResponse(BaseModel): id: int; email: str; name: str
class QuestionRequest(BaseModel): question: str; language: str; book_id: Optional[int] = None; chapter: Optional[str] = None; conversation_id: Optional[int] = None
class QuestionResponse(BaseModel): answer: str; source: str; conversation_id: Optional[int] = None
class BookResponse(BaseModel): id: int; title: str; grade: str; subject: str; filename: str; total_chunks: int; type: str; uploaded_at: str
class QuizGenerateRequest(BaseModel): book_id: int; num_questions: int = 5
class QuizGradeRequest(BaseModel): questions: list; user_answers: dict
class SessionCreateRequest(BaseModel): book_id: int; title: str
class QuizAttemptSaveRequest(BaseModel): session_id: int; questions: list; user_answers: dict; results: dict; score: int; max_score: int
class ConversationCreateRequest(BaseModel): book_id: Optional[int] = None; title: str
class MessageAddRequest(BaseModel): conversation_id: int; role: str; content: str
class TranslationRequest(BaseModel): text: str; target_language: str

# ==========================================
# HELPERS
# ==========================================
def extract_json_from_llm(text: str):
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```', '', text).strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    start_obj, end_obj = text.find('{'), text.rfind('}')
    if start_obj != -1 and end_obj != -1 and start_obj < end_obj:
        try: return json.loads(text[start_obj:end_obj+1])
        except json.JSONDecodeError: pass
    start_arr, end_arr = text.find('['), text.rfind(']')
    if start_arr != -1 and end_arr != -1 and start_arr < end_arr:
        try: return json.loads(text[start_arr:end_arr+1])
        except json.JSONDecodeError: pass
    text = text.replace('None', 'null').replace('True', 'true').replace('False', 'false')
    text = re.sub(r',\s*}', '}', text).replace(re.sub(r',\s*]', ']', text), text)
    try: return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ [JSON PARSE ERROR] {e}")
        raise

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def home(): return {"message": "🇿🇦 SA Study App API v3.0", "status": "healthy"}

@app.post("/register", response_model=UserResponse)
def register(user: UserRegister):
    if get_user_by_email(user.email): raise HTTPException(status_code=400, detail="Email already registered")
    user_id = create_user(email=user.email, password_hash=get_password_hash(user.password), name=user.name)
    if not user_id: raise HTTPException(status_code=500, detail="Failed to create user")
    return {"id": user_id, "email": user.email, "name": user.name}

@app.post("/login", response_model=Token)
def login(user: UserLogin):
    db_user = get_user_by_email(user.email)
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return {"access_token": create_access_token(data={"sub": str(db_user["id"])}), "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "email": current_user["email"], "name": current_user["name"]}

@app.get("/books", response_model=List[BookResponse])
def list_books(my_books_only: bool = False, current_user: dict = Depends(get_current_user)):
    return get_all_books(user_id=current_user["id"] if my_books_only else None)

@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: int, current_user: dict = Depends(get_current_user)):
    book = get_book_by_id(book_id)
    if not book: raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.delete("/books/{book_id}")
def remove_book(book_id: int, current_user: dict = Depends(get_current_user)):
    if not delete_book(book_id, current_user["id"]): raise HTTPException(status_code=404, detail="Book not found")
    return {"message": "Book removed"}

@app.post("/upload")
async def upload_textbook(file: UploadFile = File(...), title: str = Form(...), grade: str = Form(...), subject: str = Form(...), type: str = Form(default="textbook"), current_user: dict = Depends(get_current_user)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")): raise HTTPException(status_code=400, detail="Only PDF and PPTX allowed")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    try:
        book_id = ingest_textbook(file_path=file_path, title=title, grade=grade, subject=subject, user_id=current_user["id"], type=type)
        if type == "past_paper":
            conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
            cursor.execute("UPDATE books SET is_past_paper = 1 WHERE id = ?", (book_id,))
            conn.commit(); conn.close()
        return {"message": "Uploaded!", "book_id": book_id}
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    conv_id = request.conversation_id
    if not conv_id: conv_id = create_conversation(current_user["id"], request.book_id, request.question[:50])
    add_message(conv_id, "user", request.question)
    answer, source = hybrid_tutor(question=request.question, target_language=request.language, book_id=request.book_id, chapter=request.chapter)
    add_message(conv_id, "assistant", answer)
    return QuestionResponse(answer=answer, source=source, conversation_id=conv_id)

@app.post("/generate-quiz")
def generate_quiz(req: QuizGenerateRequest, current_user: dict = Depends(get_current_user)):
    book = get_book_by_id(req.book_id)
    if not book: raise HTTPException(status_code=404, detail="Book not found")
    b_id = book.get("id") if isinstance(book, dict) else getattr(book, "id", None)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = vectorstore.as_retriever(search_kwargs={"k": 8, "filter": {"book_id": str(b_id)}}).invoke("main concepts")
    if not docs: raise HTTPException(status_code=400, detail="No content found")
    context = "\n\n".join([d.page_content for d in docs])
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    prompt = f"""You are an expert teacher. Based ONLY on the context below, generate {req.num_questions} UNIQUE questions (mix of mcq, short, long).
Context: {context}
Output ONLY a valid JSON array. Each object must have: "id", "type", "question", "options" (array or null), "correct_answer" (int or null), "model_answer", "marks" (1 or 3). NO markdown."""
    return {"questions": extract_json_from_llm(llm.invoke(prompt).content)}

@app.post("/grade-quiz")
def grade_quiz(req: QuizGradeRequest, current_user: dict = Depends(get_current_user)):
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
    prompt = f"""Grade this quiz. Questions: {json.dumps(req.questions)} Answers: {json.dumps(req.user_answers)}. 
Output ONLY a JSON OBJECT: {{"total_score": 0, "max_score": 0, "graded_questions": [{{"id": 1, "is_correct": true, "awarded_marks": 1, "feedback": "..."}}]}}. NO markdown."""
    parsed = extract_json_from_llm(llm.invoke(prompt).content)
    if not isinstance(parsed, dict) or "graded_questions" not in parsed: raise ValueError("Invalid structure")
    return parsed

@app.post("/sessions")
def create_new_session(req: SessionCreateRequest, current_user: dict = Depends(get_current_user)):
    return {"session_id": create_session(current_user["id"], req.book_id, req.title)}
@app.get("/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)): return get_user_sessions(current_user["id"])
@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: int, current_user: dict = Depends(get_current_user)):
    attempts = get_session_quiz_history(session_id)
    for a in attempts:
        a["questions"] = json.loads(a["questions"]); a["user_answers"] = json.loads(a["user_answers"]); a["results"] = json.loads(a["results"])
    return attempts
@app.post("/sessions/{session_id}/save-attempt")
def save_quiz_attempt_endpoint(session_id: int, req: QuizAttemptSaveRequest, current_user: dict = Depends(get_current_user)):
    save_quiz_attempt(session_id, json.dumps(req.questions), json.dumps(req.user_answers), json.dumps(req.results), req.score, req.max_score)
    return {"message": "Saved"}

@app.post("/conversations")
def create_new_conversation(req: ConversationCreateRequest, current_user: dict = Depends(get_current_user)):
    return {"conversation_id": create_conversation(current_user["id"], req.book_id, req.title)}
@app.get("/conversations")
def list_conversations(current_user: dict = Depends(get_current_user)): return get_user_conversations(current_user["id"])
@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int, current_user: dict = Depends(get_current_user)): return get_conversation_messages(conversation_id)
@app.post("/conversations/{conversation_id}/messages")
def add_new_message(conversation_id: int, req: MessageAddRequest, current_user: dict = Depends(get_current_user)):
    add_message(conversation_id, req.role, req.content); return {"message": "Added"}
@app.delete("/conversations/{conversation_id}")
def delete_conv(conversation_id: int, current_user: dict = Depends(get_current_user)):
    if not delete_conversation(conversation_id, current_user["id"]): raise HTTPException(status_code=404, detail="Not found")
    return {"message": "Deleted"}
@app.post("/admin/upload-library")
async def upload_library_book(
    file: UploadFile = File(...), 
    title: str = Form(...), 
    grade: str = Form(...), 
    subject: str = Form(...), 
    type: str = Form(default="textbook"), 
    current_user: dict = Depends(get_current_admin)
):
    try:
        print(f"🚀 [UPLOAD] Starting upload for: {title}")
        
        if not file.filename or not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".pptx")):
            raise HTTPException(status_code=400, detail="Only PDF or PPTX files are allowed")
        
        # CRITICAL FIX: Stream directly to disk to prevent loading 100MB into RAM
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"💾 [UPLOAD] File saved to disk. Starting AI processing...")
        
        book_id = ingest_textbook(file_path=file_path, title=title, grade=grade, subject=subject, user_id=current_user["id"], type=type)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE books SET is_public = 1, admin_user_id = ? WHERE id = ?", (current_user["id"], book_id))
        conn.commit()
        conn.close()
        
        print(f"✅ [UPLOAD] Success! Book ID: {book_id}")
        return {"message": "Added to Library!", "book_id": book_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌❌❌ [UPLOAD] CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Server error during upload: {str(e)}")
@app.get("/library")
def get_public_library():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT * FROM books WHERE is_public = 1 ORDER BY uploaded_at DESC")
    books = [dict(b) for b in cursor.fetchall()]; conn.close()
    return books

@app.delete("/admin/library/{book_id}")
def delete_library_book(book_id: int, current_user: dict = Depends(get_current_admin)):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM books WHERE id = ? AND (is_public = 1 OR is_past_paper = 1)", (book_id,))
    book = cursor.fetchone()
    if not book: 
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    file_path = book[0]
    try:
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
        db._collection.delete(where={"book_id": str(book_id)})
    except Exception as e: 
        print(f"⚠️ Chroma delete warning: {e}")
    cursor.execute("DELETE FROM page_texts WHERE book_id = ?", (book_id,))
    cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
    conn.commit(); conn.close()
    if os.path.exists(file_path): 
        os.remove(file_path)
    return {"message": "Deleted"}

@app.get("/library/{book_id}/content")
def get_book_content(book_id: int):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT is_public FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone(); conn.close()
    if not book or not book[0]: raise HTTPException(status_code=404, detail="Not found")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    results = vectorstore._collection.get(where={"book_id": str(book_id)}, include=["documents", "metadatas"])
    chunks = []
    if results and results['documents']:
        zipped = list(zip(results['documents'], results['metadatas']))
        zipped.sort(key=lambda x: int(x[1].get('page', x[1].get('chunk_index', 0))))
        chunks = [{"text": doc, "metadata": meta} for doc, meta in zipped]
    return {"chunks": chunks}

@app.get("/library/{book_id}/pdf")
def get_pdf_file(book_id: int):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT file_path, is_public, is_past_paper FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone(); conn.close()
    if not book or (not book['is_public'] and not book['is_past_paper']): raise HTTPException(status_code=404, detail="Not found")
    if not os.path.exists(book['file_path']): raise HTTPException(status_code=404, detail="File missing")
    headers = {"Cache-Control": "public, max-age=31536000, immutable", "Accept-Ranges": "bytes"}
    return FileResponse(book['file_path'], media_type="application/pdf", headers=headers)

@app.get("/library/{book_id}/page/{page_num}/text")
def get_page_text(book_id: int, page_num: int):
    """Get the raw text for the EXACT page number."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT text_content FROM page_texts WHERE book_id = ? AND page_num = ?", (book_id, page_num))
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0]:
        return {"text": row[0]}
    return {"text": ""}

@app.post("/admin/upload-past-paper")
async def upload_past_paper(file: UploadFile = File(...), institution_type: str = Form(...), institution_name: str = Form(...), subject_or_module: str = Form(...), module_code: str = Form(default=""), year: int = Form(...), current_user: dict = Depends(get_current_admin)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")): raise HTTPException(status_code=400, detail="Only PDF/PPTX")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    try:
        book_id = ingest_textbook(file_path=file_path, title=f"{subject_or_module} {year}", grade=str(year), subject=subject_or_module, user_id=current_user["id"], type="past_paper")
        conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
        cursor.execute("UPDATE books SET is_past_paper = 1, institution_type = ?, institution_name = ?, module_code = ?, year = ? WHERE id = ?", (institution_type, institution_name, module_code, year, book_id))
        conn.commit(); conn.close()
        return {"message": "Added to Past Papers!", "book_id": book_id}
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/past-papers")
def get_past_papers(institution_type: Optional[str] = None, year: Optional[int] = None):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    query = "SELECT * FROM books WHERE is_past_paper = 1"; params = []
    if institution_type: query += " AND institution_type = ?"; params.append(institution_type)
    if year: query += " AND year = ?"; params.append(year)
    query += " ORDER BY year DESC"
    cursor.execute(query, params); papers = [dict(p) for p in cursor.fetchall()]; conn.close()
    return papers

@app.get("/past-papers/{book_id}/download")
def download_past_paper(book_id: int):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT file_path, is_past_paper FROM books WHERE id = ?", (book_id,))
    book = cursor.fetchone(); conn.close()
    if not book or not book['is_past_paper']: raise HTTPException(status_code=404, detail="Not found")
    if not os.path.exists(book['file_path']): raise HTTPException(status_code=404, detail="File missing")
    headers = {"Cache-Control": "public, max-age=31536000, immutable", "Accept-Ranges": "bytes"}
    return FileResponse(book['file_path'], media_type="application/pdf", filename=os.path.basename(book['file_path']), headers=headers)

@app.post("/translate")
def translate_text(req: TranslationRequest, current_user: dict = Depends(get_current_user)):
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    prompt = f"""Translate to {req.target_language}. Maintain formatting. NO conversational filler. Output ONLY translated text.\n\n{req.text}"""
    return {"translated_text": llm.invoke(prompt).content.strip()}

# ==========================================
# DB HELPERS
# ==========================================
def create_session(user_id, book_id, title):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor(); cursor.execute("INSERT INTO sessions (user_id, book_id, title) VALUES (?, ?, ?)", (user_id, book_id, title)); sid = cursor.lastrowid; conn.commit(); conn.close(); return sid
def get_user_sessions(user_id):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT s.*, b.title as book_title FROM sessions s JOIN books b ON s.book_id = b.id WHERE s.user_id = ?", (user_id,)); res = [dict(r) for r in cursor.fetchall()]; conn.close(); return res
def save_quiz_attempt(sid, q, a, r, s, m):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor(); cursor.execute("INSERT INTO quiz_attempts (session_id, questions, user_answers, results, score, max_score) VALUES (?, ?, ?, ?, ?, ?)", (sid, q, a, r, s, m)); conn.commit(); conn.close()
def get_session_quiz_history(sid):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT * FROM quiz_attempts WHERE session_id = ?", (sid,)); res = [dict(r) for r in cursor.fetchall()]; conn.close(); return res
def create_conversation(user_id, book_id, title):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor(); cursor.execute("INSERT INTO conversations (user_id, book_id, title) VALUES (?, ?, ?)", (user_id, book_id, title)); cid = cursor.lastrowid; conn.commit(); conn.close(); return cid
def get_user_conversations(user_id):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT c.*, b.title as book_title FROM conversations c LEFT JOIN books b ON c.book_id = b.id WHERE c.user_id = ?", (user_id,)); res = [dict(r) for r in cursor.fetchall()]; conn.close(); return res
def get_conversation_messages(cid):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT * FROM messages WHERE conversation_id = ?", (cid,)); res = [dict(r) for r in cursor.fetchall()]; conn.close(); return res
def add_message(cid, role, content):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor(); cursor.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (cid, role, content)); cursor.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (cid,)); conn.commit(); conn.close()
def delete_conversation(cid, user_id):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor(); cursor.execute("SELECT user_id FROM conversations WHERE id = ?", (cid,)); c = cursor.fetchone()
    if not c or c[0] != user_id: conn.close(); return False
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,)); cursor.execute("DELETE FROM conversations WHERE id = ?", (cid,)); conn.commit(); conn.close(); return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)