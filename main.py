import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from dotenv import load_dotenv
import json
import re
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
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

# Create uploads folder if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app = FastAPI(
    title="SA Study App API",
    description="AI tutor for South African students with authentication",
    version="3.0.0"
)

# Allow mobile apps and websites to talk to this API
from fastapi.middleware.cors import CORSMiddleware

# ... (your other imports and app = FastAPI(...) code) ...

# ADD THIS EXACT BLOCK:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allows your Vercel app, localhost, etc.
    allow_credentials=False,      # MUST be False when allow_origins is "*"
    allow_methods=["*"],          # Allows GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # Allows all headers (including Authorization)
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

class QuestionResponse(BaseModel):
    answer: str
    source: str

class BookResponse(BaseModel):
    id: int
    title: str
    grade: str
    subject: str
    filename: str
    total_chunks: int
    type: str
    uploaded_at: str

# ==========================================
# BASIC ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "message": "🇿🇦 SA Study App API v3.0",
        "features": [
            "User authentication",
            "Multi-textbook library",
            "Native language typing & responding",
            "PDF and PPTX upload"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ==========================================
# AUTHENTICATION ENDPOINTS
# ==========================================

@app.post("/register", response_model=UserResponse)
def register(user: UserRegister):
    """Register a new user."""
    # Check if user already exists
    existing_user = get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash the password
    password_hash = get_password_hash(user.password)
    
    # Create the user
    user_id = create_user(
        email=user.email,
        password_hash=password_hash,
        name=user.name
    )
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    return {
        "id": user_id,
        "email": user.email,
        "name": user.name
    }

@app.post("/login", response_model=Token)
def login(user: UserLogin):
    """Login and get an access token."""
    # Get the user
    db_user = get_user_by_email(user.email)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Verify the password
    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": str(db_user["id"])})
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info."""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"]
    }

# ==========================================
# LIBRARY ENDPOINTS (Authenticated)
# ==========================================

@app.get("/books", response_model=List[BookResponse])
def list_books(
    my_books_only: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """List books. If my_books_only=True, only show user's own books."""
    user_id = current_user["id"] if my_books_only else None
    books = get_all_books(user_id=user_id)
    return books

@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: int, current_user: dict = Depends(get_current_user)):
    """Get details of a specific book."""
    book = get_book_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.delete("/books/{book_id}")
def remove_book(book_id: int, current_user: dict = Depends(get_current_user)):
    """Remove a book from the library (only if owned by user)."""
    user_id = current_user["id"]
    success = delete_book(book_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Book not found or not owned by you")
    return {"message": "Book removed from library"}

# ==========================================
# UPLOAD ENDPOINT (Authenticated)
# ==========================================

@app.post("/upload")
async def upload_textbook(
    file: UploadFile = File(...),
    title: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    type: str = Form(default="textbook"),
    current_user: dict = Depends(get_current_user)
):
    """Upload a new textbook or slide deck (authenticated users only)."""
    # Validate file type
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".pptx")):
        raise HTTPException(status_code=400, detail="Only PDF and PPTX files are allowed")

    # Validate type
    if type not in ["textbook", "slides", "notes"]:
        raise HTTPException(status_code=400, detail="Type must be 'textbook', 'slides', or 'notes'")

    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Ingest the file
        print(f"\n🚀 Starting ingestion for: {title}...")
        book_id = ingest_textbook(
            file_path=file_path,
            title=title,
            grade=grade,
            subject=subject,
            user_id=current_user["id"],
            type=type
        )
        
        return {
            "message": "File successfully uploaded and added to the library!",
            "book_id": book_id,
            "title": title,
            "type": type
        }
        
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# ==========================================
# ASK THE TUTOR ENDPOINT (Authenticated)
# ==========================================

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, current_user: dict = Depends(get_current_user)):
    """Ask the AI tutor a question (authenticated users only)."""
    try:
        print(f"\n📩 New question from user {current_user['email']}:")
        print(f"   Question: {request.question}")
        print(f"   Language: {request.language}")
        print(f"   Book ID: {request.book_id}")
        
        answer, source = hybrid_tutor(
            question=request.question,
            target_language=request.language,
            book_id=request.book_id,
            chapter=request.chapter
        )
        
        return QuestionResponse(
            answer=answer,
            source=source
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# --- Quiz & Flashcard Models ---
class StudyMaterialRequest(BaseModel):
    topic: str = ""
    language: str = "English"
    book_id: int = None
    type: str = "quiz"  # "quiz" or "flashcard"
    num_items: int = 5

@app.post("/generate-study-material")
def generate_study_material(req: StudyMaterialRequest, current_user: dict = Depends(get_current_user)):
    # 1. Get Context from ChromaDB (similar to /ask endpoint)
    context = ""
    if req.book_id:
        # Find the book in DB to get its Chroma collection name/path
        # (Assuming you have a db helper like get_book_by_id)
        book = db.get_book(req.book_id) 
        if book:
            # Load the vector store for this book
            vectorstore = Chroma(persist_directory=f"chroma_db/{book['id']}", embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(req.topic if req.topic else "main concepts")
            context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Setup the LLM
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # 3. Create the Prompt
    if req.type == "quiz":
        prompt = f"""You are an expert teacher. Generate {req.num_items} multiple-choice questions about "{req.topic}" based on the context below. 
        Translate everything into {req.language}.
        
        Context: {context}
        
        Output ONLY a valid JSON array of objects. Each object must have:
        - "question": string
        - "options": array of 4 strings
        - "correct_answer": integer (0, 1, 2, or 3 representing the index of the correct option)
        - "explanation": string
        
        Do NOT wrap the output in ```json ... ``` markdown blocks. Just output the raw JSON array."""
    else: # flashcard
        prompt = f"""You are an expert teacher. Generate {req.num_items} flashcards about "{req.topic}" based on the context below.
        Translate everything into {req.language}.
        
        Context: {context}
        
        Output ONLY a valid JSON array of objects. Each object must have:
        - "term": string
        - "definition": string
        
        Do NOT wrap the output in ```json ... ``` markdown blocks. Just output the raw JSON array."""

    # 4. Call OpenAI
    response = llm.invoke(prompt)
    raw_text = response.content

    # 5. Clean and Parse JSON (LLMs sometimes add markdown blocks anyway)
    raw_text = re.sub(r'^```json\s*', '', raw_text)
    raw_text = re.sub(r'\s*```$', '', raw_text)
    
    try:
        parsed_data = json.loads(raw_text)
        return {"data": parsed_data, "type": req.type}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI failed to generate valid JSON. Please try again.")
# ==========================================
# RUN THE SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting SA Study App API server...")
    print("📖 API Documentation: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)