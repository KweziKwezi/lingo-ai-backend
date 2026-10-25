import sqlite3
from typing import List, Dict, Optional
import os
from datetime import datetime

DB_PATH = "./books.db"

def get_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Create the users and books tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create users table
     cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            book_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (book_id) REFERENCES books(id)
        )
    """)
    
    # Add quiz attempts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            questions TEXT NOT NULL,
            user_answers TEXT NOT NULL,
            results TEXT NOT NULL,
            score INTEGER NOT NULL,
            max_score INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# ==========================================
# USER FUNCTIONS
# ==========================================

def create_user(email: str, password_hash: str, name: str, auth_provider: str = "local") -> int:
    """Create a new user and return their ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO users (email, password_hash, name, auth_provider)
            VALUES (?, ?, ?, ?)
        """, (email, password_hash, name, auth_provider))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None  # Email already exists

def get_user_by_email(email: str) -> Optional[Dict]:
    """Get a user by email."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get a user by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None

# ==========================================
# BOOK FUNCTIONS
# ==========================================

def add_book(title: str, grade: str, subject: str, filename: str, file_path: str, 
             total_chunks: int, user_id: int, type: str = "textbook") -> int:
    """Add a new book to the database and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO books (title, grade, subject, filename, file_path, total_chunks, user_id, type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, grade, subject, filename, file_path, total_chunks, user_id, type))
    
    book_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return book_id

def get_all_books(user_id: Optional[int] = None) -> List[Dict]:
    """Get all books, optionally filtered by user."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("SELECT * FROM books WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,))
    else:
        cursor.execute("SELECT * FROM books ORDER BY uploaded_at DESC")
    
    books = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return books

def get_book_by_id(book_id: int) -> Optional[Dict]:
    """Get a specific book by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None

def delete_book(book_id: int, user_id: int) -> bool:
    """Delete a book from the database (only if owned by user)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM books WHERE id = ? AND user_id = ?", (book_id, user_id))
    conn.commit()
    deleted = cursor.rowcount > 0
    
    conn.close()
    return deleted

    # ==========================================
# SESSION MANAGEMENT (For Quiz History)
# ==========================================

def create_session(user_id: int, book_id: int, title: str) -> int:
    """Create a new study session for a document."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (user_id, book_id, title, created_at)
        VALUES (?, ?, ?, datetime('now'))
    """, (user_id, book_id, title))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_user_sessions(user_id: int):
    """Get all study sessions for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.title as book_title, b.subject, b.grade
        FROM sessions s
        JOIN books b ON s.book_id = b.id
        WHERE s.user_id = ?
        ORDER BY s.created_at DESC
    """, (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return [dict(row) for row in sessions]

def save_quiz_attempt(session_id: int, questions: str, answers: str, results: str, score: int, max_score: int):
    """Save a quiz attempt to history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO quiz_attempts (session_id, questions, user_answers, results, score, max_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (session_id, questions, answers, results, score, max_score))
    conn.commit()
    conn.close()

def get_session_quiz_history(session_id: int):
    """Get all quiz attempts for a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM quiz_attempts
        WHERE session_id = ?
        ORDER BY created_at DESC
    """, (session_id,))
    attempts = cursor.fetchall()
    conn.close()
    return [dict(row) for row in attempts]

# Initialize the database when this module is imported
init_database()