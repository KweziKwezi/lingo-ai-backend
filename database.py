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
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            name TEXT,
            auth_provider TEXT DEFAULT 'local',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create books table with user_id foreign key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            grade TEXT,
            subject TEXT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            total_chunks INTEGER DEFAULT 0,
            user_id INTEGER,
            type TEXT DEFAULT 'textbook',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
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

# Initialize the database when this module is imported
init_database()