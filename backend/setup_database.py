#!/usr/bin/env python3
"""
Database setup script for Resume Analysis System
This script helps you set up PostgreSQL database for the application.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the database if it doesn't exist"""
    # Get database configuration
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "password")
    db_name = os.getenv("DB_NAME", "resume_analyzer")
    
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
    exists = cursor.fetchone()
    
    if not exists:
        # Create database
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"✅ Database '{db_name}' created successfully")
    else:
        print(f"ℹ️ Database '{db_name}' already exists")
    
    cursor.close()
    conn.close()

def test_connection():
    """Test database connection"""
    try:
        from database import engine
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def main():
    print("🚀 Setting up PostgreSQL database for Resume Analysis System")
    print("=" * 60)
    
    # Step 1: Create database
    print("\n1. Creating database...")
    create_database()
    
    # Step 2: Test connection
    print("\n2. Testing connection...")
    if test_connection():
        print("\n✅ Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Make sure PostgreSQL is running")
        print("2. Update your .env file with correct database credentials")
        print("3. Run the application: uvicorn api:app --reload")
    else:
        print("\n❌ Database setup failed. Please check your PostgreSQL configuration.")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is installed and running")
        print("2. Check your database credentials in .env file")
        print("3. Ensure the database user has proper permissions")

if __name__ == "__main__":
    main()
