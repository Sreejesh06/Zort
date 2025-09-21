from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration with better error handling
def get_database_url():
    """Get database URL with fallback options"""
    # Try to get from environment variables
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        return database_url
    
    # Build from individual components
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "password")
    db_name = os.getenv("DB_NAME", "resume_analyzer")
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

DATABASE_URL = get_database_url()

# Create engine with connection pooling and better error handling
try:
    engine = create_engine(
        DATABASE_URL, 
        echo=True,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    print(f"✅ Database engine created: {DATABASE_URL}")
except Exception as e:
    print(f"⚠️ PostgreSQL connection failed: {e}")
    print("🔄 Falling back to SQLite...")
    # Fallback to SQLite
    DATABASE_URL = "sqlite:///./resume_analyzer.db"
    engine = create_engine(DATABASE_URL, echo=True)
    print(f"✅ Fallback to SQLite: {DATABASE_URL}")

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# Database models
class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_type = Column(String(10), nullable=False)  # pdf, docx, txt
    file_size = Column(Integer)  # in bytes
    
    # Extracted information
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    skills = Column(JSON)  # List of skills
    experience_years = Column(Float)
    education = Column(Text)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="resume")

class JobDescription(Base):
    __tablename__ = "job_descriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255))
    content = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String(500))
    file_type = Column(String(10))  # pdf, docx, txt
    
    # Extracted information
    required_skills = Column(JSON)  # List of required skills
    experience_required = Column(Float)
    location = Column(String(255))
    salary_range = Column(String(100))
    
    # Relationships
    analyses = relationship("Analysis", back_populates="job_description")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=False)
    
    # Analysis results
    overall_score = Column(Float, nullable=False)
    keyword_score = Column(Float)
    semantic_score = Column(Float)
    context_aware_score = Column(Float)
    
    # Detailed breakdown
    matched_skills = Column(JSON)  # Skills that matched
    missing_skills = Column(JSON)  # Skills that are missing
    skill_scores = Column(JSON)  # Detailed skill category scores
    
    # AI feedback
    ai_feedback = Column(Text)
    recommendations = Column(JSON)  # List of recommendations
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    analysis_duration = Column(Float)  # Time taken for analysis in seconds
    
    # Relationships
    resume = relationship("Resume", back_populates="analyses")
    job_description = relationship("JobDescription", back_populates="analyses")

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    question = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    user_rating = Column(Integer)  # 1-5 rating
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("Analysis")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

# Test database connection
def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Initialize database
def init_database():
    """Initialize database with tables"""
    print("🚀 Initializing database...")
    if test_connection():
        create_tables()
        print("✅ Database initialization completed")
        return True
    else:
        print("❌ Database initialization failed")
        return False
