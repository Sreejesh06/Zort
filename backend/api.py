from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import docx2txt
import fitz  # PyMuPDF
import spacy
import re
import os
from typing import List, Dict, Set
from collections import Counter
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (optional)
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Initialize FastAPI app
app = FastAPI(title="Resume Analysis API", version="1.0.0")

# Configure Gemini API with environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBZhaS_ylNb7cdbYjkD1S9hxN9vDXJvIdo")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize Gemini model
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        GEMINI_AVAILABLE = True
        print("✅ Gemini API initialized successfully")
    except Exception as e:
        print(f"❌ Gemini API initialization failed: {e}")
        GEMINI_AVAILABLE = False
else:
    GEMINI_AVAILABLE = False
    print("❌ Gemini API key not found")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Resume Analysis API is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), file_type: str = Form(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Parse file
    if file_type.lower() == "pdf":
        text = extract_text_from_pdf(file_location)
    elif file_type.lower() == "docx":
        text = extract_text_from_docx(file_location)
    else:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    # Normalize and extract sections
    sections = extract_sections(text)
    entities = extract_entities(text)

    return {
        "filename": file.filename,
        "raw_text": text,
        "sections": sections,
        "entities": entities
    }

@app.post("/analyze/")
async def analyze_resume_vs_jd(resume: UploadFile = File(...), jd: UploadFile = File(...)):
    try:
        # Save files
        resume_path = os.path.join(UPLOAD_DIR, resume.filename)
        jd_path = os.path.join(UPLOAD_DIR, jd.filename)
        
        with open(resume_path, "wb") as f:
            f.write(await resume.read())
        with open(jd_path, "wb") as f:
            f.write(await jd.read())
        
        # Parse files with enhanced extraction
        resume_text = extract_text_from_file(resume_path)
        jd_text = extract_text_from_file(jd_path)
        
        # Extract skills using enhanced method
        resume_skills = extract_skills_from_text(resume_text)
        jd_skills = extract_skills_from_text(jd_text)
        
        # Calculate context-aware scoring (enhanced)
        context_analysis = calculate_context_aware_score(resume_skills, jd_skills, resume_text, jd_text)
        
        # Calculate semantic similarity
        semantic_score = calculate_semantic_similarity(resume_text, jd_text)
        
        # Extract additional sections for context
        resume_sections = extract_sections(resume_text)
        jd_info = extract_jd_info(jd_text)
        
        # Use the enhanced final score
        final_score = int(context_analysis['final_score'])
        
        # Determine verdict with more nuanced thresholds
        if final_score >= 85:
            verdict = "Excellent Match"
        elif final_score >= 75:
            verdict = "Strong Match"
        elif final_score >= 65:
            verdict = "Good Match"
        elif final_score >= 55:
            verdict = "Moderate Match"
        elif final_score >= 45:
            verdict = "Fair Match"
        else:
            verdict = "Limited Match"
        
        # Generate enhanced feedback
        feedback = generate_enhanced_feedback(context_analysis, final_score, resume_text, jd_text)
        
        return {
            "resume": resume.filename,
            "jd": jd.filename,
            "score": final_score,
            "verdict": verdict,
            "feedback": feedback,
            "enhanced_analysis": {
                "final_score": context_analysis['final_score'],
                "score_breakdown": context_analysis['breakdown'],
                "weights": context_analysis['weights'],
                "transferable_insights": context_analysis['transferable_insights'],
                "project_insights": context_analysis['project_insights']
            },
            "keyword_score": context_analysis['breakdown']['basic_keyword_score'],
            "semantic_score": semantic_score,
            "detailed_analysis": context_analysis['detailed_analysis'],
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "resume_sections": resume_sections,
            "jd_info": jd_info,
            "resume_text": resume_text,
            "jd_text": jd_text
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/feedback/")
async def get_ai_feedback(resume_text: str = Form(...), jd_text: str = Form(...), 
                         question: str = Form(...)):
    """Get AI-powered feedback for specific questions"""
    try:
        if not GEMINI_AVAILABLE:
            return JSONResponse({"error": "Gemini API not available"}, status_code=503)
        
        # Create a focused prompt for the specific question
        prompt = f"""
You are an expert career coach and resume analyst. Answer this specific question about the resume and job description.

QUESTION: {question}

JOB DESCRIPTION:
{jd_text[:1500]}...

CANDIDATE RESUME:
{resume_text[:1500]}...

Please provide:
1. A direct answer to the question
2. Specific examples from the resume/JD
3. Actionable advice if applicable

Keep your response concise but helpful (2-3 paragraphs max).
"""
        
        response = gemini_model.generate_content(prompt)
        feedback_text = response.text.strip()
        
        return {
            "question": question,
            "answer": feedback_text,
            "timestamp": "2025-09-20T13:17:00Z"
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Parse file
        if file_type.lower() == "pdf":
            text = extract_text_from_pdf(file_location)
        elif file_type.lower() == "docx":
            text = extract_text_from_docx(file_location)
        else:
            return JSONResponse({"error": "Unsupported file type"}, status_code=400)
        
        jd_info = extract_jd_info(text)
        return {"filename": file.filename, "raw_text": text, "jd_info": jd_info}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def extract_text_from_pdf(path):
    """Enhanced PDF text extraction with OCR fallback"""
    text = ""
    
    # Method 1: Try PyMuPDF first
    try:
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
        doc.close()
        
        if text.strip():
            print(f"PyMuPDF extracted {len(text)} characters")
            return text
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
    
    # Method 2: Try pdfplumber
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            print(f"pdfplumber extracted {len(text)} characters")
            return text
    except Exception as e:
        print(f"pdfplumber failed: {e}")
    
    # Method 3: Try PyPDF2
    try:
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
        
        if text.strip():
            print(f"PyPDF2 extracted {len(text)} characters")
            return text
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
    
    # Method 4: OCR fallback for scanned PDFs
    try:
        print("Trying OCR extraction...")
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Use OCR on the image
            image = Image.open(io.BytesIO(img_data))
            page_text = pytesseract.image_to_string(image, config='--psm 6')
            if page_text.strip():
                text += page_text + "\n"
        
        doc.close()
        if text.strip():
            print(f"OCR extracted {len(text)} characters")
            return text
    except Exception as e:
        print(f"OCR failed: {e}")
    
    print("All extraction methods failed")
    return ""

def extract_text_from_docx(path):
    try:
        return docx2txt.process(path)
    except Exception:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_file(path):
    """Extract text from any supported file type"""
    if path.lower().endswith('.pdf'):
        return extract_text_from_pdf(path)
    elif path.lower().endswith(('.docx', '.doc')):
        return extract_text_from_docx(path)
    elif path.lower().endswith('.txt'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Text file reading failed: {e}")
            return ""
    else:
        print(f"Unsupported file type: {path}")
        return ""

def extract_sections(text):
    # Simple regex-based section extraction
    sections = {}
    patterns = {
        "skills": r"skills[:\s]*([\w\s,.-]+)",
        "experience": r"experience[:\s]*([\w\s,.-]+)",
        "education": r"education[:\s]*([\w\s,.-]+)",
    }
    for key, pat in patterns.items():
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
        else:
            sections[key] = ""
    return sections

def extract_entities(text):
    doc = nlp(text)
    entities = {"skills": [], "degrees": [], "companies": []}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["companies"].append(ent.text)
        elif ent.label_ == "PERSON":
            pass
        elif ent.label_ == "EDUCATION":
            entities["degrees"].append(ent.text)
        elif ent.label_ == "SKILL":
            entities["skills"].append(ent.text)
    
    # Custom regex for degrees/certs
    degree_regex = r"(B\.Tech|M\.Tech|MBA|PhD|Bachelor|Master|Doctor|Diploma)"
    entities["degrees"] += re.findall(degree_regex, text, re.IGNORECASE)
    
    # Custom regex for common skills
    skill_regex = r"Python|Java|C\+\+|SQL|Excel|Machine Learning|Data Science|React|Node\.js|AWS|Docker|Kubernetes"
    entities["skills"] += re.findall(skill_regex, text, re.IGNORECASE)
    
    return entities

def extract_jd_info(text):
    # Simple regex and NER for JD fields
    info = {}
    info["role_title"] = re.search(r"Role[:\s]*([\w\s,-]+)", text, re.IGNORECASE)
    info["role_title"] = info["role_title"].group(1).strip() if info["role_title"] else ""
    info["must_have_skills"] = re.findall(r"Must[- ]have skills[:\s]*([\w\s,.-]+)", text, re.IGNORECASE)
    info["good_to_have_skills"] = re.findall(r"Good[- ]to[- ]have skills[:\s]*([\w\s,.-]+)", text, re.IGNORECASE)
    info["degrees"] = re.findall(r"(B\.Tech|M\.Tech|MBA|PhD|Bachelor|Master|Doctor|Diploma)", text, re.IGNORECASE)
    info["certifications"] = re.findall(r"certification[:\s]*([\w\s,.-]+)", text, re.IGNORECASE)
    info["responsibilities"] = re.findall(r"Responsibilities[:\s]*([\w\s,.-]+)", text, re.IGNORECASE)
    
    # NER for skills
    doc = nlp(text)
    info["skills_ner"] = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    
    return info

# Enhanced skill extraction patterns
# Enhanced skill patterns with synonyms and context
SKILL_PATTERNS = {
    'programming': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'react', 'angular',
        'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel', 'rails'
    ],
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle',
        'sqlite', 'dynamodb', 'neo4j', 'firebase', 'supabase'
    ],
    'cloud_platforms': [
        'aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'microsoft azure',
        'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd'
    ],
    'data_science': [
        'machine learning', 'deep learning', 'artificial intelligence', 'ai', 'ml', 'dl',
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'opencv',
        'data analysis', 'statistics', 'data visualization', 'tableau', 'power bi'
    ],
    'frameworks': [
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel', 'rails',
        'express', 'fastapi', 'next.js', 'nuxt.js', 'svelte', 'ember'
    ],
    'tools': [
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'trello',
        'postman', 'swagger', 'figma', 'sketch', 'adobe', 'photoshop', 'illustrator'
    ]
}

# Skill synonym groups for intelligent matching
SKILL_SYNONYMS = {
    'spreadsheet_tools': ['excel', 'powerbi', 'power bi', 'google sheets', 'sheets', 'spreadsheet', 'spreadsheets'],
    'data_analysis': ['data analysis', 'analytics', 'statistical analysis', 'business intelligence', 'bi'],
    'web_development': ['web development', 'frontend', 'backend', 'full stack', 'web apps', 'web applications'],
    'database_management': ['database', 'databases', 'db', 'data management', 'data storage'],
    'project_management': ['project management', 'agile', 'scrum', 'kanban', 'project coordination'],
    'communication': ['communication', 'presentation', 'reporting', 'documentation', 'technical writing'],
    'problem_solving': ['problem solving', 'troubleshooting', 'debugging', 'analysis', 'critical thinking'],
    'leadership': ['leadership', 'team lead', 'mentoring', 'supervision', 'management'],
    'automation': ['automation', 'scripting', 'workflow', 'process improvement', 'efficiency'],
    'testing': ['testing', 'qa', 'quality assurance', 'validation', 'verification']
}

# Transferable skill mappings
TRANSFERABLE_SKILLS = {
    'ai_ml': ['artificial intelligence', 'machine learning', 'data science', 'analytics', 'automation'],
    'data_analysis': ['statistics', 'analytics', 'reporting', 'visualization', 'insights'],
    'programming': ['problem solving', 'logic', 'automation', 'scripting', 'development'],
    'project_management': ['organization', 'planning', 'coordination', 'leadership'],
    'communication': ['presentation', 'documentation', 'collaboration', 'stakeholder management']
}

def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """Extract skills from text using multiple methods"""
    text_lower = text.lower()
    found_skills = {category: [] for category in SKILL_PATTERNS.keys()}
    
    # Direct pattern matching
    for category, skills in SKILL_PATTERNS.items():
        for skill in skills:
            if skill in text_lower:
                found_skills[category].append(skill)
    
    # Enhanced regex patterns for skills
    skill_patterns = [
        r'(?:proficient in|expert in|skilled in|experience with|knowledge of)\s+([^,.\n]+)',
        r'(?:technologies?|tools?|languages?|frameworks?|platforms?)[:\s]+([^,.\n]+)',
        r'(?:programming languages?|tech stack)[:\s]+([^,.\n]+)',
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by common separators and clean
            skills = re.split(r'[,;|&]', match)
            for skill in skills:
                skill = skill.strip()
                if len(skill) > 2 and len(skill) < 50:  # Reasonable skill length
                    # Try to categorize the skill
                    for category, known_skills in SKILL_PATTERNS.items():
                        if any(known_skill in skill for known_skill in known_skills):
                            if skill not in found_skills[category]:
                                found_skills[category].append(skill)
    
    return found_skills

def calculate_context_aware_score(resume_skills: Dict[str, List[str]], jd_skills: Dict[str, List[str]], 
                                 resume_text: str, jd_text: str) -> Dict:
    """Calculate context-aware scoring with synonyms, transferable skills, and project analysis"""
    
    # 1. Basic keyword matching
    basic_analysis = calculate_keyword_match_score(resume_skills, jd_skills)
    
    # 2. Synonym matching
    synonym_matches = calculate_synonym_matches(resume_text, jd_text)
    
    # 3. Transferable skills analysis
    transferable_score = calculate_transferable_skills_score(resume_text, jd_text)
    
    # 4. Project context analysis
    project_score = analyze_project_context(resume_text, jd_text)
    
    # 5. LLM-based contextual understanding
    contextual_score = get_contextual_understanding_score(resume_text, jd_text)
    
    # Weighted final score
    weights = {
        'basic_keyword': 0.25,      # Reduced from 0.7
        'synonym_match': 0.20,      # New: intelligent synonym matching
        'transferable': 0.20,       # New: transferable skills
        'project_context': 0.15,    # New: project understanding
        'contextual': 0.20          # New: LLM contextual analysis
    }
    
    final_score = (
        weights['basic_keyword'] * basic_analysis['overall_score'] +
        weights['synonym_match'] * synonym_matches +
        weights['transferable'] * transferable_score +
        weights['project_context'] * project_score +
        weights['contextual'] * contextual_score
    )
    
    return {
        'final_score': round(final_score, 2),
        'breakdown': {
            'basic_keyword_score': basic_analysis['overall_score'],
            'synonym_match_score': synonym_matches,
            'transferable_skills_score': transferable_score,
            'project_context_score': project_score,
            'contextual_understanding_score': contextual_score
        },
        'weights': weights,
        'detailed_analysis': basic_analysis,
        'synonym_matches': synonym_matches,
        'transferable_insights': get_transferable_insights(resume_text, jd_text),
        'project_insights': get_project_insights(resume_text, jd_text)
    }

def calculate_synonym_matches(resume_text: str, jd_text: str) -> float:
    """Calculate matches using skill synonyms"""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    total_matches = 0
    total_required = 0
    
    for synonym_group, synonyms in SKILL_SYNONYMS.items():
        jd_has_group = any(synonym in jd_lower for synonym in synonyms)
        resume_has_group = any(synonym in resume_lower for synonym in synonyms)
        
        if jd_has_group:
            total_required += 1
            if resume_has_group:
                total_matches += 1
    
    return (total_matches / max(total_required, 1)) * 100

def calculate_transferable_skills_score(resume_text: str, jd_text: str) -> float:
    """Calculate score based on transferable skills"""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    transferable_matches = 0
    total_transferable = 0
    
    for skill_category, related_skills in TRANSFERABLE_SKILLS.items():
        jd_needs_category = any(skill in jd_lower for skill in related_skills)
        resume_has_category = any(skill in resume_lower for skill in related_skills)
        
        if jd_needs_category:
            total_transferable += 1
            if resume_has_category:
                transferable_matches += 1
    
    return (transferable_matches / max(total_transferable, 1)) * 100

def analyze_project_context(resume_text: str, jd_text: str) -> float:
    """Analyze project context and relevance"""
    # Extract project-related keywords
    project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed', 'managed']
    jd_keywords = ['responsibilities', 'duties', 'requirements', 'experience', 'skills']
    
    resume_projects = sum(1 for keyword in project_keywords if keyword in resume_text.lower())
    jd_requirements = sum(1 for keyword in jd_keywords if keyword in jd_text.lower())
    
    # Simple project relevance scoring
    if resume_projects > 0 and jd_requirements > 0:
        return min(100, (resume_projects / max(jd_requirements, 1)) * 100)
    return 50  # Default score

def get_contextual_understanding_score(resume_text: str, jd_text: str) -> float:
    """Use LLM to understand contextual fit"""
    if not GEMINI_AVAILABLE:
        return 50.0  # Default if LLM not available
    
    try:
        prompt = f"""
        Analyze the contextual fit between this resume and job description. 
        Consider transferable skills, project relevance, and overall potential fit.
        
        Resume: {resume_text[:800]}
        Job Description: {jd_text[:800]}
        
        Rate the contextual fit from 0-100, considering:
        1. Transferable skills (e.g., AI/ML skills for data analysis roles)
        2. Project relevance and complexity
        3. Overall potential and adaptability
        4. Industry experience alignment
        
        Respond with only a number between 0-100.
        """
        
        response = gemini_model.generate_content(prompt)
        score_text = response.text.strip()
        
        # Extract number from response
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = int(numbers[0])
            return min(100, max(0, score))  # Clamp between 0-100
        
        return 50.0
        
    except Exception as e:
        print(f"Contextual understanding error: {e}")
        return 50.0

def get_transferable_insights(resume_text: str, jd_text: str) -> List[str]:
    """Get insights about transferable skills"""
    insights = []
    
    # Check for AI/ML transferability
    if any(skill in resume_text.lower() for skill in ['ai', 'machine learning', 'data science']) and \
       any(skill in jd_text.lower() for skill in ['analytics', 'data', 'automation']):
        insights.append("Your AI/ML experience is highly transferable to data analysis and automation roles")
    
    # Check for programming transferability
    if any(skill in resume_text.lower() for skill in ['python', 'programming', 'development']) and \
       any(skill in jd_text.lower() for skill in ['problem solving', 'automation', 'scripting']):
        insights.append("Programming skills demonstrate strong problem-solving and automation capabilities")
    
    return insights

def get_project_insights(resume_text: str, jd_text: str) -> List[str]:
    """Get insights about project relevance"""
    insights = []
    
    # Extract project mentions
    project_words = ['project', 'developed', 'built', 'created', 'implemented']
    resume_projects = [word for word in project_words if word in resume_text.lower()]
    
    if resume_projects:
        insights.append(f"Resume shows {len(resume_projects)} project-related experiences")
    
    return insights

def calculate_keyword_match_score(resume_skills: Dict[str, List[str]], jd_skills: Dict[str, List[str]]) -> Dict:
    """Calculate detailed keyword matching score"""
    total_matches = 0
    total_required = 0
    category_scores = {}
    matched_skills = {}
    missing_skills = {}
    
    for category in SKILL_PATTERNS.keys():
        resume_cat_skills = set(resume_skills.get(category, []))
        jd_cat_skills = set(jd_skills.get(category, []))
        
        # Exact matches
        exact_matches = resume_cat_skills & jd_cat_skills
        
        # Fuzzy matches (similar skills)
        fuzzy_matches = set()
        for jd_skill in jd_cat_skills:
            for resume_skill in resume_cat_skills:
                # Check for partial matches or similar words
                if (jd_skill in resume_skill or resume_skill in jd_skill or 
                    difflib.SequenceMatcher(None, jd_skill, resume_skill).ratio() > 0.7):
                    fuzzy_matches.add(jd_skill)
        
        all_matches = exact_matches | fuzzy_matches
        missing = jd_cat_skills - all_matches
        
        category_score = len(all_matches) / max(len(jd_cat_skills), 1) * 100
        category_scores[category] = round(category_score, 2)
        matched_skills[category] = list(all_matches)
        missing_skills[category] = list(missing)
        
        total_matches += len(all_matches)
        total_required += len(jd_cat_skills)
    
    overall_score = (total_matches / max(total_required, 1)) * 100
    
    return {
        'overall_score': round(overall_score, 2),
        'category_scores': category_scores,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'total_matches': total_matches,
        'total_required': total_required
    }
    
    # Exact matches
    exact_matches = resume_cat_skills & jd_cat_skills
    
    # Fuzzy matches (similar skills)
    fuzzy_matches = set()
    for jd_skill in jd_cat_skills:
        for resume_skill in resume_cat_skills:
            # Check for partial matches or similar words
            if (jd_skill in resume_skill or resume_skill in jd_skill or 
                difflib.SequenceMatcher(None, jd_skill, resume_skill).ratio() > 0.7):
                fuzzy_matches.add(jd_skill)
    
    all_matches = exact_matches | fuzzy_matches
    missing = jd_cat_skills - all_matches
    
    category_score = len(all_matches) / max(len(jd_cat_skills), 1) * 100
    category_scores[category] = round(category_score, 2)
    matched_skills[category] = list(all_matches)
    missing_skills[category] = list(missing)
    
    total_matches += len(all_matches)
    total_required += len(jd_cat_skills)
    
    overall_score = (total_matches / max(total_required, 1)) * 100
    
    return {
        'overall_score': round(overall_score, 2),
        'category_scores': category_scores,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'total_matches': total_matches,
        'total_required': total_required
    }

def calculate_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """Calculate semantic similarity using TF-IDF and cosine similarity"""
    try:
        # Clean and preprocess texts
        resume_clean = re.sub(r'[^\w\s]', ' ', resume_text.lower())
        jd_clean = re.sub(r'[^\w\s]', ' ', jd_text.lower())
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(similarity * 100, 2)
    except Exception as e:
        print(f"Error in semantic similarity: {e}")
        return 50.0  # Default score if calculation fails

def generate_enhanced_feedback(context_analysis: Dict, final_score: int, resume_text: str, jd_text: str) -> List[str]:
    """Generate enhanced feedback that addresses the limitations mentioned"""
    feedback = []
    
    # Overall assessment with context
    breakdown = context_analysis['breakdown']
    
    if final_score >= 85:
        feedback.append("🎉 Excellent match! Your skills and experience align very well with this role.")
    elif final_score >= 75:
        feedback.append("✅ Strong match! You have most required skills and good transferable experience.")
    elif final_score >= 65:
        feedback.append("👍 Good match! You have solid foundational skills with room for growth.")
    elif final_score >= 55:
        feedback.append("⚠️ Moderate match. Consider highlighting transferable skills and relevant projects.")
    elif final_score >= 45:
        feedback.append("📋 Fair match. Focus on transferable skills and project relevance.")
    else:
        feedback.append("🔍 Limited match. Consider if this role aligns with your career goals.")
    
    # Address the limitations mentioned
    feedback.append("\n🧠 Intelligent Analysis (Beyond Basic Keywords):")
    
    # Transferable skills insights
    if context_analysis['transferable_insights']:
        feedback.append("💡 Transferable Skills:")
        for insight in context_analysis['transferable_insights']:
            feedback.append(f"  • {insight}")
    
    # Project context insights
    if context_analysis['project_insights']:
        feedback.append("🏗️ Project Experience:")
        for insight in context_analysis['project_insights']:
            feedback.append(f"  • {insight}")
    
    # Score breakdown explanation
    feedback.append(f"\n📊 Scoring Breakdown:")
    feedback.append(f"  • Basic Keywords: {breakdown['basic_keyword_score']:.1f}% (traditional ATS matching)")
    feedback.append(f"  • Smart Synonyms: {breakdown['synonym_match_score']:.1f}% (Excel ↔ PowerBI recognition)")
    feedback.append(f"  • Transferable Skills: {breakdown['transferable_skills_score']:.1f}% (AI/ML → Analytics)")
    feedback.append(f"  • Project Context: {breakdown['project_context_score']:.1f}% (experience relevance)")
    feedback.append(f"  • Contextual Understanding: {breakdown['contextual_understanding_score']:.1f}% (AI analysis)")
    
    # Specific recommendations
    feedback.append(f"\n🎯 Recommendations:")
    
    if breakdown['synonym_match_score'] < 70:
        feedback.append("  • Highlight equivalent tools (e.g., if JD mentions Excel, mention your PowerBI experience)")
    
    if breakdown['transferable_skills_score'] < 70:
        feedback.append("  • Emphasize how your skills apply to this role (e.g., AI experience for data analysis)")
    
    if breakdown['project_context_score'] < 70:
        feedback.append("  • Add more project details showing relevant experience and impact")
    
    if breakdown['contextual_understanding_score'] < 70:
        feedback.append("  • Consider how your background brings unique value to this role")
    
    # Final note about recruiter perspective
    feedback.append(f"\n💼 Recruiter Perspective:")
    feedback.append("  • This analysis considers context, not just keywords")
    feedback.append("  • Transferable skills and project relevance matter more than exact matches")
    feedback.append("  • Your potential and adaptability are key factors")
    
    return feedback
    """Generate advanced LLM-powered feedback using Gemini"""
    if not GEMINI_AVAILABLE:
        return generate_detailed_feedback(resume_skills, jd_skills, keyword_analysis, final_score)
    
    try:
        # Prepare context for the LLM
        missing_skills_summary = []
        for category, missing in keyword_analysis['missing_skills'].items():
            if missing:
                missing_skills_summary.append(f"{category.replace('_', ' ').title()}: {', '.join(missing[:3])}")
        
        matched_skills_summary = []
        for category, matched in keyword_analysis['matched_skills'].items():
            if matched:
                matched_skills_summary.append(f"{category.replace('_', ' ').title()}: {', '.join(matched[:3])}")
        
        prompt = f"""
You are an expert career coach and resume analyst. Analyze this resume against the job description and provide detailed, actionable feedback.

JOB DESCRIPTION:
{jd_text[:1000]}...

CANDIDATE RESUME:
{resume_text[:1000]}...

ANALYSIS RESULTS:
- Overall Match Score: {final_score}/100
- Keyword Match Score: {keyword_analysis['overall_score']}/100
- Skill Match Rate: {keyword_analysis['total_matches']}/{keyword_analysis['total_required']}

MATCHED SKILLS:
{chr(10).join(matched_skills_summary) if matched_skills_summary else "None"}

MISSING SKILLS:
{chr(10).join(missing_skills_summary) if missing_skills_summary else "None"}

Please provide:
1. Overall assessment (1-2 sentences)
2. Top 3 specific skills to add/improve
3. Top 3 actionable recommendations
4. Career advice for this role

Format as a JSON object with keys: "overall_assessment", "missing_skills", "recommendations", "career_advice"
Each value should be an array of strings.
"""
        
        response = gemini_model.generate_content(prompt)
        
        # Parse the response
        feedback_text = response.text
        
        # Try to extract JSON from the response
        try:
            import json
            # Look for JSON in the response
            json_start = feedback_text.find('{')
            json_end = feedback_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = feedback_text[json_start:json_end]
                feedback_data = json.loads(json_str)
                
                feedback = []
                feedback.append(f"🤖 AI Career Coach Assessment:")
                feedback.append(feedback_data.get('overall_assessment', [''])[0])
                
                if feedback_data.get('missing_skills'):
                    feedback.append("📋 Skills to Add:")
                    for skill in feedback_data['missing_skills'][:3]:
                        feedback.append(f"  • {skill}")
                
                if feedback_data.get('recommendations'):
                    feedback.append("💡 Recommendations:")
                    for rec in feedback_data['recommendations'][:3]:
                        feedback.append(f"  • {rec}")
                
                if feedback_data.get('career_advice'):
                    feedback.append("🎯 Career Advice:")
                    for advice in feedback_data['career_advice'][:2]:
                        feedback.append(f"  • {advice}")
                
                return feedback
        except:
            pass
        
        # Fallback: parse as plain text
        lines = feedback_text.split('\n')
        feedback = ["🤖 AI Career Coach Assessment:"]
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                feedback.append(line)
        
        return feedback[:8]  # Limit to 8 items
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return generate_detailed_feedback(resume_skills, jd_skills, keyword_analysis, final_score)

def generate_detailed_feedback(resume_skills: Dict[str, List[str]], jd_skills: Dict[str, List[str]], 
                               keyword_analysis: Dict, final_score: int) -> List[str]:
    """Generate detailed, actionable feedback based on real analysis"""
    feedback = []
    
    # Overall assessment
    if final_score >= 85:
        feedback.append("🎉 Excellent match! Your skills align very well with the job requirements.")
    elif final_score >= 70:
        feedback.append("✅ Good match! You have most of the required skills.")
    elif final_score >= 55:
        feedback.append("⚠️ Moderate match. Consider improving some key areas.")
    elif final_score >= 40:
        feedback.append("❌ Poor match. Significant skill gaps need to be addressed.")
    else:
        feedback.append("🚫 Very poor match. Consider if this role is suitable.")
    
    # Category-specific feedback
    for category, score in keyword_analysis['category_scores'].items():
        if score < 50 and keyword_analysis['missing_skills'][category]:
            missing = keyword_analysis['missing_skills'][category]
            category_name = category.replace('_', ' ').title()
            feedback.append(f"📋 {category_name}: Missing {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
    
    # Positive feedback for matched skills
    total_matched = sum(len(skills) for skills in keyword_analysis['matched_skills'].values())
    if total_matched > 0:
        feedback.append(f"✨ You have {total_matched} matching skills across different categories.")
    
    # Specific recommendations
    if keyword_analysis['total_required'] > 0:
        match_percentage = (keyword_analysis['total_matches'] / keyword_analysis['total_required']) * 100
        feedback.append(f"📊 Skill match rate: {match_percentage:.1f}% ({keyword_analysis['total_matches']}/{keyword_analysis['total_required']})")
    
    # Actionable suggestions
    suggestions = []
    for category, missing in keyword_analysis['missing_skills'].items():
        if missing:
            category_name = category.replace('_', ' ').title()
            suggestions.append(f"• Consider learning {missing[0]} for {category_name}")
    
    if suggestions:
        feedback.append("💡 Recommendations:")
        feedback.extend(suggestions[:3])  # Limit to top 3 suggestions
    
    return feedback

def generate_feedback(resume_sections, jd_info, hard_feedback, final_score):
    # Legacy function for backward compatibility
    feedback = []
    if hard_feedback.get("missing"):
        feedback.append(f"Add missing skills: {', '.join(hard_feedback['missing'])}")
    if final_score < 80:
        feedback.append("Consider adding more relevant experience or certifications.")
    if not resume_sections.get("education"):
        feedback.append("Add education details.")
    return "\n".join(feedback)