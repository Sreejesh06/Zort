# 🚀 Automated Resume Relevance Check System

A full-stack application that analyzes resume relevance against job descriptions using AI-powered context-aware scoring.

## 📁 Project Structure

```
Zort/
├── Backend/                    # FastAPI Backend
│   ├── api.py                 # Main API endpoints
│   ├── main.py                # FastAPI app entry point
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables (API keys)
│   ├── .env.example           # Environment variables template
│   ├── .gitignore             # Backend gitignore
│   ├── venv/                  # Python virtual environment
│   └── uploads/               # File upload directory
├── Frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styling
│   │   └── index.js           # React entry point
│   ├── public/                # Static assets
│   ├── package.json           # Node.js dependencies
│   ├── .env                   # Frontend environment variables
│   ├── .env.example           # Frontend env template
│   └── .gitignore             # Frontend gitignore
├── JD/                        # Sample Job Descriptions
├── Resumes/                   # Sample Resumes
├── .gitignore                 # Root gitignore
└── README.md                  # This file
```

## 🚀 Quick Start

### Backend Setup
```bash
cd Backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd Frontend
npm install
npm start
```

## 🔧 Environment Variables

### Backend (.env)
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Frontend (.env)
```env
REACT_APP_API_URL=http://localhost:8000
```

## ✨ Features

- **Context-Aware Analysis**: Beyond simple keyword matching
- **Skill Synonym Recognition**: Understands related skills
- **Transferable Skills**: Identifies adaptable skills across domains
- **Project Context Analysis**: Analyzes project relevance and impact
- **AI-Powered Feedback**: Gemini-powered career coaching
- **Real-time Processing**: Instant analysis and feedback
- **Modern UI**: Clean, responsive design

## 🛠️ API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /analyze/` - Analyze resume against JD
- `POST /feedback/` - Get AI-powered feedback

## 📱 Usage

1. Open `http://localhost:3000` in your browser
2. Upload a resume (PDF/DOCX/TXT)
3. Upload a job description
4. Get detailed analysis with context-aware scoring
5. Ask the AI Career Coach for personalized feedback

## 🔒 Security

- API keys stored in `.env` files (not committed to git)
- `.gitignore` configured to exclude sensitive files
- Environment variable templates provided

## 📝 License

MIT License - feel free to use and modify!