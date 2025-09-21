# Security Setup Guide

This guide explains how to properly configure API keys and environment variables for the Resume Analysis System.

## 🔐 API Key Security

### 1. Environment Variables Setup

**NEVER** commit API keys to version control. Always use environment variables.

#### Step 1: Copy the example file
```bash
cp .env.example .env
```

#### Step 2: Edit your .env file
```env
# API Keys
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///./resume_analyzer.db

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### 2. Getting Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Paste it in your `.env` file

### 3. Security Best Practices

#### ✅ DO:
- Store API keys in `.env` files
- Add `.env` to `.gitignore`
- Use different keys for development and production
- Rotate API keys regularly
- Use environment-specific `.env` files

#### ❌ DON'T:
- Hardcode API keys in source code
- Commit `.env` files to version control
- Share API keys in chat/email
- Use production keys in development
- Log API keys in console output

### 4. Environment File Structure

```
Backend/
├── .env                 # Your actual keys (DO NOT COMMIT)
├── .env.example         # Template file (safe to commit)
├── .env.local          # Local overrides (DO NOT COMMIT)
├── .env.production     # Production keys (DO NOT COMMIT)
└── .gitignore          # Ensures .env files are ignored
```

### 5. Production Deployment

For production, set environment variables directly on your server:

```bash
export GEMINI_API_KEY="your_production_key"
export DATABASE_URL="postgresql://user:pass@host:port/db"
```

### 6. Testing Your Setup

```bash
# Test if environment variables are loaded
python -c "import os; print('GEMINI_API_KEY loaded:', bool(os.getenv('GEMINI_API_KEY')))"
```

### 7. Troubleshooting

#### Issue: "GEMINI_API_KEY not found"
- Check if `.env` file exists
- Verify the key name is exactly `GEMINI_API_KEY`
- Ensure no extra spaces or quotes around the key

#### Issue: "Invalid API key"
- Verify the key is correct
- Check if the key has proper permissions
- Ensure the key is not expired

### 8. Code Security

The application is designed to:
- Load API keys only from environment variables
- Never expose keys in error messages
- Gracefully handle missing keys
- Use secure defaults when keys are unavailable

## 🛡️ Additional Security Measures

1. **Database Security**: Use strong passwords and encrypted connections
2. **File Uploads**: Validate file types and sizes
3. **CORS**: Configure appropriate origins for production
4. **HTTPS**: Always use HTTPS in production
5. **Monitoring**: Set up logging and monitoring for security events

## 📝 Quick Start

1. Copy `.env.example` to `.env`
2. Add your Gemini API key to `.env`
3. Run the application
4. Verify it works without exposing your key

Your API keys are now secure! 🔒
