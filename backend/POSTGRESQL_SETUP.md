# PostgreSQL Setup Guide

This guide will help you set up PostgreSQL for the Resume Analysis System.

## Prerequisites

1. **Install PostgreSQL** on your system:
   - Windows: Download from https://www.postgresql.org/download/windows/
   - macOS: `brew install postgresql`
   - Linux: `sudo apt-get install postgresql postgresql-contrib`

2. **Start PostgreSQL service**:
   - Windows: Start PostgreSQL service from Services
   - macOS: `brew services start postgresql`
   - Linux: `sudo systemctl start postgresql`

## Database Setup

### Step 1: Create Database and User

1. **Connect to PostgreSQL**:
   ```bash
   psql -U postgres
   ```

2. **Create database**:
   ```sql
   CREATE DATABASE resume_analyzer;
   ```

3. **Create user** (optional, you can use postgres user):
   ```sql
   CREATE USER resume_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE resume_analyzer TO resume_user;
   ```

4. **Exit psql**:
   ```sql
   \q
   ```

### Step 2: Update Environment Variables

Update your `.env` file with the correct database credentials:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/resume_analyzer

# Alternative format:
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=resume_analyzer
```

### Step 3: Test Database Connection

Run the setup script to test the connection:

```bash
python setup_database.py
```

### Step 4: Start the Application

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Database Schema

The application will automatically create the following tables:

- **resumes**: Stores uploaded resume files and extracted information
- **job_descriptions**: Stores job description files and extracted requirements
- **analyses**: Stores analysis results and scores
- **user_feedback**: Stores user feedback and AI responses

## New API Endpoints

With PostgreSQL integration, you now have these additional endpoints:

- `GET /resumes/` - Get all resumes
- `GET /job-descriptions/` - Get all job descriptions
- `GET /analyses/` - Get all analyses
- `GET /analyses/{id}` - Get specific analysis by ID

## Troubleshooting

### Common Issues:

1. **Connection refused**:
   - Make sure PostgreSQL is running
   - Check if the port (5432) is correct

2. **Authentication failed**:
   - Verify username and password
   - Check if the user has proper permissions

3. **Database does not exist**:
   - Create the database manually or run the setup script

4. **Permission denied**:
   - Grant proper permissions to the database user

### Testing Connection:

```python
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        user="postgres",
        password="your_password",
        database="resume_analyzer"
    )
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## Production Considerations

For production deployment:

1. **Use environment variables** for sensitive data
2. **Set up proper database backups**
3. **Configure connection pooling**
4. **Use SSL connections** for security
5. **Set up monitoring** for database performance

## Next Steps

1. Set up PostgreSQL following this guide
2. Update your `.env` file with correct credentials
3. Run the application and test the new database features
4. Use the new API endpoints to retrieve stored data
