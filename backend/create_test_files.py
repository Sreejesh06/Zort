# Create test files for demonstration
resume_content = """
SURYA KUMAR
Software Engineer

SKILLS:
- Python programming
- JavaScript
- React.js
- Node.js
- SQL databases
- Git version control
- AWS cloud services
- Machine Learning with scikit-learn

EXPERIENCE:
- 2 years as Full Stack Developer
- Built web applications using React and Node.js
- Worked with Python for data analysis
- Experience with AWS deployment

EDUCATION:
- Bachelor of Technology in Computer Science
"""

jd_content = """
Software Engineer Position

REQUIRED SKILLS:
- Python programming
- JavaScript
- React.js
- AWS cloud services
- SQL databases
- Git

RESPONSIBILITIES:
- Develop web applications
- Work with cloud platforms
- Database management
- Version control

QUALIFICATIONS:
- Bachelor's degree in Computer Science
- 2+ years experience
"""

# Save as text files for testing
with open('uploads/test_resume.txt', 'w') as f:
    f.write(resume_content)

with open('uploads/test_jd.txt', 'w') as f:
    f.write(jd_content)

print("Test files created successfully!")
