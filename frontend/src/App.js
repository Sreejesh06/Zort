
import React, { useState } from 'react';
import './App.css';

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdFile, setJdFile] = useState(null);
  const [results, setResults] = useState([]);
  const [filter, setFilter] = useState('');
  const [feedbackQuestion, setFeedbackQuestion] = useState('');
  const [aiFeedback, setAiFeedback] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  // Handlers for file input
  const handleResumeChange = (e) => setResumeFile(e.target.files[0]);
  const handleJdChange = (e) => setJdFile(e.target.files[0]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Connect to backend API
  const handleUpload = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults([]);
    setAiFeedback(null);
    if (!resumeFile || !jdFile) {
      setError('Please select both resume and job description files.');
      setLoading(false);
      return;
    }
    try {
      const formData = new FormData();
      formData.append('resume', resumeFile);
      formData.append('jd', jdFile);
      const response = await fetch('http://localhost:8000/analyze/', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error('Failed to analyze files.');
      const data = await response.json();
      setResults([
        {
          candidate: data.resume,
          score: data.score,
          feedback: data.feedback,
          verdict: data.verdict,
          keyword_score: data.keyword_score,
          semantic_score: data.semantic_score,
          jd: data.jd,
          detailed_analysis: data.detailed_analysis,
          resume_text: data.resume_text || '',
          jd_text: data.jd_text || ''
        }
      ]);
    } catch (err) {
      setError(err.message || 'Upload failed.');
    }
    setLoading(false);
  };

  // Handle AI feedback request
  const handleFeedbackRequest = async (e) => {
    e.preventDefault();
    if (!feedbackQuestion.trim()) {
      setError('Please enter a question.');
      return;
    }
    if (results.length === 0) {
      setError('Please analyze files first.');
      return;
    }

    setFeedbackLoading(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('resume_text', results[0].resume_text || 'Resume text not available');
      formData.append('jd_text', results[0].jd_text || 'Job description text not available');
      formData.append('question', feedbackQuestion);
      
      const response = await fetch('http://localhost:8000/feedback/', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Failed to get AI feedback.');
      const data = await response.json();
      setAiFeedback(data);
      setFeedbackQuestion('');
    } catch (err) {
      setError(err.message || 'Failed to get AI feedback.');
    }
    setFeedbackLoading(false);
  };

  return (
    <div className="dashboard">
      <h1>🤖 Zort Resume & JD Analyzer with AI Feedback</h1>
      
      {/* File Upload Section */}
      <form className="upload-form" onSubmit={handleUpload}>
        <div>
          <label>📄 Upload Resume (PDF/DOCX): </label>
          <input type="file" accept=".pdf,.docx,.txt" onChange={handleResumeChange} />
        </div>
        <div>
          <label>📋 Upload Job Description (PDF/DOCX): </label>
          <input type="file" accept=".pdf,.docx,.txt" onChange={handleJdChange} />
        </div>
        <button type="submit">🚀 Upload & Analyze</button>
      </form>
      
      {error && <div style={{color:'#d32f2f', marginTop:'12px', textAlign:'center'}}>{error}</div>}
      {loading && <div className="loader">🔄 Analyzing...</div>}

      {/* Results Section */}
      {results.length > 0 && (
        <div className="results-section">
          <h2>📊 Analysis Results</h2>
          <div className="result-card">
            <h3>{results[0].candidate}</h3>
            <div className="score-display">
              <div className="score-item">
                <span className="score-label">Overall Score:</span>
                <span className={`score-value ${results[0].score >= 80 ? 'excellent' : results[0].score >= 60 ? 'good' : 'poor'}`}>
                  {results[0].score}/100
                </span>
              </div>
              <div className="score-item">
                <span className="score-label">Keyword Match:</span>
                <span className="score-value">{results[0].keyword_score}/100</span>
              </div>
              <div className="score-item">
                <span className="score-label">Semantic Match:</span>
                <span className="score-value">{results[0].semantic_score}/100</span>
              </div>
              <div className="score-item">
                <span className="score-label">Verdict:</span>
                <span className={`verdict ${results[0].verdict.toLowerCase().replace(' ', '-')}`}>
                  {results[0].verdict}
                </span>
              </div>
            </div>
            
            <div className="feedback-section">
              <h4>📝 Analysis Feedback:</h4>
              <div className="feedback-text">
                {Array.isArray(results[0].feedback) ? 
                  results[0].feedback.map((item, idx) => <div key={idx}>{item}</div>) :
                  results[0].feedback
                }
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Feedback Section */}
      {results.length > 0 && (
        <div className="ai-feedback-section">
          <h2>🤖 Ask AI Career Coach</h2>
          <p>Get personalized feedback and advice about your resume and job match!</p>
          
          <form onSubmit={handleFeedbackRequest} className="feedback-form">
            <div className="question-input">
              <label>💬 Ask a question:</label>
              <textarea
                value={feedbackQuestion}
                onChange={(e) => setFeedbackQuestion(e.target.value)}
                placeholder="e.g., What skills should I highlight more? How can I improve my resume for this role? What certifications would help?"
                rows="3"
              />
            </div>
            <button type="submit" disabled={feedbackLoading}>
              {feedbackLoading ? '🔄 Getting AI feedback...' : '🚀 Get AI Feedback'}
            </button>
          </form>

          {aiFeedback && (
            <div className="ai-response">
              <h3>🤖 AI Career Coach Response:</h3>
              <div className="question-asked">
                <strong>Question:</strong> {aiFeedback.question}
              </div>
              <div className="ai-answer">
                <strong>Answer:</strong>
                <div className="answer-text">{aiFeedback.answer}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
