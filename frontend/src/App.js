
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdFile, setJdFile] = useState(null);
  const [results, setResults] = useState([]);
  const [filter, setFilter] = useState('');
  const [feedbackQuestion, setFeedbackQuestion] = useState('');
  const [aiFeedback, setAiFeedback] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : true; // Default to dark mode
  });

  // Theme toggle functionality
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

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
      <div className="header-section">
        <h1>Zort Resume & JD Analyzer with AI Feedback</h1>
        <button 
          className="theme-toggle" 
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          <span className="theme-icon">{isDarkMode ? '☀️' : '🌙'}</span>
        </button>
      </div>
      
      {/* File Upload Section */}
      <form className="upload-form" onSubmit={handleUpload}>
        <div className="file-input-group">
          <label className="file-label">
            <span className="label-text">Upload Resume</span>
            <span className="label-subtitle">PDF, DOCX, or TXT files</span>
            <input 
              type="file" 
              accept=".pdf,.docx,.txt" 
              onChange={handleResumeChange}
              className="file-input"
            />
            <div className="file-input-display">
              {resumeFile ? (
                <span className="file-selected">{resumeFile.name}</span>
              ) : (
                <span className="file-placeholder">Choose resume file</span>
              )}
            </div>
          </label>
        </div>
        <div className="file-input-group">
          <label className="file-label">
            <span className="label-text">Upload Job Description</span>
            <span className="label-subtitle">PDF, DOCX, or TXT files</span>
            <input 
              type="file" 
              accept=".pdf,.docx,.txt" 
              onChange={handleJdChange}
              className="file-input"
            />
            <div className="file-input-display">
              {jdFile ? (
                <span className="file-selected">{jdFile.name}</span>
              ) : (
                <span className="file-placeholder">Choose job description file</span>
              )}
            </div>
          </label>
        </div>
        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? (
            <div className="button-loading">
              <div className="loading-spinner"></div>
              <span>Analyzing...</span>
            </div>
          ) : (
            'Analyze Documents'
          )}
        </button>
      </form>
      
      {error && (
        <div className="error-message">
          <div className="error-icon">!</div>
          <span>{error}</span>
        </div>
      )}
      
      {loading && (
        <div className="loading-overlay">
          <div className="loading-container">
            <div className="loading-spinner-large"></div>
            <h3>Analyzing Documents</h3>
            <p>Processing your resume and job description...</p>
            <div className="loading-progress">
              <div className="progress-bar"></div>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {results.length > 0 && (
        <div className="results-section">
          <h2>Analysis Results</h2>
          <div className="result-card">
            <div className="result-header">
              <h3>{results[0].candidate}</h3>
              <div className="result-badge">
                <span className={`verdict ${results[0].verdict.toLowerCase().replace(' ', '-')}`}>
                  {results[0].verdict}
                </span>
              </div>
            </div>
            
            <div className="score-display">
              <div className="score-item">
                <div className="score-icon">Overall</div>
                <div className="score-content">
                  <span className="score-label">Overall Score</span>
                  <span className={`score-value ${results[0].score >= 80 ? 'excellent' : results[0].score >= 60 ? 'good' : 'poor'}`}>
                    {results[0].score}/100
                  </span>
                </div>
                <div className="score-progress">
                  <div 
                    className="progress-fill" 
                    style={{width: `${results[0].score}%`}}
                  ></div>
                </div>
              </div>
              
              <div className="score-item">
                <div className="score-icon">Keywords</div>
                <div className="score-content">
                  <span className="score-label">Keyword Match</span>
                  <span className="score-value">{results[0].keyword_score}/100</span>
                </div>
                <div className="score-progress">
                  <div 
                    className="progress-fill" 
                    style={{width: `${results[0].keyword_score}%`}}
                  ></div>
                </div>
              </div>
              
              <div className="score-item">
                <div className="score-icon">Semantic</div>
                <div className="score-content">
                  <span className="score-label">Semantic Match</span>
                  <span className="score-value">{results[0].semantic_score}/100</span>
                </div>
                <div className="score-progress">
                  <div 
                    className="progress-fill" 
                    style={{width: `${results[0].semantic_score}%`}}
                  ></div>
                </div>
              </div>
            </div>
            
            <div className="feedback-section">
              <h4>Analysis Feedback</h4>
              <div className="feedback-text">
                {Array.isArray(results[0].feedback) ? 
                  results[0].feedback.map((item, idx) => (
                    <div key={idx} className="feedback-item">
                      <span className="feedback-content">{item}</span>
                    </div>
                  )) :
                  <div className="feedback-item">
                    <span className="feedback-content">{results[0].feedback}</span>
                  </div>
                }
              </div>
            </div>

            {/* Data Visualization Section */}
            <div className="data-visualization">
              <h4>Performance Metrics</h4>
              <div className="charts-container">
                {results[0].charts && results[0].charts.score_comparison ? (
                  <>
                    <div className="chart-card">
                      <h5>Score Comparison</h5>
                      <div className="python-chart">
                        <img 
                          src={results[0].charts.score_comparison} 
                          alt="Score Comparison Chart"
                          className="chart-image"
                        />
                      </div>
                    </div>

                    <div className="chart-card">
                      <h5>Radial Progress</h5>
                      <div className="python-chart">
                        <img 
                          src={results[0].charts.radial_progress} 
                          alt="Radial Progress Chart"
                          className="chart-image"
                        />
                      </div>
                    </div>

                    <div className="chart-card">
                      <h5>Skill Analysis</h5>
                      <div className="python-chart">
                        <img 
                          src={results[0].charts.skill_analysis} 
                          alt="Skill Analysis Chart"
                          className="chart-image"
                        />
                      </div>
                    </div>

                    {results[0].charts.interactive_chart && (
                      <div className="chart-card interactive-chart">
                        <h5>Interactive Dashboard</h5>
                        <div 
                          className="plotly-chart"
                          dangerouslySetInnerHTML={{__html: results[0].charts.interactive_chart}}
                        />
                      </div>
                    )}
                  </>
                ) : (
                  // Fallback to CSS charts if Python charts are not available
                  <>
                    <div className="chart-card">
                      <h5>Score Distribution</h5>
                      <div className="score-chart">
                        <div className="chart-bar">
                          <div className="bar-label">Overall</div>
                          <div className="bar-container">
                            <div 
                              className="bar-fill overall-bar" 
                              style={{width: `${results[0].score}%`}}
                            ></div>
                            <span className="bar-value">{results[0].score}%</span>
                          </div>
                        </div>
                        <div className="chart-bar">
                          <div className="bar-label">Keywords</div>
                          <div className="bar-container">
                            <div 
                              className="bar-fill keyword-bar" 
                              style={{width: `${results[0].keyword_score}%`}}
                            ></div>
                            <span className="bar-value">{results[0].keyword_score}%</span>
                          </div>
                        </div>
                        <div className="chart-bar">
                          <div className="bar-label">Semantic</div>
                          <div className="bar-container">
                            <div 
                              className="bar-fill semantic-bar" 
                              style={{width: `${results[0].semantic_score}%`}}
                            ></div>
                            <span className="bar-value">{results[0].semantic_score}%</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="chart-card">
                      <h5>Match Analysis</h5>
                      <div className="radial-chart">
                        <div className="radial-progress">
                          <svg className="radial-svg" viewBox="0 0 120 120">
                            <circle
                              className="radial-bg"
                              cx="60"
                              cy="60"
                              r="50"
                              fill="none"
                              stroke="var(--border-color)"
                              strokeWidth="8"
                            />
                            <circle
                              className="radial-fill"
                              cx="60"
                              cy="60"
                              r="50"
                              fill="none"
                              stroke="url(#gradient)"
                              strokeWidth="8"
                              strokeLinecap="round"
                              strokeDasharray={`${2 * Math.PI * 50}`}
                              strokeDashoffset={`${2 * Math.PI * 50 * (1 - results[0].score / 100)}`}
                              transform="rotate(-90 60 60)"
                            />
                            <defs>
                              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#667eea" />
                                <stop offset="100%" stopColor="#764ba2" />
                              </linearGradient>
                            </defs>
                          </svg>
                          <div className="radial-text">
                            <span className="radial-score">{results[0].score}</span>
                            <span className="radial-label">Match Score</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Feedback Section */}
      {results.length > 0 && (
        <div className="ai-feedback-section">
          <h2>AI Career Coach</h2>
          <p>Get personalized feedback and advice about your resume and job match</p>
          
          <form onSubmit={handleFeedbackRequest} className="feedback-form">
            <div className="question-input">
              <label>Ask a question</label>
              <textarea
                value={feedbackQuestion}
                onChange={(e) => setFeedbackQuestion(e.target.value)}
                placeholder="What skills should I highlight more? How can I improve my resume for this role? What certifications would help?"
                rows="3"
                className="question-textarea"
              />
            </div>
            <button type="submit" disabled={feedbackLoading} className="feedback-button">
              {feedbackLoading ? (
                <div className="button-loading">
                  <div className="loading-spinner"></div>
                  <span>Getting AI feedback...</span>
                </div>
              ) : (
                'Get AI Feedback'
              )}
            </button>
          </form>

          {aiFeedback && (
            <div className="ai-response">
              <div className="response-header">
                <h3>AI Career Coach Response</h3>
                <div className="ai-avatar">AI</div>
              </div>
              <div className="question-asked">
                <div className="question-label">Question</div>
                <div className="question-text">{aiFeedback.question}</div>
              </div>
              <div className="ai-answer">
                <div className="answer-label">Answer</div>
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
