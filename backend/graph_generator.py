"""
Professional Graph Generator for Resume Analysis
Generates high-quality charts and visualizations using matplotlib and seaborn
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import io
import base64
from typing import Dict, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Set professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GraphGenerator:
    """Generate professional charts and graphs for resume analysis"""
    
    def __init__(self):
        """Initialize the graph generator with professional styling"""
        self.setup_matplotlib_style()
        self.setup_plotly_style()
    
    def setup_matplotlib_style(self):
        """Configure matplotlib for professional appearance"""
        plt.rcParams.update({
            'font.family': 'Segoe UI',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#333333',
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333'
        })
    
    def setup_plotly_style(self):
        """Configure plotly for professional appearance"""
        pio.templates.default = "plotly_white"
    
    def create_score_comparison_chart(self, data: Dict[str, Any]) -> str:
        """Create a professional score comparison bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data preparation
        categories = ['Overall Score', 'Keyword Match', 'Semantic Match']
        scores = [data['score'], data['keyword_score'], data['semantic_score']]
        colors = ['#667eea', '#4facfe', '#43e97b']
        
        # Create bars with gradient effect
        bars = ax.bar(categories, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Customize chart
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score (%)', fontweight='bold')
        ax.set_title('Resume Analysis Score Breakdown', fontweight='bold', pad=20)
        
        # Add horizontal line at 80% for reference
        ax.axhline(y=80, color='#ff6b6b', linestyle='--', alpha=0.7, label='Excellent Threshold')
        ax.axhline(y=60, color='#f59e0b', linestyle='--', alpha=0.7, label='Good Threshold')
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Improve layout
        plt.tight_layout()
        
        return self._fig_to_base64(fig)
    
    def create_radial_progress_chart(self, data: Dict[str, Any]) -> str:
        """Create a professional radial progress chart"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Data
        score = data['score']
        categories = ['Overall Match', 'Keyword Match', 'Semantic Match']
        values = [score, data['keyword_score'], data['semantic_score']]
        colors = ['#667eea', '#4facfe', '#43e97b']
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=3, color='#667eea', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='#667eea')
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add title
        ax.set_title('Resume Match Analysis\nRadial View', fontweight='bold', pad=30, fontsize=14)
        
        # Add center text
        ax.text(0, 0, f'{score}%\nOverall', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='#667eea')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_skill_analysis_chart(self, data: Dict[str, Any]) -> str:
        """Create a skill analysis donut chart"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Mock skill data (in real implementation, this would come from analysis)
        skills = ['Technical Skills', 'Soft Skills', 'Experience Match', 'Education Match']
        values = [data['keyword_score'], data['semantic_score'], 
                 min(100, data['score'] + 10), min(100, data['score'] - 5)]
        colors = ['#667eea', '#4facfe', '#43e97b', '#f093fb']
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(values, labels=skills, colors=colors, autopct='%1.1f%%',
                                         startangle=90, pctdistance=0.85, textprops={'fontweight': 'bold'})
        
        # Create donut hole
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        # Add center text
        ax.text(0, 0, f'Skill\nAnalysis\n{data["score"]}%', ha='center', va='center',
               fontsize=14, fontweight='bold', color='#333333')
        
        # Customize
        ax.set_title('Skill Match Analysis', fontweight='bold', pad=20, fontsize=16)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_plotly_interactive_chart(self, data: Dict[str, Any]) -> str:
        """Create an interactive plotly chart"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Breakdown', 'Match Analysis', 'Performance Metrics', 'Trend Analysis'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Score breakdown bar chart
        categories = ['Overall', 'Keywords', 'Semantic']
        scores = [data['score'], data['keyword_score'], data['semantic_score']]
        colors = ['#667eea', '#4facfe', '#43e97b']
        
        fig.add_trace(
            go.Bar(x=categories, y=scores, marker_color=colors, name='Scores',
                  text=scores, textposition='auto'),
            row=1, col=1
        )
        
        # Match analysis pie chart
        fig.add_trace(
            go.Pie(labels=['Matched', 'Unmatched'], 
                  values=[data['score'], 100-data['score']],
                  marker_colors=['#667eea', '#e0e0e0']),
            row=1, col=2
        )
        
        # Performance metrics scatter
        x_vals = [1, 2, 3, 4, 5]
        y_vals = [data['score'] - 10, data['keyword_score'], data['semantic_score'], 
                 data['score'] + 5, data['score']]
        
        fig.add_trace(
            go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                      line=dict(color='#667eea', width=3),
                      marker=dict(size=8, color='#667eea')),
            row=2, col=1
        )
        
        # Gauge indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=data['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Match"},
                delta={'reference': 80},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "#667eea"},
                      'steps': [{'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 90}}),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Comprehensive Resume Analysis Dashboard",
            title_x=0.5,
            font=dict(family="Segoe UI", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Convert to HTML
        return fig.to_html(include_plotlyjs='cdn', div_id="plotly-chart")
    
    def create_professional_dashboard(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create a complete professional dashboard with multiple charts"""
        dashboard = {}
        
        # Generate all chart types
        dashboard['score_comparison'] = self.create_score_comparison_chart(data)
        dashboard['radial_progress'] = self.create_radial_progress_chart(data)
        dashboard['skill_analysis'] = self.create_skill_analysis_chart(data)
        dashboard['interactive_chart'] = self.create_plotly_interactive_chart(data)
        
        return dashboard
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

# Global instance
graph_generator = GraphGenerator()
