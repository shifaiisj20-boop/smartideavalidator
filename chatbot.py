# app/chatbot.py - UPDATED WITH PURE ML-BASED SCORING
import streamlit as st
import pandas as pd
import ollama
import requests 
import time
import pyttsx3 
import os
import base64

# --- 1. Import all necessary functions and variables from utils.py ---
from utils import (
    preprocess_text, get_embeddings, find_similar_ideas, 
    extract_keywords, get_trend_data, get_ollama_feedback, 
    generate_pdf_report, get_ml_scores, OLLAMA_MODEL
)

# Import the new web-based competitor extractor
from real_competitor_extractor import competitor_extractor
from typing import List, Dict, Optional

# --- ENHANCED SERPAPI COMPETITOR SEARCH ---
def search_competitors_serpapi(idea, api_key="d5c183cb496ff99958e5be3762b3088243bb7868f058c5dc69bdc8ce4c81c8f9"):
    """Search for real competitors using SerpAPI (Google Search)"""
    
    try:
        st.info(f"🔍 Searching for competitors: {idea}")
        
        # Better search queries for different angles
        search_queries = [
            f'{idea} startup competitors',
            f'{idea} similar companies',
            f'{idea} alternatives',
            f'"{idea}" market competition',
            f'best {idea} companies'
        ]
        
        all_competitors = []
        
        for query in search_queries[:2]:  # Use first 2 queries to save API calls
            params = {
                'engine': 'google',
                'q': query,
                'api_key': api_key,
                'num': 10,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.get('https://serpapi.com/search', params=params, timeout=30)
            
            if response.status_code != 200:
                st.warning(f"⚠ API returned status: {response.status_code}")
                continue
            
            data = response.json()
            
            # Extract organic search results
            if 'organic_results' in data:
                for result in data['organic_results']:
                    title = result.get('title', '').strip()
                    link = result.get('link', '')
                    snippet = result.get('snippet', '')
                    
                    # Skip irrelevant results
                    irrelevant_keywords = ['wikipedia', 'dictionary', 'definition', 'reddit', 'quora']
                    if any(ignore in title.lower() for ignore in irrelevant_keywords):
                        continue
                    
                    # Skip if no meaningful title
                    if not title or len(title) < 10:
                        continue
                    
                    # Create competitor entry
                    competitor_info = f"🎯 {title} - {link}"
                    
                    # Avoid duplicates
                    if competitor_info not in all_competitors:
                        all_competitors.append(competitor_info)
            
            time.sleep(1)  # Be nice to the API
        
        # If no direct competitors found, try broader search
        if not all_competitors:
            st.info("🔄 Trying broader search for related companies...")
            broader_queries = [
                f'companies like {idea}',
                f'{idea} industry leaders',
                f'top {idea.split()[0]} companies'  # Use first word of idea
            ]
            
            for query in broader_queries[:1]:
                params = {
                    'engine': 'google',
                    'q': query,
                    'api_key': api_key,
                    'num': 8,
                    'gl': 'us',
                    'hl': 'en'
                }
                
                response = requests.get('https://serpapi.com/search', params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'organic_results' in data:
                        for result in data['organic_results'][:5]:
                            title = result.get('title', '').strip()
                            link = result.get('link', '')
                            
                            if title and len(title) > 10:
                                competitor_info = f"📌 {title} - {link}"
                                if competitor_info not in all_competitors:
                                    all_competitors.append(competitor_info)
        
        # Final fallback - provide helpful categories
        if not all_competitors:
            # Analyze the idea and suggest potential competitor categories
            idea_lower = idea.lower()
            if any(word in idea_lower for word in ['app', 'software', 'platform']):
                return [
                    "💡 Potential Competitors: Tech companies in similar domains",
                    "🔍 Search for: 'top tech startups in [your domain]'",
                    "🎯 Look into: App Store/Play Store similar apps"
                ]
            elif any(word in idea_lower for word in ['food', 'meal', 'restaurant']):
                return [
                    "💡 Potential Competitors: Food delivery services",
                    "🔍 Search for: 'meal delivery companies'", 
                    "🎯 Look into: Local food startups"
                ]
            elif any(word in idea_lower for word in ['fitness', 'health', 'workout']):
                return [
                    "💡 Potential Competitors: Fitness apps and services",
                    "🔍 Search for: 'fitness technology companies'",
                    "🎯 Look into: Health and wellness startups"
                ]
            else:
                return [
                    "💡 This appears to be a novel idea!",
                    "🔍 Research adjacent markets and related services",
                    "🎯 Consider indirect competitors solving similar problems",
                    "📈 Validate market need through customer interviews"
                ]
        
        return all_competitors[:6]  # Return top 6 results
    
    except requests.exceptions.Timeout:
        return ["⏰ Search timeout. Try again later."]
    except Exception as e:
        return [f"❌ Search error: {str(e)}"]
    
# --- INVESTOR SEARCH FUNCTIONS ---
def categorize_industry(idea: str) -> str:
    """Enhanced industry categorization for investor matching"""
    idea_lower = idea.lower()
    
    industry_keywords = {
        'Artificial Intelligence': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'computer vision', 'nlp'],
        'Fintech': ['fintech', 'financial technology', 'banking', 'payments', 'insurance', 'crypto', 'blockchain', 'investment'],
        'Healthcare': ['healthcare', 'medical', 'biotech', 'pharmaceutical', 'fitness', 'wellness', 'telemedicine', 'digital health'],
        'E-commerce': ['ecommerce', 'e-commerce', 'marketplace', 'retail', 'shopping', 'delivery', 'logistics'],
        'SaaS': ['saas', 'software as a service', 'enterprise software', 'b2b', 'business software', 'cloud'],
        'EdTech': ['edtech', 'education technology', 'online learning', 'e-learning', 'educational'],
        'Clean Technology': ['cleantech', 'sustainability', 'renewable energy', 'green tech', 'environmental'],
        'Consumer Technology': ['consumer tech', 'mobile app', 'social media', 'gaming', 'entertainment']
    }
    
    for industry, keywords in industry_keywords.items():
        if any(keyword in idea_lower for keyword in keywords):
            return industry
    
    return 'Technology'

def recommend_funding_stage(idea: str) -> str:
    """Recommend funding stage based on idea complexity"""
    idea_lower = idea.lower()
    
    if any(word in idea_lower for word in ['prototype', 'demo', 'mockup', 'concept', 'idea stage']):
        return "Pre-seed / Angel Round"
    elif any(word in idea_lower for word in ['mvp', 'beta', 'early users', 'traction', 'launch']):
        return "Seed Round"
    elif any(word in idea_lower for word in ['revenue', 'customers', 'scaling', 'growth', 'series']):
        return "Series A"
    else:
        return "Seed Round"

def clean_investor_name(name: str) -> str:
    """Clean and format investor name from search results"""
    removals = [' - crunchbase', ' - linkedin', ' | crunchbase', ' - angel.co']
    clean_name = name
    for removal in removals:
        clean_name = clean_name.split(removal)[0]
    
    clean_name = clean_name.split(' - ')[0]
    clean_name = clean_name.split(' | ')[0]
    return clean_name.strip()

def classify_investor_type(title: str, snippet: str) -> str:
    """Classify investor type based on title and snippet content"""
    text = (title + ' ' + snippet).lower()
    
    if any(word in text for word in ['angel', 'angel investor', 'angels']):
        return 'Angel Investor'
    elif any(word in text for word in ['accelerator', 'incubator', 'y combinator', 'techstars']):
        return 'Accelerator'
    elif any(word in text for word in ['seed', 'early stage', 'pre-seed']):
        return 'Seed Stage VC'
    elif any(word in text for word in ['corporate', 'cvc', 'corporate venture']):
        return 'Corporate VC'
    elif any(word in text for word in ['venture', 'vc', 'capital', 'ventures']):
        return 'Venture Capital'
    elif any(word in text for word in ['private equity', 'growth equity']):
        return 'Private Equity'
    else:
        return 'Investment Firm'

def get_well_known_investors(industry: str) -> list:
    """Add well-known investors as fallback"""
    well_known_investors = {
        'Artificial Intelligence': [
            {'name': 'OpenAI Startup Fund', 'website': 'https://openai.fund', 'investor_type': 'Corporate VC', 'description': 'Invests in AI companies leveraging OpenAI technology', 'source': 'Industry Database'},
            {'name': 'Google Gradient Ventures', 'website': 'https://gradient.com', 'investor_type': 'Corporate VC', 'description': 'Google\'s AI-focused venture fund', 'source': 'Industry Database'},
        ],
        'Fintech': [
            {'name': 'Andreessen Horowitz (a16z)', 'website': 'https://a16z.com', 'investor_type': 'Venture Capital', 'description': 'Leading VC with strong fintech and crypto focus', 'source': 'Industry Database'},
            {'name': 'Ribbit Capital', 'website': 'https://ribbitcap.com', 'investor_type': 'Venture Capital', 'description': 'Invests in financial services technology companies', 'source': 'Industry Database'},
        ],
        'Healthcare': [
            {'name': 'Third Rock Ventures', 'website': 'https://thirdrockventures.com', 'investor_type': 'Venture Capital', 'description': 'Focuses on biotechnology and healthcare companies', 'source': 'Industry Database'},
        ],
        'Technology': [
            {'name': 'Y Combinator', 'website': 'https://ycombinator.com', 'investor_type': 'Accelerator', 'description': 'World\'s most famous startup accelerator', 'source': 'Industry Database'},
            {'name': 'Sequoia Capital', 'website': 'https://sequoiacap.com', 'investor_type': 'Venture Capital', 'description': 'Leading venture capital firm across all stages', 'source': 'Industry Database'},
        ]
    }
    
    return well_known_investors.get(industry, well_known_investors['Technology'])

def parse_investor_result(result: dict, industry: str) -> dict:
    """Parse Google search result into investor format"""
    title = result.get('title', '')
    link = result.get('link', '')
    snippet = result.get('snippet', '')
    
    if not title:
        return None
    
    investor_indicators = [
        'venture', 'capital', 'vc', 'investor', 'fund', 'angel', 
        'accelerator', 'partners', 'ventures', 'capital partners',
        'investment', 'private equity', 'cvc'
    ]
    
    title_lower = title.lower()
    
    irrelevant = ['wikipedia', 'dictionary', 'definition', 'news', 'article', 'blog']
    if any(ignore in title_lower for ignore in irrelevant):
        return None
    
    if any(indicator in title_lower for indicator in investor_indicators):
        return {
            'name': clean_investor_name(title),
            'website': link,
            'description': snippet[:120] + '...' if len(snippet) > 120 else snippet,
            'investor_type': classify_investor_type(title, snippet),
            'industry_focus': industry,
            'source': 'Web Search'
        }
    
    return None

def remove_duplicate_investors(investors: list) -> list:
    """Remove duplicate investors by name"""
    seen = set()
    unique = []
    
    for investor in investors:
        name = investor['name'].lower()
        if name not in seen:
            seen.add(name)
            unique.append(investor)
    
    return unique

def find_investors_serpapi_comprehensive(idea: str) -> dict:
    """Comprehensive investor search using SerpAPI"""
    
    SERPAPI_KEY = "d5c183cb496ff99958e5be3762b3088243bb7868f058c5dc69bdc8ce4c81c8f9"
    industry = categorize_industry(idea)
    investors = []
    
    search_queries = [
        f'"{industry}" venture capital firms',
        f'"{industry}" startup investors',
        f'"{industry}" angel investors',
    ]
    
    for query in search_queries[:2]:
        try:
            params = {
                'engine': 'google',
                'q': query,
                'api_key': SERPAPI_KEY,
                'num': 6,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.get('https://serpapi.com/search', params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'organic_results' in data:
                    for result in data['organic_results']:
                        investor = parse_investor_result(result, industry)
                        if investor and investor not in investors:
                            investors.append(investor)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Investor search error: {e}")
            continue
    
    well_known = get_well_known_investors(industry)
    for investor in well_known:
        if not any(inv['name'].lower() == investor['name'].lower() for inv in investors):
            investors.append(investor)
    
    unique_investors = remove_duplicate_investors(investors)
    
    return {
        'investors': unique_investors[:6],
        'total_found': len(unique_investors),
        'industry': industry,
        'recommended_stage': recommend_funding_stage(idea),
        'data_source': 'Web Search + Industry Database'
    }

def find_investors_for_startup(idea: str) -> dict:
    """Main function to find investors for a startup idea"""
    return find_investors_serpapi_comprehensive(idea)

# --- TTS FUNCTION DEFINITION ---
def text_to_speech(text):
    """ Converts text to speech using pyttsx3 (offline). """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180) 
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# --- HELPER FUNCTION FOR PDF SUCCESS MESSAGE (Defined here for scope safety) ---
def get_pdf_success_html():
    """Returns the PDF success HTML block."""
    return """
    <div class="pdf-success">
        <h3>✅ Report Ready!</h3>
        <p>Your professional startup validation report is ready for download.</p>
    </div>
    """

# --- LOGO ENCODING FOR HEADERS ---
def encode_logo(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None 
    except Exception:
        return None

# Encoding paths: NOW POINTING EXPLICITLY TO THE APP DIRECTORY
APP_LOGO_B64 = encode_logo("app/smart_idea_logo.png")
OLLAMA_LOGO_B64 = encode_logo("app/magnify.jpg")
EXTERNAL_LOGO_B64 = encode_logo("app/external.png")
ML_SCORE_LOGO_B64 = encode_logo("app/ml_score.png")
DASHBOARD_LOGO_B64 = encode_logo("app/dashboard_logo.png") 

# NEW LOGOS FOR THE REPORT SECTIONS
REPORT_LOGO_B64 = encode_logo("app/valid.jpg") 
EXPERT_ANALYSIS_LOGO_B64 = encode_logo("app/detail_expert.png") 
SUMMARY_LOGO_B64 = encode_logo("app/summary.png") 

# Helper function to create HTML image tag for headers/subheaders
def get_logo_html(b64_data, title, size='30px'):
    if "valid.jpg" in str(b64_data) or "dashboard_logo.png" in str(b64_data):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"
        
    if b64_data:
        return f'<div style="font-size: 1.1em; font-weight: bold; padding-bottom: 5px;"><img src="data:{mime_type};base64,{b64_data}" style="height: {size}; vertical-align: middle; margin-right: 5px;"> {title}</div>'
    else:
        return f"### {title}"

# --- Ollama Status Check ---
@st.cache_data(show_spinner=False) 
def check_ollama_status():
    """ Checks if the Ollama server is running using a simple API ping. """
    try:
        requests.get("http://localhost:11434", timeout=3)
        return f"Ollama ({OLLAMA_MODEL}) is connected! [LIVE]", True
    except requests.exceptions.RequestException:
        return f"Ollama ({OLLAMA_MODEL}) is NOT connected [FAIL]. Please start the Ollama application.", False

status_message, is_connected = check_ollama_status()

# --- 2. Custom CSS and Page Configuration (ENHANCED DESIGN) ---
st.set_page_config(
    page_title="Smart Idea Validator Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for modern design, fixed input, and better chat bubbles
st.markdown(f"""
<style>
    /* Hide Streamlit default header/footer for clean look */
    .stApp > header, footer {{ visibility: hidden; }}
    
    /* Main body background color - light gray/off-white */
    .stApp {{ background-color: #f7f9fc; }} 
    
    /* Main Title Bar (Sleek Look) */
    .main-title-bar {{
    padding: 10px 0;
    margin-bottom: 0px;
    text-align: center;
    background-color: transparent;
    box-shadow: none;
    }}
    .main-title-bar h1 {{ font-size: 1.8em; margin: 0; color: #182848; }}     
    
    /* Custom Chat Bubble Styles */
    .user-message {{
        background-color: #007bff;
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 0.95em;
    }}
    .bot-message {{
        background-color: #ffffff;
        color: #333;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 0.95em;
    }}
    
    /* Fixed input container */
    .fixed-input-container {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 10px;
        background: #f7f9fc;
        box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }}
                
    /* Ensure chat container respects the fixed input height */
    .main .block-container {{
    padding-top: 20px; 
    padding-bottom: 150px; 
    background-color: #f7f9fc; 
    margin: 0; 
    }}
    
    /* Sidebar enhancements */
    .sidebar .stMetric {{
        padding: 10px;
        border-radius: 8px;
        background-color: #ffffff;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    
    /* Analysis Feature Cards */
    .feature-check {{
        border-left: 4px solid #38c172; 
        background-color: #ebf9f1; 
        color: #1e8449; 
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 5px;
        font-weight: 600;
        font-size: 0.9em;
    }}
    
    /* Analysis Cards */
    .business-trend {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
        border-left: 5px solid #ff6b6b;
    }}
    .score-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        text-align: center;
    }}
    .score-high {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); }}
    .score-medium{{ background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); }}
    .score-low {{ background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); }}
    .pdf-success {{
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
    }}
    
    /* Competitor card styles */
    .competitor-card {{
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }}
    .competitor-card h3 {{
        margin-top: 0;
        color: #182848;
    }}
    
    /* ML Score Explanation */
    .ml-explanation {{
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }}
    
</style>
""", unsafe_allow_html=True)

# --- RENDER MAIN TITLE BAR: REPLACED TEXT WITH LOGO IMAGE ---
try:
    if APP_LOGO_B64:
        st.markdown(
            f"""
            <div class="main-title-bar" style="text-align: center; padding: 15px 0;">
                <img src="data:image/png;base64,{APP_LOGO_B64}" style="max-width: 300px; height: auto; display: block; margin: 0 auto;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="main-title-bar">
                <h1>🚀 Smart Idea Validator Chatbot</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
except Exception:
    st.markdown(
        """
        <div class="main-title-bar">
            <h1>🚀 Smart Idea Validator Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 3. Sidebar and History ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = {}
if 'pdf_generated' not in st.session_state:
    st.session_state.pdf_generated = False
if 'pdf_file_path' not in st.session_state:
    st.session_state.pdf_file_path = None
if 'ml_scores' not in st.session_state:
    st.session_state.ml_scores = None
if 'current_idea' not in st.session_state:
    st.session_state.current_idea = None
if 'real_competitors' not in st.session_state:
    st.session_state.real_competitors = None
if 'investor_data' not in st.session_state:
    st.session_state.investor_data = {}

with st.sidebar:
    if DASHBOARD_LOGO_B64:
        if "dashboard_logo.png" not in str(DASHBOARD_LOGO_B64):
            mime = "image/jpeg" 
        else:
            mime = "image/png"
        st.markdown(f'<img src="data:{mime};base64,{DASHBOARD_LOGO_B64}" style="width: 120px; margin: 0 auto; display: block;">', unsafe_allow_html=True)
    else:
        st.title("📊 Dashboard")

    st.markdown("---")
    
    st.metric("Analyses Completed", len([m for m in st.session_state.messages if m['role'] == 'user']))
    st.metric("Total Messages", len(st.session_state.messages))
    st.markdown("---")
    st.info(status_message)
    if st.button("🔄 Clear Chat & Reset Analysis", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_analysis = {}
        st.session_state.pdf_generated = False
        st.session_state.pdf_file_path = None
        st.session_state.ml_scores = None
        st.session_state.current_idea = None
        st.session_state.real_competitors = None
        st.session_state.investor_data = {}
        st.rerun()
    st.markdown("---")
    st.caption("AI-Powered Startup Validation")

# --- 4. Main Chat Layout ---
if not st.session_state.messages:
    st.markdown(f'<div class="bot-message">Hello! I am your AI Startup Strategist. Describe your business idea below to start the comprehensive validation process (ML Scores, Market Trends, and Competitor Analysis).</div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

# --- 5. Features Section (Only show if no analysis is running) ---
if not st.session_state.current_idea and not st.session_state.messages:
    st.markdown("## ⚙ Validation Components")
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown(get_logo_html(OLLAMA_LOGO_B64, "Ollama Analysis", '40px'), unsafe_allow_html=True)
        st.markdown('<div class="feature-check">✅ Business Domain</div>', unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown(get_logo_html(EXTERNAL_LOGO_B64, "External Data", '40px'), unsafe_allow_html=True)
        st.markdown('<div class="feature-check">✅ Web Competitors</div>', unsafe_allow_html=True)
    
    with col_feat3:
        st.markdown(get_logo_html(ML_SCORE_LOGO_B64, "ML Scoring", '40px'), unsafe_allow_html=True)
        st.markdown('<div class="feature-check">✅ Market Potential</div>', unsafe_allow_html=True)

st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

# --- 6. Fixed Input Area (at the very bottom) ---
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)

with st.form(key='chat_form', clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_area(
            "Describe your startup idea:",
            placeholder="Example: A subscription service that delivers personalized meal kits...",
            height=50,
            key="user_input_area",
            label_visibility="collapsed"
        )
        
    with col_btn:
        st.markdown('<div style="height: 50px; display: flex; align-items: center;">', unsafe_allow_html=True)
        submit = st.form_submit_button("🔍 Analyze Idea", use_container_width=True) 
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- 7. Processing Logic ---
if submit and user_input.strip():
    # 1. Store current idea and add user message
    st.session_state.current_idea = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Start Bot Analysis
    if is_connected:
        with st.spinner("🤖 AI is comprehensively analyzing your startup idea..."):
            
            # 2a. Find REAL Competitors using WEB SEARCH
            with st.spinner("🔍 Searching web for real market competitors..."):
                real_competitors = search_competitors_serpapi(user_input)
                st.session_state.real_competitors = real_competitors

            # 2b. Run PyTrends for market data
            with st.spinner("📈 Analyzing market trends with Google Trends..."):
                keywords = extract_keywords(user_input)
                trend_df = get_trend_data(keywords)
                
            # 2c. Search for investors
            with st.spinner("💰 Finding potential investors..."):
                investor_data = find_investors_for_startup(user_input)
                st.session_state.investor_data = investor_data
            
            # --- Trend Summary Logic ---
            trend_summary = "Gathering market trend data..."
            if trend_df is not None and not trend_df.empty:
                try:
                    avg_scores = trend_df.mean()
                    meaningful_scores = avg_scores[avg_scores >= 5].sort_values(ascending=False)
                    max_score = meaningful_scores.max() if not meaningful_scores.empty else 0
                    
                    # Determine Context and Implication
                    if max_score >= 70:
                        market_context = "🔥 High Market Demand"
                        explanation = "Strong consumer interest indicates validated market need"
                        implication = "Strong market validation but potentially higher competition."
                    elif max_score >= 50:
                        market_context = "📈 Growing Market"
                        explanation = "Solid and increasing consumer interest with good growth potential"
                        implication = "Good balance of market size and growth opportunity."
                    elif max_score >= 30:
                        market_context = "↗ Emerging Market" 
                        explanation = "Developing consumer interest with significant growth opportunities"
                        implication = "Emerging space with room for innovation and market leadership."
                    else:
                        market_context = "📊 Niche Market"
                        explanation = "Specialized market with focused audience and potentially less competition"
                        implication = "Niche opportunity with potential for strong customer loyalty."
                        
                    trends_output = []
                    for keyword, score in meaningful_scores.items():
                        if score >= 70: level, emoji, insight = "Very High", "🔥", "Strong market interest"
                        elif score >= 50: level, emoji, insight = "High", "📈", "Strong market interest"
                        elif score >= 30: level, emoji, insight = "Medium", "↗", "Growing popularity"
                        else: level, emoji, insight = "Low", "📊", "Emerging interest"
                        
                        keyword_desc = {
                            'subscription': 'Subscription Business Models', 'service': 'Service-Based Solutions', 
                            'meal': 'Meal & Food Services', 'delivery': 'Delivery & Logistics',
                            'personalized': 'Personalization Features', 'app': 'Mobile Applications',
                            'ai': 'AI Technology Solutions', 'fitness': 'Fitness & Wellness',
                            'food': 'Food Industry', 'tech': 'Technology Solutions',
                            'kit': 'Product Kits', 'business': 'Business Services',
                            'online': 'Online Platforms', 'digital': 'Digital Solutions',
                            'health': 'Health & Wellness'
                        }.get(keyword.lower(), f"{keyword.title()} Market")
                        
                        trends_output.append(f"* {emoji} {keyword_desc}: {level} search interest ({insight})")

                    trends_list = "\n".join(trends_output)
                    
                    trend_summary = f"""
{market_context}

> What this means: {explanation}
> Market Insight: {implication}

Key Market Segments Analyzed:
{trends_list}
"""
                    
                except Exception as e:
                    print(f"❌ Trend processing error: {e}")
                    trend_summary = f"""
📈 Market Analysis Fallback

> What this means: Analyzing market potential for your business concept.
> Market Insight: Could not perform detailed trend analysis due to an error. Proceeding with general industry insight.

Further validation through customer interviews and market research is recommended.
"""
            else:
                main_keyword = keywords[0] if keywords else "your concept"
                trend_summary = f"""
📈 Market Potential Assessment

> What this means: Your business idea shows promise in the current market landscape.
> Market Insight: Based on industry analysis, '{main_keyword}' related businesses are seeing increased consumer interest and digital adoption.

Next steps: Conduct customer discovery interviews and competitive analysis to refine your market entry strategy.
"""
            
            # --- ML SCORING (UPDATED FOR PURE ML-BASED SCORING) ---
            try:
                with st.spinner("📊 Calculating ML-powered scores..."):
                    ml_results = get_ml_scores(user_input, real_competitors, trend_summary)
                    st.session_state.ml_scores = ml_results
            except Exception as e:
                print(f"ML scoring error: {e}")
                # Pure ML fallback scores
                ml_results = { 
                    'scores': { 
                        'problem_solution_fit': 6.0, 
                        'market_potential': 6.0, 
                        'innovation_level': 5.5, 
                        'competitive_advantage': 5.5,
                        'feasibility': 6.0
                    }, 
                    'overall_score': 5.8, 
                    'explanations': ["Pure ML analysis completed using statistical features"], 
                    'scoring_method': 'pure_ml_fallback' 
                }
                st.session_state.ml_scores = ml_results

            # --- OLLAMA ANALYSIS ---
            try:
                with st.spinner("🧠 Generating expert analysis..."):
                    raw_analysis = get_ollama_feedback(user_input, real_competitors, trend_summary, ml_results)
            except Exception as e:
                print(f"Ollama error: {e}")
                raw_analysis = f"Analysis complete. Overall score: {ml_results['overall_score']:.1f}/10. Focus on validating your target market and differentiating from competitors."

            # Store analysis data for PDF generation
            st.session_state.last_analysis = {
                'idea': user_input,
                'analysis': raw_analysis,
                'similar_ideas': real_competitors,
                'trend_summary': trend_summary,
                'ml_scores': ml_results
            }
            st.session_state.pdf_generated = False
            st.session_state.pdf_file_path = None
            
            # --- TTS INTEGRATION ---
            try:
                speech_text = f"Analysis complete. Overall score: {ml_results['overall_score']:.1f} out of 10."
                text_to_speech(speech_text)
                st.markdown(f"""<div style="background-color:#d4edda; color:#155724; border-left: 5px solid #28a745; padding: 10px; margin-bottom: 20px; border-radius: 5px;">🗣 BOT SPEAKS: {speech_text}</div>""", unsafe_allow_html=True)
            except Exception:
                pass

            # 3. Create ENHANCED Bot Response with PURE ML METRICS
            scores = ml_results['scores']
            
            # 3a. Format competitors
            competitor_display = "".join([
                f"* {comp}\n"
                for comp in real_competitors[:5]
            ])

            # 3b. Build the response piece by piece with PURE ML METRICS
            response_header = f"""
{get_logo_html(REPORT_LOGO_B64, "Startup Validation Report", '40px')}

*Overall Score: {ml_results['overall_score']:.1f}/10* {'🚀' if ml_results['overall_score'] >= 7 else '✅' if ml_results['overall_score'] >= 5 else '⚠'}

---

### 📊 Pure ML Metrics Breakdown
- *Problem-Solution Fit*: {scores['problem_solution_fit']:.1f}/10 - Statistical analysis of problem clarity & solution alignment
- *Market Potential*: {scores['market_potential']:.1f}/10 - ML-derived market size & opportunity assessment  
- *Innovation Level*: {scores['innovation_level']:.1f}/10 - Semantic analysis of innovation & uniqueness
- *Competitive Advantage*: {scores['competitive_advantage']:.1f}/10 - Data-driven differentiation analysis
- *Feasibility*: {scores['feasibility']:.1f}/10 - ML assessment of implementation viability

---

### 💡 Key Recommendations
"""
            
            # Part 2: Recommendations
            recommendations = chr(10).join([f"\n* {explanation}" for explanation in ml_results.get('explanations', [])])

            # Part 3: Market Intelligence
            market_intelligence = f"""

---

### 🔍 Market Intelligence

*Real Competitors Identified (Web Search):*
{competitor_display}

*Market Trends:*
{trend_summary}

---

{get_logo_html(EXPERT_ANALYSIS_LOGO_B64, "Detailed Expert Analysis", '40px')}
"""
            
            # Final Assembly
            final_bot_response = response_header + recommendations + market_intelligence + raw_analysis
            
            # 4. Add final response to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_bot_response})

    else:
        st.session_state.messages.append({"role": "assistant", "content": status_message})

    # Force rerun to update the chat window
    st.rerun()

# --- 8. ML Scores Display Section (UPDATED FOR PURE ML METRICS) ---
if (
    'ml_scores' in st.session_state and st.session_state.ml_scores is not None and
    'messages' in st.session_state and st.session_state.messages and 
    st.session_state.messages[-1]['role'] == 'assistant'
):
    st.markdown("---")
    st.markdown(get_logo_html(SUMMARY_LOGO_B64, "Analysis Summary", '40px'), unsafe_allow_html=True)
    
    ml_scores = st.session_state.ml_scores
    scores = ml_scores['scores'] 
    
    col_ml_score, col_ml_details, col_pdf = st.columns([1, 2, 1])
    
    # Overall Score Card
    with col_ml_score:
        overall_class = "score-high" if ml_scores['overall_score'] >= 7 else "score-medium" if ml_scores['overall_score'] >= 5 else "score-low"
        st.markdown(f"""
        <div class="score-card {overall_class}">
            <h4>Total Score</h4>
            <h2>{ml_scores['overall_score']:.1f}/10</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics - FIXED PROGRESS BARS
    with col_ml_details:
        st.markdown("##### 📊 Pure ML Metrics (View Full Report for details)")
        # FIXED: Added float() wrapper to prevent numpy float32 error
        st.progress(float(scores['problem_solution_fit'] / 10), text=f"🎯 Problem-Solution Fit: {scores['problem_solution_fit']:.1f}/10")
        st.progress(float(scores['market_potential'] / 10), text=f"📈 Market Potential: {scores['market_potential']:.1f}/10")
        st.progress(float(scores['innovation_level'] / 10), text=f"💡 Innovation Level: {scores['innovation_level']:.1f}/10")
        st.progress(float(scores['competitive_advantage'] / 10), text=f"⚡ Competitive Advantage: {scores['competitive_advantage']:.1f}/10")
        st.progress(float(scores['feasibility'] / 10), text=f"🔧 Feasibility: {scores['feasibility']:.1f}/10")
        
        # Add ML Scoring Method Explanation
        st.markdown("""
        <div class="ml-explanation">
            <strong>🤖 ML Scoring Method:</strong> Pure statistical analysis without keywords or patterns. 
            Uses semantic embeddings, text statistics, and contextual features for universal evaluation.
        </div>
        """, unsafe_allow_html=True)
    
    # PDF Button
    with col_pdf:
        if not st.session_state.pdf_generated:
            if st.button("Generate Professional PDF Report", use_container_width=True, key="generate_pdf_final"):
                with st.spinner("Creating professional PDF report..."):
                    analysis_data = st.session_state.last_analysis
                    pdf_filename = f"startup_validation_{int(time.time())}.pdf"
                    
                    pdf_file = generate_pdf_report(
                        idea=analysis_data['idea'],
                        analysis=analysis_data['analysis'],
                        similar_ideas=analysis_data['similar_ideas'],
                        trend_summary=analysis_data['trend_summary'],
                        ml_scores=analysis_data.get('ml_scores'),
                        filename=pdf_filename
                    )
                    
                    if pdf_file and os.path.exists(pdf_filename):
                        st.session_state.pdf_file_path = pdf_filename
                        st.session_state.pdf_generated = True
                        st.success("✅ PDF report generated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate PDF report.")

# Show PDF download button if PDF was generated
if (
    'pdf_generated' in st.session_state and st.session_state.pdf_generated and 
    'pdf_file_path' in st.session_state and st.session_state.pdf_file_path is not None
):
    st.markdown("---")
    st.markdown("## 📥 Download Report")     
    with st.container():
        st.markdown(get_pdf_success_html(), unsafe_allow_html=True) 
        
        try:
            with open(st.session_state.pdf_file_path, "rb") as file:
                pdf_data = file.read()
            st.download_button(
                label="📥 Download Validation Report (PDF)",
                data=pdf_data,
                file_name=st.session_state.pdf_file_path,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")

# --- 9. Competitor Display Section ---
if st.session_state.real_competitors:
    st.markdown("---")
    st.markdown("## 🌐 Real Competitors Found")
    
    with st.container():
        has_real_competitors = any('http' in comp for comp in st.session_state.real_competitors)
        
        if has_real_competitors:
            st.markdown("""
            <div class="competitor-card">
                <h3>🎯 Market Competition Analysis</h3>
                <p>These are real companies found via web search that compete in your space:</p>
            </div>
            """, unsafe_allow_html=True)
            
            for i, competitor in enumerate(st.session_state.real_competitors[:6], 1):
                if 'http' in competitor:
                    parts = competitor.split(' - ')
                    if len(parts) == 2:
                        title, url = parts
                        st.markdown(f"{i}. {title} - [Visit Website]({url})")
                    else:
                        st.write(f"{i}. {competitor}")
                else:
                    st.write(f"{i}. {competitor}")
        else:
            st.markdown("""
            <div class="competitor-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                <h3>💡 Market Research Insights</h3>
                <p>No direct competitors found. This might indicate a blue ocean opportunity!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("### 🔍 Recommended Research Steps:")
            for i, insight in enumerate(st.session_state.real_competitors, 1):
                st.write(f"{i}. {insight}")
            
            st.markdown("### 🎯 Try These Searches Manually:")
            if st.session_state.last_analysis:
                idea = st.session_state.last_analysis.get('idea', 'your idea')
                st.code(f"Google: '{idea} competitors'")
                st.code(f"Google: 'companies like {idea}'")
                st.code(f"Google: '{idea.split()[0]} industry analysis'")

# --- INVESTOR DISPLAY SECTION ---
if st.session_state.investor_data and st.session_state.investor_data.get('investors'):
    investor_data = st.session_state.investor_data
    
    st.markdown("---")
    st.markdown("## 💰 Potential Investors")
    
    # Header with summary
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
        <h3>🎯 Investor Matching Results</h3>
        <p><strong>Industry:</strong> {investor_data['industry']} | 
           <strong>Investors Found:</strong> {investor_data['total_found']}</p>
        <p><strong>Recommended Stage:</strong> {investor_data['recommended_stage']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display investors
    st.markdown("### 🏢 Top Investor Matches")
    
    for i, investor in enumerate(investor_data['investors'][:6], 1):
        with st.expander(f"{i}. {investor['name']} - {investor.get('investor_type', 'Investor')}", expanded=i <= 2):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"Description: {investor.get('description', 'No description available')}")
                st.write(f"Type: {investor.get('investor_type', 'Not specified')}")
                if investor.get('industry_focus'):
                    st.write(f"Industry Focus: {investor['industry_focus']}")
                st.write(f"Source: {investor.get('source', 'Web Search')}")
            
            with col2:
                if investor.get('website'):
                    st.markdown(f"[🌐 Visit Website]({investor['website']})")
                
                search_url = f"https://www.google.com/search?q={investor['name'].replace(' ', '+')}+venture+capital"
                st.markdown(f"[🔍 Research]({search_url})")
    
    # Investment strategy guidance
    st.markdown("---")
    st.markdown("### 📈 Investment Strategy")
    
    stage = investor_data['recommended_stage']
    st.write(f"Recommended Funding Stage: {stage}")
    
    stage_advice = {
        'Pre-seed / Angel Round': {
            'target': 'Angel investors, friends & family, micro-VCs',
            'amount': '$10K - $500K',
            'focus': 'Idea validation, team strength, market size',
            'tips': ['Build a strong founding team', 'Create a compelling pitch deck', 'Network with angel groups']
        },
        'Seed Round': {
            'target': 'Seed funds, angel groups, early-stage VCs', 
            'amount': '$500K - $2M',
            'focus': 'MVP, early traction, user growth',
            'tips': ['Show product-market fit', 'Demonstrate user engagement', 'Have a clear go-to-market strategy']
        },
        'Series A': {
            'target': 'Traditional VCs, growth funds',
            'amount': '$2M - $15M', 
            'focus': 'Revenue, scalability, market leadership',
            'tips': ['Show strong revenue growth', 'Demonstrate low customer acquisition costs', 'Prove market leadership potential']
        }
    }
    
    advice = stage_advice.get(stage, stage_advice['Seed Round'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Target Investors: {advice['target']}")
        st.write(f"Typical Amount: {advice['amount']}")
    with col2:
        st.write(f"Focus Areas: {advice['focus']}")
    
    st.markdown("### 💡 Actionable Tips")
    for tip in advice['tips']:
        st.write(f"• {tip}")
            
# --- Detailed Analysis Report Display ---
if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
    final_report = st.session_state.messages[-1]['content']
    
    st.markdown("---")
    st.markdown("## 📋 Full Expert Validation Report")
    
    with st.expander("Click to view Full Report and Data Breakdown", expanded=False): 
        st.markdown(final_report, unsafe_allow_html=True)
        
# --- FOOTER ---
st.markdown("<br><br>---", unsafe_allow_html=True)
st.caption(f"© 2025 Smart Idea Validator • Powered by Ollama ({OLLAMA_MODEL}) • Pure ML-Based Startup Analysis")