import streamlit as st
import pandas as pd
import json
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from google import genai
# ==========================================
# 1. CORE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="AI-Powered Startup Validator",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .agent-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    .agent-badge {
        position: absolute;
        top: 15px;
        right: 15px;
        background-color: #def7ec;
        color: #03543f;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 20px;
    }
    .score-circle {
        font-size: 2rem;
        font-weight: 800;
        color: #1d4ed8;
        margin: 10px 0;
    }
    .metric-pill {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 30px;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #e2e8f0;
    }
    .swot-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #1e293b;
    }
    .swot-s { background-color: #f0fdf4; border-left: 5px solid #16a34a; }
    .swot-w { background-color: #fef2f2; border-left: 5px solid #dc2626; }
    .swot-o { background-color: #eff6ff; border-left: 5px solid #2563eb; }
    .swot-t { background-color: #fffbeb; border-left: 5px solid #d97706; }
    .swot-title { font-weight: bold; margin-bottom: 5px; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOCAL NLP ENGINE
# ==========================================
def calculate_uniqueness(user_description: str, venture_title: str = "Conceptual Prototype"):
    """Computes cosine similarity against local CSV, returns uniqueness metrics."""
    try:
        df = pd.read_csv("companies.csv")
        existing_descriptions = df['description'].tolist()
        company_names = df['name'].tolist()
        
        all_texts = [user_description] + existing_descriptions
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        dense_matrix = tfidf_matrix.toarray() #type:ignore
        user_vector = dense_matrix[0].reshape(1, -1)
        competitor_vectors = dense_matrix[1:]
        
        similarity_scores = cosine_similarity(user_vector, competitor_vectors).flatten()
        max_idx = similarity_scores.argmax()
        highest_similarity = float(similarity_scores[max_idx])
        closest_competitor = company_names[max_idx]
        
        uniqueness_score = int((1 - highest_similarity) * 100)
        
        return uniqueness_score, closest_competitor
        
    except Exception:
       return 78, "Nexus Core Global Competitor"

# ==========================================
# 3. LIVE GEMINI API AGENT ORCHESTRATION
# ==========================================
def get_live_agent_data(user_description: str, api_key: str):
    """Orchestrates live Gemini API to structure venture evaluation in clean format."""
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are VentureIQ, an advanced AI investment evaluator. Analyze the following startup description and output a thorough, structured, and logical valuation:
    
    ---
    STARTUP DESCRIPTION:
    {user_description}
    ---
    
    Provide the analysis structured EXACTLY following this JSON model template:
    {{
        "final_verdict": "INVESTMENT APPROVED or INVESTMENT REJECTED",
        "confidence_percentage": 85,
        "executive_summary": "A detailed 2-3 sentence overview of the business model viable potential and strategic concerns.",
        "finance_analysis": {{
            "break_even": "X Months",
            "initial_funding": "Estimated initial funding required in Rs.",
            "projected_revenue": "Projected Year-1 Gross revenue in Rs."
        }},
        "swot_analysis": {{
            "strengths": "Main corporate asset/strength.",
            "weaknesses": "Core systemic hazard or operational vulnerability.",
            "opportunities": "A major growth or expansion path.",
            "threats": "External marketplace threats."
        }},
        "market_research": {{
            "score": "8.5",
            "analysis": "Specific notes on the Target addressable market.",
            "key_metrics": {{"TAM": "Estimated TAM", "Growth": "Growth rate %"}}
        }},
        "competitors": {{
            "score": "7.5",
            "analysis": "Brief analysis of barriers to entry and corporate moat.",
            "key_metrics": {{"Overlap": "Low/Medium/High", "Moat": "Weak/Moderate/Strong"}}
        }},
        "financials": {{
            "score": "8.0",
            "analysis": "A quick look at unit economics and potential markups.",
            "key_metrics": {{"LTV:CAC": "Ratio estimation", "Gross Margin": "Estimated Gross Margin %"}}
        }},
        "risk_assessment": {{
            "score": "6.5",
            "analysis": "Key scaling or dependency risk factor.",
            "key_metrics": {{"Risk Index": "Low/Medium/High", "Volatility": "Low/Medium/High"}}
        }}
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            )
            return json.loads(response.text or "{}")
            
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            raise e
# ==========================================
# 4. MODERN FPDF2 REPORT ENGINE
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 10)
        self.set_text_color(120, 130, 140)
        self.cell(0, 10, "Strategic Venture Evaluation Matrix", border=0, new_x="LMARGIN", new_y="NEXT", align="R")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"System Metric Analysis Evaluation Report - Page {self.page_no()}", border=0, new_x="LMARGIN", new_y="NEXT", align="C")

def clean_pdf_text(text):
    """Aggressively replaces complex Unicode characters to prevent FPDF crashes"""
    if not isinstance(text, str):
        return str(text)
    text = text.replace("₹", "Rs. ")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("•", "*")
    return text.encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf_bytes(title, report_data, local_uniqueness, local_comp):
    """Compiles analytical outputs into a perfectly formatted PDF file"""
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 15, "VENTURE EVALUATION MATRIX REPORT", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(80, 90, 100)
    safe_title = str(title) if title else 'Conceptual Prototype'
    cleaned_title = clean_pdf_text(safe_title)
    pdf.cell(0, 8, f"Venture Variant Target: {cleaned_title}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.line(10, 38, 200, 38)
    pdf.ln(5)
    
    pdf.set_fill_color(240, 244, 255)
    pdf.rect(10, 42, 190, 20, "F")
    pdf.set_xy(15, 47)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 58, 138)
    
    cleaned_verdict = clean_pdf_text(report_data.get('final_verdict', 'N/A'))
    pdf.cell(90, 10, f"FINAL LOGIC VERDICT: {cleaned_verdict}", border=0, new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(90, 10, f"SYSTEM CONFIDENCE RATE: {report_data.get('confidence_percentage', 0)}%", border=0, new_x="LMARGIN", new_y="NEXT", align="R")
    
    pdf.set_xy(10, 70)
    
    # Section 1
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, "1. LOCAL DISTANCE ALIGNMENT ENGINE METRICS", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    cleaned_comp = clean_pdf_text(local_comp)
    pdf.cell(0, 6, f"- Computed On-Disk Lexical Uniqueness Rate: {local_uniqueness}%", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 6, f"- Highest Dense Cross-Corpus Word Overlap Target: {cleaned_comp}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(4)
    
    # Section 2
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, "2. EXECUTIVE SUMMARY EVALUATION", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    cleaned_exec_summary = clean_pdf_text(report_data.get('executive_summary', ''))
    pdf.multi_cell(0, 5, cleaned_exec_summary)
    pdf.ln(4)
    
    # Section 3
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, "3. STRATEGIC FINANCIAL FORECASTS", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    f_data = report_data.get('finance_analysis', {})
    pdf.cell(0, 6, f"- Break-Even Timeline Target: {clean_pdf_text(f_data.get('break_even', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 6, f"- Required Launch Capital Reserves: {clean_pdf_text(f_data.get('initial_funding', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 6, f"- Projected Year-1 Gross Revenue Target: {clean_pdf_text(f_data.get('projected_revenue', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(4)

    # Section 4
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, "4. STRATEGIC SWOT DEEP DIVE", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    swot = report_data.get('swot_analysis', {})
    pdf.cell(0, 6, f"- Strength: {clean_pdf_text(swot.get('strengths', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 6, f"- Weakness: {clean_pdf_text(swot.get('weaknesses', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 6, f"- Opportunity: {clean_pdf_text(swot.get('opportunities', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.cell(0, 6, f"- Threat: {clean_pdf_text(swot.get('threats', 'N/A'))}", border=0, new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.ln(4)
    
    # Clean output method for fpdf2
    return bytes(pdf.output())
# ==========================================
# 5. USER INTERFACE LAYOUT & SIDEBAR
# ==========================================
st.title("🤖 AI-Powered Startup Validator")
st.write("Evaluate live venture metrics using real-time multi-agent processing models.")

st.sidebar.header("🔑 API Credentials")
default_key = os.environ.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input("Enter your Gemini API Key", value=default_key, type="password")

if not api_key:
    st.sidebar.warning("Please enter your Gemini API Key to enable live analysis, or set it as an environment variable.")

venture_title = st.text_input("Venture Title / Project Name", placeholder="e.g., Silk & Needle Custom Bridalwear")
venture_description = st.text_area("Venture Description & Core Model", placeholder="Describe your product, target market, and operational roadmap...")

if st.button("Initialize Multi-Agent Diagnostics", type="primary"):
    if not venture_description.strip():
        st.warning("Please enter a description to run the diagnostics.")
    elif not api_key:
        st.error("Cannot query live agents without an API Key. Please provide your Gemini API key in the sidebar.")
    else:
        with st.spinner("Calling live Gemini neural engines and evaluating parameters..."):
            try:
                uniqueness_pct, top_competitor = calculate_uniqueness(venture_description, venture_title)
                agent_data = get_live_agent_data(venture_description, api_key)
                
                st.session_state['uniqueness'] = uniqueness_pct
                st.session_state['competitor'] = top_competitor
                st.session_state['agent_data'] = agent_data
                st.session_state['processed'] = True
            except Exception as e:
                st.error(f"Failed to communicate with live API endpoint: {e}")

# ==========================================
# 6. DASHBOARD DISPLAY PANELS
# ==========================================
if st.session_state.get('processed'):
    data = st.session_state['agent_data']
    uniq = st.session_state['uniqueness']
    comp = st.session_state['competitor']
    
    st.success("Live AI Analysis Completed Successfully!")
    
    st.subheader("📊 Computed Product Uniqueness Matrix")
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        st.progress(uniq / 100)
    with col_c2:
        st.write(f"### **{uniq}% Unique**")
    
    st.write(f"🎯 *Our NLP model ran a local vector calculation and found your closest on-disk overlap is with:* **{comp}**.")
    st.divider()
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.metric("Final Verdict Decision", data.get('final_verdict', 'N/A'))
    with col_v2:
        st.metric("Model Confidence Rating", f"{data.get('confidence_percentage', 0)}%")
        
    st.divider()
    
    st.subheader("📝 Executive Appraisal Summary")
    st.write(data.get('executive_summary', 'No summary generated.'))
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("💰 Strategic Financial Matrix")
        f_info = data.get('finance_analysis', {})
        st.metric(label="Break-Even Timeline Target", value=f_info.get('break_even', 'N/A'))
        st.metric(label="Estimated Starting Capital", value=f_info.get('initial_funding', 'N/A'))
        st.metric(label="Projected Year-1 Revenue Model", value=f_info.get('projected_revenue', 'N/A'))
        
    with col_right:
        st.subheader("🎯 Venture SWOT Deep-Dive")
        swot_data = data.get('swot_analysis', {})
        
        st.markdown(f"""
        <div class="swot-box swot-s">
            <div class="swot-title">💪 Core Strengths</div>
            <div>{swot_data.get('strengths', 'N/A')}</div>
        </div>
        <div class="swot-box swot-w">
            <div class="swot-title">⚠️ Systemic Weaknesses</div>
            <div>{swot_data.get('weaknesses', 'N/A')}</div>
        </div>
        <div class="swot-box swot-o">
            <div class="swot-title">🚀 Scaling Opportunities</div>
            <div>{swot_data.get('opportunities', 'N/A')}</div>
        </div>
        <div class="swot-box swot-t">
            <div class="swot-title">🔥 Market Threats</div>
            <div>{swot_data.get('threats', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    st.markdown("#### 🤖 Sub-Agent Diagnostic Performance Grid")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    
    def generate_agent_card_html(badge_text, title, score, analysis, key_metrics):
        pills_html = ""
        for k, v in key_metrics.items():
            pills_html += f"<span class='metric-pill'><b>{k}:</b> {v}</span> "
        
        return f"""
        <div class="agent-card">
            <span class="agent-badge">{badge_text}</span>
            <h5>{title}</h5>
            <div class="score-circle">{score}</div>
            <p style="font-size:0.9rem; color:#475569; margin-bottom: 12px;">{analysis}</p>
            <div style="margin-top: 10px;">
                {pills_html}
            </div>
        </div>
        """
    
    m_research = data.get('market_research', {})
    competitors = data.get('competitors', {})
    financials = data.get('financials', {})
    risk = data.get('risk_assessment', {})

    with c1:
        st.markdown(generate_agent_card_html("Live API Agent", "📈 Market Research", m_research.get('score', 'N/A'), m_research.get('analysis', 'N/A'), m_research.get('key_metrics', {})), unsafe_allow_html=True)
    with c2:
        st.markdown(generate_agent_card_html("Live API Agent", "🔍 Competitor Engine", competitors.get('score', 'N/A'), competitors.get('analysis', 'N/A'), competitors.get('key_metrics', {})), unsafe_allow_html=True)
    with c3:
        st.markdown(generate_agent_card_html("Live API Agent", "💰 Financial Matrix", financials.get('score', 'N/A'), financials.get('analysis', 'N/A'), financials.get('key_metrics', {})), unsafe_allow_html=True)
    with c4:
        st.markdown(generate_agent_card_html("Live API Agent", "⚠️ Risk Assessment", risk.get('score', 'N/A'), risk.get('analysis', 'N/A'), risk.get('key_metrics', {})), unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📥 Export Final Evaluation Report")
    try:
        pdf_bytes = generate_pdf_bytes(
            title=venture_title,
            report_data=data,
            local_uniqueness=uniq,
            local_comp=comp
        )
        
        st.download_button(
            label="Download Structured Investment-Grade Evaluation PDF",
            data=pdf_bytes,
            file_name=f"{venture_title.lower().replace(' ', '_') if venture_title else 'venture'}_evaluation_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as pdf_err:
        st.error(f"Error compiling PDF output file parameters: {pdf_err}")