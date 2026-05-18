"""
Socrates — Loquere ut te videam
Streamlit Web Application for CEO Personality Analysis.

Flow:
  1. Disclaimer popup on first visit
  2. Upload transcript (PDF/DOCX)
  3. Parse & identify CEO → confirmation dialog
  4. Run prediction → display results
"""

import os
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

from config import APP_TITLE, APP_SUBTITLE, APP_ICON, TRAITS, TRAIT_LABELS
from src.modeling.predictor import Predictor
from src.visualization.radar_chart import create_radar_chart
from src.visualization.percentile_bands import create_percentile_table, style_percentile_dataframe
from src.visualization.pdf_report import generate_pdf_report
from src.ingestion.transcript_parser import (
    extract_text_from_pdf,
    identify_speakers,
    find_senior_executive,
    clean_financial_text,
    _name_matches,
)


# Page Config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,500;1,500&display=swap');

  
  .main .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { font-family: 'Inter', sans-serif; }

  
  .hero-title {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 2.6rem;
      font-weight: 500;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0;
      line-height: 1.2;
  }
  .hero-tagline {
      font-family: 'Playfair Display', Georgia, serif;
      font-style: italic;
      font-size: 1.15rem;
      color: #94a3b8;
      margin-top: 0;
      letter-spacing: 0.04em;
  }

  
  .disclaimer-box {
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
      border: 1px solid #4338ca;
      border-radius: 12px;
      padding: 2rem;
      color: #e0e7ff;
      font-size: 0.95rem;
      line-height: 1.7;
      margin-bottom: 1rem;
  }
  .disclaimer-box strong { color: #a5b4fc; }

  
  .confirm-card {
      background: linear-gradient(135deg, #0c4a6e 0%, #164e63 100%);
      border: 1px solid #0e7490;
      border-radius: 12px;
      padding: 1.5rem 2rem;
      color: #e0f2fe;
      margin: 1rem 0;
  }
  .confirm-card .ceo-name {
      font-size: 1.4rem;
      font-weight: 700;
      color: #7dd3fc;
  }
  .confirm-card .match-tier {
      font-size: 0.85rem;
      color: #67e8f9;
      margin-top: 0.25rem;
  }

  
  .metric-card {
      background-color: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 0.5rem;
      padding: 1rem;
      text-align: center;
  }
  .metric-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: #0f172a;
  }
  .metric-label {
      font-size: 0.875rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }

  
  .upload-section {
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
      transition: border-color 0.3s ease;
  }
  .upload-section:hover { border-color: #818cf8; }
</style>
""", unsafe_allow_html=True)


# Cached Resources
@st.cache_resource
def load_predictor():
    """Cache the predictor loading since models are large."""
    predictor = Predictor()
    success = predictor._load_models()
    return predictor if success else None


# Session State Defaults
def init_session_state():
    defaults = {
        "disclaimer_accepted": False,
        "parse_done": False,
        "ceo_confirmed": False,
        "identified_ceo": "",
        "match_tier": "",
        "all_turns": [],
        "all_speakers": [],
        "speaker_word_counts": {},
        "ceo_speech": "",
        "word_count": 0,
        "transcript_summary": {},
        "last_result": None,
        "uploaded_file_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_analysis():
    """Clear analysis state when a new file is uploaded."""
    keys = [
        "parse_done", "ceo_confirmed", "identified_ceo", "match_tier",
        "all_turns", "all_speakers", "speaker_word_counts", "ceo_speech",
        "word_count", "transcript_summary", "last_result",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()


# Parsing Helper
def parse_uploaded_file(tmp_path: str):
    """
    Parse the uploaded file and identify the senior executive.
    Uses management-hierarchy approach: first listed = most senior.
    """
    full_text = extract_text_from_pdf(tmp_path, skip_first_page=False)
    if not full_text:
        st.error("❌ Could not extract text from the uploaded file.")
        return False

    turns = identify_speakers(full_text)
    if not turns:
        st.error("❌ No speaker turns detected in this transcript.")
        return False

    unique_speakers = list(set(t.speaker_name for t in turns))

    # Find senior executive from participant listing
    exec_name, exec_title = find_senior_executive(full_text)

    # Match executive to speaker turns
    exec_turns = []
    if exec_name:
        for t in turns:
            if _name_matches(exec_name, t.speaker_name):
                exec_turns.append(t)

    # Compute per-speaker word counts
    speaker_wc = {}
    for t in turns:
        speaker_wc[t.speaker_name] = speaker_wc.get(t.speaker_name, 0) + t.word_count

    # Store in session state
    st.session_state.all_turns = turns
    st.session_state.all_speakers = unique_speakers
    st.session_state.speaker_word_counts = speaker_wc
    st.session_state.identified_ceo = exec_name
    st.session_state.match_tier = f"First listed under management ({exec_title})" if exec_title else "First listed under management"

    if exec_name and exec_turns:
        raw_speech = " ".join(t.text for t in exec_turns)
        speech = clean_financial_text(raw_speech)
        st.session_state.ceo_speech = speech
        st.session_state.word_count = len(speech.split())

    st.session_state.parse_done = True
    return True


def extract_speech_for_name(target_name: str):
    """Find turns matching a manually entered name."""
    turns = st.session_state.all_turns
    matched_turns = []
    for t in turns:
        if _name_matches(target_name, t.speaker_name):
            matched_turns.append(t)

    if matched_turns:
        speech = " ".join(t.text for t in matched_turns)
        st.session_state.ceo_speech = speech
        st.session_state.word_count = len(speech.split())
        st.session_state.identified_ceo = target_name
        return True
    return False


# UI Sections

def render_disclaimer():
    """Show the research disclaimer popup."""
    st.markdown('<p class="hero-title">🏛️ Socrates</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-tagline">Loquere ut te videam — "Speak, so that I may see you"</p>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Research Disclaimer</strong><br><br>
        This application has been built <strong>as a project for research purposes only</strong>.
        The personality assessments generated by this tool are based on automated linguistic analysis
        of publicly available earnings call transcripts using proxy-labelled machine learning models.<br><br>
        The results represent <strong>relative stylistic tendencies</strong> within the analysed sample
        and are <strong>not</strong> clinically validated psychological assessments.<br><br>
        <strong>The outputs are not suitable to drive real-world decisions</strong> including but not
        limited to hiring, investing, or any form of individual evaluation.
    </div>
    """, unsafe_allow_html=True)

    if st.button("I Understand — Proceed", type="primary", use_container_width=True):
        st.session_state.disclaimer_accepted = True
        st.rerun()


def render_sidebar(predictor):
    """Render the sidebar."""
    with st.sidebar:
        st.markdown('<p class="hero-title" style="font-size:1.8rem;">🏛️ Socrates</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-tagline" style="font-size:0.9rem;">Loquere ut te videam</p>', unsafe_allow_html=True)
        st.divider()

        page = st.radio("Navigation", ["Analyse", "About"], label_visibility="collapsed")

        st.divider()
        st.markdown("**System Status**")
        if predictor:
            st.success("Models Loaded ✅")
        else:
            st.error("Models Not Found ❌")

        st.divider()
        st.caption("Built for academic research only.")

    return page


def render_analysis(predictor: Predictor):
    """Main analysis page with the multi-step flow."""
    st.markdown('<p class="hero-title">Analyse Transcript</p>', unsafe_allow_html=True)
    st.markdown("Upload an earnings call transcript to estimate the CEO's Big Five personality profile.")
    st.markdown("")

    if not predictor:
        st.warning("⚠️ The ML models are not loaded. Please train models first using `run_training.py`.")
        return

    # Step 1: Upload
    uploaded_file = st.file_uploader(
        "Upload Earnings Call Transcript",
        type=["pdf", "docx"],
        help="Supported formats: PDF, DOCX",
        key="file_uploader",
    )

    # Reset state if a new file is uploaded
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get("uploaded_file_name", ""):
            reset_analysis()
            st.session_state.uploaded_file_name = uploaded_file.name

    if uploaded_file is None:
        st.session_state.uploaded_file_name = ""
        return

    # Step 2: Parse
    if not st.session_state.parse_done:
        if st.button("📄 Parse Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text and identifying speakers..."):
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    success = parse_uploaded_file(tmp_path)
                    if success:
                        st.rerun()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        return

    # Step 3: Confirm or manually specify the target executive
    if st.session_state.parse_done and not st.session_state.ceo_confirmed:
        identified = st.session_state.identified_ceo

        if identified:
            tier_label = st.session_state.match_tier

            st.markdown(f"""
            <div class="confirm-card">
                I have identified <span class="ceo-name">{identified}</span>
                as the chief managerial executive of this transcript.
                <div class="match-tier">Detection: {tier_label} &nbsp;•&nbsp; {st.session_state.word_count:,} words of speech extracted</div>
            </div>
            """, unsafe_allow_html=True)

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Proceed with this identification", type="primary", use_container_width=True):
                    st.session_state.ceo_confirmed = True
                    st.rerun()
            with col_no:
                if st.button("No, let me specify manually", use_container_width=True):
                    st.session_state.identified_ceo = ""
                    st.rerun()
        else:
            st.warning("Could not automatically identify the chief managerial executive.")
            st.markdown("**Type the name of the executive you want to analyse:**")
            st.caption("The parser will fuzzy-match this against all speakers detected in the transcript.")

            typed_name = st.text_input(
                "Executive name",
                placeholder="e.g. Rohit Jawa",
                key="manual_name_input",
                label_visibility="collapsed",
            )

            if typed_name.strip() and st.button("Proceed", type="primary", use_container_width=True):
                found = extract_speech_for_name(typed_name.strip())
                if found:
                    speech = clean_financial_text(st.session_state.ceo_speech)
                    st.session_state.ceo_speech = speech
                    st.session_state.word_count = len(speech.split())
                    st.session_state.ceo_confirmed = True
                    st.rerun()
                else:
                    st.error(
                        f"No speaker matching \"{typed_name.strip()}\" was found in this transcript. "
                        "Try using the name as it appears in the document."
                    )
        return

    # Step 4: Run Prediction
    if st.session_state.ceo_confirmed and st.session_state.last_result is None:
        with st.spinner("🧠 Running Big Five personality analysis..."):
            ceo_name = st.session_state.identified_ceo
            speech = st.session_state.ceo_speech

            result = predictor.predict_from_text(
                ceo_name=ceo_name,
                speech_text=speech,
                summary={},
                warnings=[],
            )

            if result:
                st.session_state.last_result = result
                st.rerun()
            else:
                st.error("❌ Analysis failed. The model could not generate predictions.")
                return

    # Step 5: Display Results
    if st.session_state.last_result is not None:
        render_results(st.session_state.last_result)


def render_results(res):
    """Display prediction results."""
    st.divider()

    # Header
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"### 🎯 {res.ceo_name}")
    with col_info2:
        st.metric("Words Analysed", f"{res.word_count:,}")

    # Warnings
    for w in res.warnings:
        st.warning(w)

    st.markdown("---")

    # Chart + Table
    col_chart, col_data = st.columns([1.2, 1])

    with col_chart:
        st.subheader("Personality Profile")
        fig = create_radar_chart(res.scores, ceo_name=res.ceo_name)
        st.plotly_chart(fig, use_container_width=True)

    with col_data:
        st.subheader("Trait Scores")
        df = create_percentile_table(res.scores, res.percentiles)
        styled_df = style_percentile_dataframe(df)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # PDF Report
        try:
            pdf_bytes = generate_pdf_report(
                ceo_name=res.ceo_name,
                company_name="Transcript Analysis",
                word_count=res.word_count,
                scores=res.scores,
                percentiles=res.percentiles,
                radar_fig=fig,
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{res.ceo_name.replace(' ', '_')}_Profile.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"PDF generation unavailable: {e}")

    # Trait interpretation
    with st.expander("📊 Score Interpretation Guide"):
        st.markdown("""
        | Score Range | Meaning |
        |---|---|
        | **1.0 – 2.5** | Very Low — significantly below sample average |
        | **2.5 – 3.5** | Low — below average |
        | **3.5 – 4.5** | Average — typical for the sample |
        | **4.5 – 5.5** | High — above average |
        | **5.5 – 7.0** | Very High — significantly above sample average |

        *Scores reflect relative positioning within the training corpus of ~360 Indian CEOs.*
        """)


def render_about():
    """About / methodology page."""
    st.markdown('<p class="hero-title">About Socrates</p>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
    ### Methodology

    **Socrates** estimates the Big Five personality traits (OCEAN) of corporate executives
    based on their spoken language during earnings conference calls.

    ### Pipeline

    1. **Transcript Parsing** — PDFs are parsed using PyMuPDF. A 3-tier speaker identification
       cascade (exact name → title-based → opening statement) isolates the target executive's speech.
    2. **Semantic Embedding** — The CEO's speech is encoded into a 384-dimensional vector using
       the `all-MiniLM-L6-v2` Sentence Transformer, capturing meaning and style.
    3. **Prediction** — Five independent XGBoost regressors (one per trait) map the embeddings
       to scores on a 1–7 scale.

    ### Ground Truth & Limitations

    This system replicates the methodology of Harrison et al. (2019, *Strategic Management Journal*).
    Because proprietary psychological ratings are unavailable, a proxy-labelling approach based on
    Mairesse et al. (2007) is used. Outputs represent **relative linguistic positioning** within the
    sample, not clinical assessments.

    ### References

    - Harrison, J. S. et al. (2019). "Measuring CEO personality." *Strategic Management Journal*.
    - Mairesse, F. et al. (2007). "Using Linguistic Cues for Automatic Recognition of Personality." *JAIR*.

    ---

    *"Loquere ut te videam" — Speak, so that I may see you. — attributed to Socrates*
    """)


# Main

def main():
    init_session_state()

    # Disclaimer gate
    if not st.session_state.disclaimer_accepted:
        render_disclaimer()
        return

    predictor = load_predictor()
    page = render_sidebar(predictor)

    if page == "Analyse":
        render_analysis(predictor)
    else:
        render_about()


if __name__ == "__main__":
    main()
