"""
Theme & Visual Design System for AI Exam Proctoring Portal.
Provides global CSS styling, glassmorphism cards, cyber-AI aesthetics,
and responsive layout enhancements across all 7 wizard steps.
"""

import textwrap
import streamlit as st


def inject_custom_theme():
    """
    Injects global CSS styles, custom fonts, glassmorphism panel classes,
    cyber-AI glow aesthetics, and responsive layout rules into Streamlit.
    """
    css = textwrap.dedent("""
        <style>
        /* ------------------------------------------------------------------ */
        /* GOOGLE FONTS & GLOBAL ROOT STYLING                                 */
        /* ------------------------------------------------------------------ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #070A12 !important;
            background-image:
                radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(79, 172, 254, 0.07) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(13, 21, 39, 0.5) 0%, transparent 80%);
            background-attachment: fixed;
            color: #E2E8F0 !important;
        }

        header[data-testid="stHeader"] {
            background: rgba(7, 10, 18, 0.8) !important;
            backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid rgba(0, 242, 254, 0.12) !important;
        }

        .main .block-container {
            max-width: 1280px !important;
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }

        /* ------------------------------------------------------------------ */
        /* SIDEBAR STYLING                                                    */
        /* ------------------------------------------------------------------ */
        section[data-testid="stSidebar"] {
            background-color: rgba(10, 16, 28, 0.85) !important;
            backdrop-filter: blur(16px) !important;
            border-right: 1px solid rgba(0, 242, 254, 0.15) !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #00F2FE !important;
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: 0.5px;
        }

        /* ------------------------------------------------------------------ */
        /* TYPOGRAPHY                                                         */
        /* ------------------------------------------------------------------ */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            color: #F8FAFC !important;
            letter-spacing: 0.3px;
        }

        h1 { font-weight: 700 !important; }
        h2 { font-weight: 600 !important; }
        h3 { font-weight: 600 !important; }

        /* ------------------------------------------------------------------ */
        /* GLASSMORPHISM CARD COMPONENTS                                      */
        /* ------------------------------------------------------------------ */
        .cyber-card {
            background: rgba(13, 21, 39, 0.65) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(0, 242, 254, 0.18) !important;
            border-radius: 14px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.25rem !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .cyber-card:hover {
            border-color: rgba(0, 242, 254, 0.35) !important;
            box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.12), inset 0 0 0 1px rgba(0, 242, 254, 0.1) !important;
        }

        .cyber-card-glow {
            background: linear-gradient(135deg, rgba(13, 21, 39, 0.8) 0%, rgba(8, 14, 28, 0.9) 100%) !important;
            border: 1px solid rgba(0, 242, 254, 0.3) !important;
            box-shadow: 0 0 25px rgba(0, 242, 254, 0.15) !important;
        }

        /* ------------------------------------------------------------------ */
        /* AI BADGES & STATUS INDICATORS                                      */
        /* ------------------------------------------------------------------ */
        .ai-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-cyan {
            background: rgba(0, 242, 254, 0.12);
            color: #00F2FE;
            border: 1px solid rgba(0, 242, 254, 0.3);
        }

        .badge-green {
            background: rgba(0, 255, 157, 0.12);
            color: #00FF9D;
            border: 1px solid rgba(0, 255, 157, 0.3);
        }

        .badge-amber {
            background: rgba(255, 170, 0, 0.12);
            color: #FFAA00;
            border: 1px solid rgba(255, 170, 0, 0.3);
        }

        .badge-red {
            background: rgba(255, 75, 75, 0.12);
            color: #FF4B4B;
            border: 1px solid rgba(255, 75, 75, 0.3);
        }

        /* ------------------------------------------------------------------ */
        /* FORM CONTROLS & INPUT OVERRIDES                                    */
        /* ------------------------------------------------------------------ */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background-color: rgba(10, 16, 28, 0.7) !important;
            border: 1px solid rgba(0, 242, 254, 0.2) !important;
            color: #F1F5F9 !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.8rem !important;
            transition: all 0.25s ease !important;
        }

        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div:focus-within {
            border-color: #00F2FE !important;
            box-shadow: 0 0 12px rgba(0, 242, 254, 0.25) !important;
        }

        /* Buttons Overrides */
        .stButton > button {
            border-radius: 10px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            padding: 0.55rem 1.4rem !important;
        }

        /* Primary Action Button (Glowing Cyan Gradient) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
            color: #070A12 !important;
            border: none !important;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 28px rgba(0, 242, 254, 0.5) !important;
            color: #000000 !important;
        }

        /* Secondary Glass Button */
        .stButton > button[kind="secondary"] {
            background: rgba(13, 21, 39, 0.6) !important;
            color: #00F2FE !important;
            border: 1px solid rgba(0, 242, 254, 0.3) !important;
            backdrop-filter: blur(8px) !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background: rgba(0, 242, 254, 0.15) !important;
            border-color: #00F2FE !important;
            color: #FFFFFF !important;
        }

        /* ------------------------------------------------------------------ */
        /* TAB BAR OVERRIDES                                                  */
        /* ------------------------------------------------------------------ */
        div[data-baseweb="tab-highlight"] {
            background-color: #00F2FE !important;
            height: 3px !important;
            box-shadow: 0 0 10px #00F2FE !important;
        }

        button[data-baseweb="tab"] {
            background: transparent !important;
            color: #94A3B8 !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00F2FE !important;
            font-weight: 600 !important;
        }

        /* ------------------------------------------------------------------ */
        /* METRICS & STAT CARDS                                               */
        /* ------------------------------------------------------------------ */
        div[data-testid="stMetric"] {
            background: rgba(13, 21, 39, 0.6) !important;
            border: 1px solid rgba(0, 242, 254, 0.18) !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #00F2FE !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
        }

        /* ------------------------------------------------------------------ */
        /* PROGRESS BAR & EXPANDERS                                           */
        /* ------------------------------------------------------------------ */
        div[data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%) !important;
            box-shadow: 0 0 12px rgba(0, 242, 254, 0.5) !important;
            border-radius: 10px !important;
        }

        div[data-testid="stExpander"] {
            background: rgba(13, 21, 39, 0.5) !important;
            border: 1px solid rgba(0, 242, 254, 0.15) !important;
            border-radius: 12px !important;
        }

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(0, 242, 254, 0.18) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* Alert boxes (st.info, st.success, st.error, st.warning) */
        div.stAlert {
            background: rgba(13, 21, 39, 0.75) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 12px !important;
        }

        /* ------------------------------------------------------------------ */
        /* RESPONSIVE LAYOUT & MEDIA QUERIES (Mobile / Tablet)               */
        /* ------------------------------------------------------------------ */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
            }

            .cyber-card {
                padding: 1rem !important;
                border-radius: 10px !important;
            }

            .ai-sentinel-panel {
                margin-bottom: 1.5rem !important;
            }

            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
    """).strip()
    st.markdown(css, unsafe_allow_html=True)


def render_ai_sentinel_panel():
    """
    Renders a 3D Cybernetic AI Sentinel Graphic Panel for authentication & portal branding.
    Used in Step 1 as the left column visual panel (or stacked on mobile).
    Uses unindented dedented HTML strings to prevent CommonMark code block rendering.
    """
    panel_html = textwrap.dedent("""
        <div class="cyber-card cyber-card-glow ai-sentinel-panel" style="text-align: center; padding: 2rem 1.5rem;">
            <div style="position: relative; width: 140px; height: 140px; margin: 0 auto 1.5rem auto;">
                <svg viewBox="0 0 200 200" width="140" height="140" style="filter: drop-shadow(0 0 18px rgba(0,242,254,0.4));">
                    <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(0, 242, 254, 0.25)" stroke-width="2" stroke-dasharray="10 8" />
                    <circle cx="100" cy="100" r="78" fill="none" stroke="rgba(79, 172, 254, 0.4)" stroke-width="1.5" stroke-dasharray="14 6" />
                    <path d="M100 30 L155 55 V105 C155 145 100 175 100 175 C100 175 45 145 45 105 V55 Z" fill="url(#shieldGrad)" stroke="#00F2FE" stroke-width="2.5" />
                    <path d="M75 80 Q100 65 125 80 T100 120 Z" fill="none" stroke="#FFFFFF" stroke-width="2" opacity="0.85"/>
                    <circle cx="100" cy="95" r="8" fill="#00F2FE" />
                    <circle cx="75" cy="80" r="4" fill="#00FF9D" />
                    <circle cx="125" cy="80" r="4" fill="#00FF9D" />
                    <circle cx="100" cy="120" r="4" fill="#4FACFE" />
                    <defs>
                        <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="rgba(0, 242, 254, 0.35)"/>
                            <stop offset="100%" stop-color="rgba(13, 21, 39, 0.85)"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <h3 style="color: #00F2FE; margin-bottom: 0.4rem; font-size: 1.35rem;">AI Guardian Sentinel</h3>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.2rem; line-height: 1.4;">
                Autonomous Multi-Model Examination Proctoring & Real-Time Security Verification
            </p>
            <div style="display: flex; flex-direction: column; gap: 8px; text-align: left;">
                <div class="ai-badge badge-cyan" style="width: 100%; justify-content: flex-start;">
                    <span>🛡️</span> <span>4-CNN Ensemble Behavior Classifier</span>
                </div>
                <div class="ai-badge badge-green" style="width: 100%; justify-content: flex-start;">
                    <span>🎯</span> <span>Spatial YOLOv5 Object Localization</span>
                </div>
                <div class="ai-badge badge-amber" style="width: 100%; justify-content: flex-start;">
                    <span>⚖️</span> <span>2-of-4 Multi-Model Consensus Protocol</span>
                </div>
                <div class="ai-badge badge-cyan" style="width: 100%; justify-content: flex-start;">
                    <span>🖥️</span> <span>Parallel Dual Stream Monitoring</span>
                </div>
            </div>
        </div>
    """).strip()
    st.markdown(panel_html, unsafe_allow_html=True)
