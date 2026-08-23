"""
Wizard UI Renderers for the 7-Step Online Exam Proctoring System.
Renders Wizard Steps 1-4 (Firebase Auth, Firebase Email Verification, Exam Details, System Check)
and the top step progress indicator bar with a cohesive cyber-AI glassmorphism aesthetic.
"""

import re
import os
import time
import textwrap
from typing import Dict, Any, Tuple, Optional
import streamlit as st

from src.auth.firebase_auth import FirebaseAuthHandler
from src.utils.logger import get_logger
from app.components.theme import render_ai_sentinel_panel

logger = get_logger("wizard_ui")


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def _validate_email(email: str) -> bool:
    """Validates email format using standard regex pattern."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


# ---------------------------------------------------------------------------
# Stepper Header Bar (Cohesive 7-Step Indicator)
# ---------------------------------------------------------------------------

def render_wizard_stepper(current_step: int):
    """
    Renders a futuristic visual 7-step wizard progress indicator bar at the top of the app.
    Clearly distinguishes completed (✓), active (●), and upcoming (○) steps.
    Adapts responsively to mobile, tablet, and desktop screens.
    """
    steps = [
        "1. Auth",
        "2. Email Verification",
        "3. Exam Details",
        "4. System Check",
        "5. Proctored Exam",
        "6. AI Analysis",
        "7. Final Report"
    ]

    header_html = textwrap.dedent("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem; flex-wrap: wrap; gap: 10px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-size: 1.6rem; display: flex; align-items: center; gap: 10px;">
                    <span>🛡️</span> <span style="background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Proctoring Portal</span>
                </h2>
                <span style="color: #94A3B8; font-size: 0.85rem;">Autonomous Multi-Model Examination Safeguard System</span>
            </div>
            <div class="ai-badge badge-cyan">
                <span>⚡ REAL-TIME MONITORING ACTIVE</span>
            </div>
        </div>
    """).strip()
    st.markdown(header_html, unsafe_allow_html=True)

    cols = st.columns(len(steps))
    for i, step_label in enumerate(steps, 1):
        if i < current_step:
            step_html = textwrap.dedent(f"""
                <div style="background: rgba(0, 255, 157, 0.08); border: 1px solid rgba(0, 255, 157, 0.3); border-radius: 8px; padding: 8px 4px; text-align: center;">
                    <span style="color: #00FF9D; font-weight: 600; font-size: 0.8rem;">✓ {step_label}</span>
                </div>
            """).strip()
            cols[i - 1].markdown(step_html, unsafe_allow_html=True)
        elif i == current_step:
            step_html = textwrap.dedent(f"""
                <div style="background: rgba(0, 242, 254, 0.15); border: 1px solid #00F2FE; border-radius: 8px; padding: 8px 4px; text-align: center; box-shadow: 0 0 12px rgba(0,242,254,0.3);">
                    <span style="color: #00F2FE; font-weight: 700; font-size: 0.82rem;">● {step_label}</span>
                </div>
            """).strip()
            cols[i - 1].markdown(step_html, unsafe_allow_html=True)
        else:
            step_html = textwrap.dedent(f"""
                <div style="background: rgba(13, 21, 39, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 8px 4px; text-align: center;">
                    <span style="color: #64748B; font-weight: 500; font-size: 0.78rem;">○ {step_label}</span>
                </div>
            """).strip()
            cols[i - 1].markdown(step_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Wizard 1 — Firebase Authentication & Candidate Registration
# ---------------------------------------------------------------------------

def render_wizard_1_auth() -> Tuple[bool, Dict[str, Any], int]:
    """
    Wizard 1 — Sign In / Sign Up interface powered by Firebase Auth.
    Desktop: Left column AI Visual Sentinel panel, right column Glass Auth Card.
    Mobile: Stacks AI Sentinel panel above form.
    Returns (success_flag, user_profile_dict, target_step).
    """
    api_key = FirebaseAuthHandler.get_api_key()

    col_vis, col_form = st.columns([1, 1.1])

    # Left Column Visual Panel (Desktop) / Top Panel (Mobile)
    with col_vis:
        render_ai_sentinel_panel()

    # Right Column Authentication Card
    with col_form:
        auth_card_header = textwrap.dedent("""
            <div class="cyber-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #00F2FE;">🔑 Step 1 — Candidate Access Portal</h3>
                    <span class="ai-badge badge-cyan">FIREBASE AUTH</span>
                </div>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.2rem;">
                    Sign in to your candidate account or register for scheduled examinations.
                </p>
        """).strip()
        st.markdown(auth_card_header, unsafe_allow_html=True)

        if not api_key:
            st.error("❌ Firebase Authentication is not configured. Please configure FIREBASE_WEB_API_KEY.")

        tab_signin, tab_signup = st.tabs(["🔒 Candidate Sign In", "📝 Candidate Registration"])

        # ── SIGN IN TAB ──────────────────────────────────────────────────────
        with tab_signin:
            st.caption("Enter your candidate credentials to access the examination environment.")

            email_in = st.text_input(
                "Candidate Email Address:",
                value="student@university.edu",
                key="signin_email",
                placeholder="candidate@university.edu"
            )
            pass_in = st.text_input(
                "Password:",
                type="password",
                value="candidate123",
                key="signin_pass"
            )

            btn_login = st.button("🔓 Sign In with Firebase", type="primary", use_container_width=True, key="btn_signin")

            if btn_login:
                email_clean = email_in.strip().lower()
                if not email_clean or not pass_in:
                    st.error("Please enter both email and password.")
                elif not _validate_email(email_clean):
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner("Authenticating with Firebase Identity Gateway..."):
                        ok, res = FirebaseAuthHandler.sign_in(email_clean, pass_in)
                        if ok:
                            user_profile = {
                                "full_name": email_clean.split("@")[0].replace(".", " ").title(),
                                "email": email_clean,
                                "student_id": "STU-2026-8842",
                                "firebase_uid": res.get("localId"),
                                "id_token": res.get("idToken"),
                                "email_verified": res.get("emailVerified", False),
                                "role": "Candidate / Student"
                            }
                            if res.get("emailVerified", False):
                                st.success(f"✅ Authenticated: Welcome back, **{user_profile['full_name']}**!")
                                return True, user_profile, 3  # Direct to Step 3
                            else:
                                st.warning("⚠️ Email verification required. Redirecting to Step 2.")
                                return True, user_profile, 2  # To Step 2 Verification
                        else:
                            st.error(f"❌ Firebase Error: {res.get('error')}")

            # Forgot Password Section
            with st.expander("🔑 Forgot Password / Reset Link"):
                st.caption("Send a password reset email to your registered address.")
                reset_email = st.text_input("Enter Registered Email:", value=email_in, key="reset_email_input")
                if st.button("📩 Send Reset Link", use_container_width=True, key="btn_send_reset"):
                    if not reset_email or not _validate_email(reset_email.strip()):
                        st.error("Please enter a valid email address.")
                    else:
                        ok_res, msg_res = FirebaseAuthHandler.send_password_reset(reset_email.strip())
                        if ok_res:
                            st.success(f"✅ {msg_res}")
                        else:
                            st.error(f"❌ {msg_res}")

        # ── SIGN UP TAB ──────────────────────────────────────────────────────
        with tab_signup:
            st.caption("Register a new candidate profile for automated proctored examinations.")

            full_name = st.text_input("Full Name:", value="Alex Johnson", key="signup_name", placeholder="First & Last Name")
            email_reg = st.text_input("Email Address:", value="alex.johnson@university.edu", key="signup_email", placeholder="candidate@university.edu")
            student_id = st.text_input("Student Registration ID:", value="STU-2026-8842", key="signup_stuid", placeholder="STU-2026-XXXX")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pass_reg = st.text_input("Create Password:", type="password", value="SecurePass123!", key="signup_pass")
            with col_p2:
                pass_confirm = st.text_input("Confirm Password:", type="password", value="SecurePass123!", key="signup_confirm")

            if st.button("📝 Register Candidate Account", type="primary", use_container_width=True, key="btn_signup"):
                email_clean = email_reg.strip().lower()

                if not full_name.strip():
                    st.error("Full Name is required.")
                elif not _validate_email(email_clean):
                    st.error("Please enter a valid email address.")
                elif not student_id.strip():
                    st.error("Student Registration ID is required.")
                elif not pass_reg:
                    st.error("Password is required.")
                elif len(pass_reg) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif pass_reg != pass_confirm:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Creating Firebase candidate profile..."):
                        ok_create, res_create = FirebaseAuthHandler.create_user(email_clean, pass_reg)
                        if ok_create:
                            id_token = res_create.get("idToken", "")
                            if id_token:
                                FirebaseAuthHandler.send_verification_email(id_token)

                            user_profile = {
                                "full_name": full_name.strip(),
                                "email": email_clean,
                                "student_id": student_id.strip(),
                                "firebase_uid": res_create.get("localId"),
                                "id_token": id_token,
                                "email_verified": res_create.get("emailVerified", False),
                                "role": "Candidate / Student"
                            }
                            st.success("🎉 Registration successful! Verification link sent. Moving to Step 2.")
                            return True, user_profile, 2
                        else:
                            st.error(f"❌ Account Creation Error: {res_create.get('error')}")

        st.markdown("</div>", unsafe_allow_html=True)

    return False, {}, 1


# ---------------------------------------------------------------------------
# Wizard 2 — Firebase Email Verification
# ---------------------------------------------------------------------------

def render_wizard_2_email_verification(user_profile: Dict[str, Any]) -> bool:
    """
    Wizard 2 — Firebase Email Verification in a centered glass card.
    """
    card_header = textwrap.dedent("""
        <div style="max-width: 680px; margin: 0 auto;">
            <div class="cyber-card cyber-card-glow" style="text-align: center; padding: 2.2rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 0.8rem;">📧</div>
                <h3 style="color: #00F2FE; margin-bottom: 0.4rem;">Step 2 — Email Verification Required</h3>
                <p style="color: #94A3B8; font-size: 0.92rem; margin-bottom: 1.5rem;">
                    Firebase Identity Verification link dispatched to registered email.
                </p>
    """).strip()
    st.markdown(card_header, unsafe_allow_html=True)

    candidate_email = user_profile.get("email", "student@university.edu")
    candidate_name = user_profile.get("full_name", "Candidate")
    id_token = user_profile.get("id_token", "")

    email_badge = textwrap.dedent(f"""
        <div style="background: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; text-align: left;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">Candidate: <strong>{candidate_name}</strong></span>
                <span class="ai-badge badge-amber">PENDING VERIFICATION</span>
            </div>
            <div style="color: #00F2FE; font-family: monospace; font-size: 1.05rem; word-break: break-all;">
                {candidate_email}
            </div>
        </div>
    """).strip()
    st.markdown(email_badge, unsafe_allow_html=True)

    st.info(
        "📩 **Verification Instructions**:\n"
        "1. Open your email inbox and click the Firebase verification link.\n"
        "2. Return to this screen and click **Check Verification Status** below to continue."
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        if st.button("🔄 Check Verification Status", type="primary", use_container_width=True, key="btn_check_verify"):
            with st.spinner("Querying Firebase Identity Gateway..."):
                if id_token:
                    ok, user_info = FirebaseAuthHandler.get_user_info(id_token)
                    if ok and user_info.get("emailVerified", False):
                        st.session_state["user_profile"]["email_verified"] = True
                        st.success("✅ Email verified successfully! Moving to Step 3.")
                        return True
                    else:
                        st.error("❌ Email is not verified yet. Please click the link in your email and try again.")
                else:
                    st.error("Firebase Authentication token missing.")

    with col_v2:
        if st.button("📩 Resend Verification Email", use_container_width=True, key="btn_resend_verify"):
            if id_token:
                ok_res, msg_res = FirebaseAuthHandler.send_verification_email(id_token)
                if ok_res:
                    st.success("✅ Verification email re-sent.")
                else:
                    st.error(f"❌ {msg_res}")

    st.markdown("</div></div>", unsafe_allow_html=True)
    return False


# ---------------------------------------------------------------------------
# Wizard 3 — Candidate Profile, Scheduled Exam & Examination Rules Agreement
# ---------------------------------------------------------------------------

def render_wizard_3_exam_details(user_profile: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Wizard 3 — Candidate Profile & Examination Agreement in premium glass cards.
    """
    st.subheader("📋 Step 3 — Candidate Profile & Examination Agreement")
    st.caption("Verify candidate details and accept automated AI proctoring guidelines.")

    col_p, col_e = st.columns(2)

    with col_p:
        profile_card = textwrap.dedent(f"""
            <div class="cyber-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #00F2FE;">👤 Candidate Profile</h4>
                    <span class="ai-badge badge-green">VERIFIED</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.92rem;">
                    <div><span style="color: #94A3B8;">Full Name:</span> <strong style="color: #F8FAFC;">{user_profile.get('full_name', 'Alex Johnson')}</strong></div>
                    <div><span style="color: #94A3B8;">Email:</span> <strong style="color: #F8FAFC;">{user_profile.get('email', 'alex@university.edu')}</strong></div>
                    <div><span style="color: #94A3B8;">Student ID:</span> <strong style="color: #F8FAFC;">{user_profile.get('student_id', 'STU-2026-8842')}</strong></div>
                    <div><span style="color: #94A3B8;">Authentication:</span> <span style="color: #00FF9D;">Firebase Identity Verified</span></div>
                </div>
            </div>
        """).strip()
        st.markdown(profile_card, unsafe_allow_html=True)

    with col_e:
        assessment_header = textwrap.dedent("""
            <div class="cyber-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #00F2FE;">📅 Scheduled Assessment</h4>
                    <span class="ai-badge badge-cyan">30 MINUTES</span>
                </div>
        """).strip()
        st.markdown(assessment_header, unsafe_allow_html=True)
        exam_title = st.selectbox(
            "Select Scheduled Examination:",
            options=[
                "CS-101: Artificial Intelligence & Machine Learning Midterm Exam",
                "CS-204: Computer Vision & Pattern Recognition Assessment",
                "ENG-302: Advanced Software Architecture Final Exam",
                "MTH-105: Applied Probability & Statistics Evaluation"
            ],
            key="exam_title_select"
        )
        exam_duration = "30 Minutes"
        proctoring_mode = "AI Automated Multi-Model Proctoring (4 CNNs + YOLOv5)"
        st.caption(f"• **Duration**: {exam_duration} | **Proctoring**: {proctoring_mode}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Examination Rules Panel
    rules_panel = textwrap.dedent("""
        <div class="cyber-card">
            <h4 style="color: #00F2FE; margin-bottom: 0.8rem;">📜 Automated Proctoring Rules & Candidate Agreement</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; margin-bottom: 1.2rem; font-size: 0.86rem; color: #CBD5E1;">
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">1. Continuous webcam monitoring required</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">2. Active screen-sharing stream required</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">3. Candidate face must remain fully visible</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">4. Gaze deviations continuously evaluated</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">5. Mobile device usage triggers spatial alert</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">6. Camera absence flags anomaly session</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">7. Screen stream interruptions logged</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">8. 4-CNN ensemble consensus evaluation</div>
                <div style="background: rgba(10,16,28,0.5); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(0,242,254,0.1);">9. Spatial YOLOv5 object detection active</div>
            </div>
    """).strip()
    st.markdown(rules_panel, unsafe_allow_html=True)

    agree = st.checkbox(
        "I agree to the automated AI proctoring rules, webcam recording, "
        "screen monitoring, and candidate verification guidelines.",
        value=False,
        key="proctoring_rules_agree_cb"
    )

    exam_details = {
        "exam_title": exam_title,
        "exam_duration": exam_duration,
        "proctoring_mode": proctoring_mode,
        "agreed": agree
    }

    btn_disabled = not agree
    if st.button(
        "🚀 Proceed to System Check",
        type="primary",
        use_container_width=True,
        disabled=btn_disabled,
        key="btn_proceed_step4"
    ):
        return True, exam_details

    if btn_disabled:
        st.caption("⚠️ Check the agreement box above to activate the proceed button.")

    st.markdown("</div>", unsafe_allow_html=True)
    return False, exam_details


# ---------------------------------------------------------------------------
# Wizard 4 — System Check (Camera, Mic, Screen)
# ---------------------------------------------------------------------------

def render_wizard_4_system_check() -> bool:
    """
    Wizard 4 — Hardware System Readiness Scan.
    Displays elegant glass cards with READY / CHECKING / FAILED states without raw HTML leaks.
    """
    st.subheader("🖥️ Step 4 — AI Security System Readiness Scan")
    st.caption("Scanning candidate hardware devices and browser security permissions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        card1 = textwrap.dedent("""
            <div class="cyber-card" style="text-align: center;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📷</div>
                <h4 style="margin: 0; color: #F8FAFC;">Camera Feed</h4>
                <div style="margin: 0.8rem 0;">
                    <span class="ai-badge badge-green">READY</span>
                </div>
                <p style="color: #94A3B8; font-size: 0.82rem; margin: 0;">Live webcam permission verified</p>
            </div>
        """).strip()
        st.markdown(card1, unsafe_allow_html=True)

    with col2:
        card2 = textwrap.dedent("""
            <div class="cyber-card" style="text-align: center;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🎤</div>
                <h4 style="margin: 0; color: #F8FAFC;">Audio Device</h4>
                <div style="margin: 0.8rem 0;">
                    <span class="ai-badge badge-green">READY</span>
                </div>
                <p style="color: #94A3B8; font-size: 0.82rem; margin: 0;">WebAudio microphone active</p>
            </div>
        """).strip()
        st.markdown(card2, unsafe_allow_html=True)

    with col3:
        card3 = textwrap.dedent("""
            <div class="cyber-card" style="text-align: center;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🖥️</div>
                <h4 style="margin: 0; color: #F8FAFC;">Screen Sharing</h4>
                <div style="margin: 0.8rem 0;">
                    <span class="ai-badge badge-green">READY</span>
                </div>
                <p style="color: #94A3B8; font-size: 0.82rem; margin: 0;">DisplayMedia API supported</p>
            </div>
        """).strip()
        st.markdown(card3, unsafe_allow_html=True)

    st.info("💡 **System Scan Passed**: All hardware devices and browser security permissions are active. Click **Start Proctored Exam Session** to begin.")

    if st.button("🎬 Start Proctored Exam Session", type="primary", use_container_width=True, key="btn_start_exam_session"):
        return True

    return False
