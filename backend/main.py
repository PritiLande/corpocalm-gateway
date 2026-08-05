import os
import smtplib
import httpx
import json
import bcrypt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Query, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
from supabase.lib.client_options import SyncClientOptions
from groq import Groq
from dotenv import load_dotenv
from jose import jwt, JWTError
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, Field

from schemas import JobCreate, ApplicationCreate

# ──────────────────────────────────────────
# 1. SETUP
# ──────────────────────────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SECRET_KEY   = os.getenv("SECRET_GATEWAY_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Email config (optional)
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
HR_EMAIL      = os.getenv("HR_EMAIL", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

# Fix SSL certificate verification issue on Windows
_http_client = httpx.Client(verify=False)
supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    options=SyncClientOptions(httpx_client=_http_client)
)

# Groq AI client — disable SSL verify for Windows compatibility
groq_client = Groq(
    api_key=GROQ_API_KEY,
    http_client=httpx.Client(verify=False)
) if GROQ_API_KEY else None

# ── Auth setup ──
JWT_SECRET       = os.getenv("JWT_SECRET", "corpocalm_secret_jwt_key_2026")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRY_HOURS = 24
bearer_scheme    = HTTPBearer()

# ── Auth helpers ──
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

# ── Auth models ──
class HRSignup(BaseModel):
    name: str
    company_name: str
    email: EmailStr
    password: str

class HRLogin(BaseModel):
    email: EmailStr
    password: str

class GenerateQuestionsRequest(BaseModel):
    role_title: str
    skills_required: Optional[str] = None

class GenerateDescriptionRequest(BaseModel):
    role_title: str
    skills_required: Optional[str] = None
    job_type: Optional[str] = None
    experience_required: Optional[str] = None
    location: Optional[str] = None

class GenerateSkillsRequest(BaseModel):
    role_title: str
    job_type: Optional[str] = None
    experience_required: Optional[str] = None
    description: Optional[str] = None

class GenerateFullJobRequest(BaseModel):
    role_title: str

def create_token(hr_id: str, email: str, company_id: str = "", role: str = "admin") -> str:
    payload = {
        "sub": hr_id,
        "email": email,
        "company_id": company_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_hr(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please login again.")

app = FastAPI(title="CorpoCalm Gateway API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# 2. AI SCORING ENGINE (Groq LLaMA 3)
# ──────────────────────────────────────────

def compute_ai_score(answers: list) -> tuple[int, str]:
    """
    Uses Groq LLaMA 3 to intelligently score candidate answers.
    Falls back to keyword-based scoring if Groq is not configured.
    Returns (score 0-100, feedback_text).
    """
    if not answers:
        return 0, ""

    # ── Fallback: keyword + length scoring if no Groq key ──
    if not groq_client:
        return _fallback_score(answers), ""

    try:
        # Build Q&A pairs for the prompt
        qa_text = ""
        for i, a in enumerate(answers, 1):
            qa_text += f"Q{i}: {a.question_text}\nAnswer: {a.candidate_answer or '(no answer provided)'}\n\n"

        prompt = f"""You are a senior technical HR evaluator. Score and explain the following candidate answers.

For each answer evaluate:
- Accuracy and correctness
- Depth and detail of explanation
- Practical understanding shown
- Communication clarity

Answers to evaluate:
{qa_text}

Respond ONLY with a valid JSON object in this exact format, nothing else:
{{
  "scores": [score1, score2, ...],
  "per_question": ["one sentence feedback for Q1", "one sentence feedback for Q2", ...],
  "strengths": "2-3 specific things the candidate did well",
  "improvements": "2-3 specific areas where the candidate could improve",
  "overall_feedback": "2-3 sentence professional summary explaining why this score was given"
}}

Where each score is an integer 0-100 matching each question in order."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        scores = result.get("scores", [])

        if not scores:
            return _fallback_score(answers), ""

        avg = round(sum(scores) / len(scores))
        score = max(0, min(avg, 100))

        # Build structured feedback text stored as JSON string
        feedback = json.dumps({
            "per_question":   result.get("per_question", []),
            "strengths":      result.get("strengths", ""),
            "improvements":   result.get("improvements", ""),
            "overall":        result.get("overall_feedback", ""),
            "question_scores": scores
        })

        return score, feedback

    except Exception as e:
        print(f"[Groq Warning] AI scoring failed, using fallback: {e}")
        return _fallback_score(answers), ""


def _fallback_score(answers: list) -> int:
    """Keyword + length based fallback scoring."""
    KEYWORDS = [
        "algorithm", "database", "api", "index", "query", "cache", "rest",
        "authentication", "jwt", "docker", "microservice", "sql", "git",
        "async", "server", "framework", "library", "function", "variable",
        "object", "class", "method", "array", "loop", "condition", "deploy"
    ]
    if not answers:
        return 0
    total = 0
    for a in answers:
        text = (a.candidate_answer or "").strip().lower()
        wc = len(text.split())
        pts = 0
        if wc >= 80: pts += 60
        elif wc >= 50: pts += 50
        elif wc >= 30: pts += 38
        elif wc >= 15: pts += 25
        elif wc >= 5: pts += 10
        kw_hits = sum(1 for kw in KEYWORDS if kw in text)
        pts += min(kw_hits * 8, 40)
        total += pts
    return max(0, min(round(total / len(answers)), 100))


def determine_status(score: int, passing_threshold: int) -> str:
    return "Passed" if score >= passing_threshold else "Failed"


# ──────────────────────────────────────────
# 3. EMAIL NOTIFICATION
# ──────────────────────────────────────────

def send_candidate_confirmation(candidate_name: str, candidate_email: str, role_title: str,
                               ai_score: int = None, status: str = None):
    """Sends a confirmation email to the candidate after submission, including their score and result."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return

    try:
        passed = status == "Passed"
        result_color  = "#2e7d32" if passed else "#c62828"
        result_bg     = "#e8f5e9" if passed else "#ffebee"
        result_border = "#a5d6a7" if passed else "#ef9a9a"
        result_label  = "✅ Passed" if passed else "❌ Did Not Meet Threshold"
        result_msg    = (
            "Congratulations! You have passed the screening assessment. Our HR team will review your application and reach out with next steps."
            if passed else
            "Unfortunately, your score did not meet the minimum threshold for this role. You are welcome to apply for other positions."
        )

        score_block = ""
        if ai_score is not None:
            score_block = f"""
            <div style="background:{result_bg};border:1px solid {result_border};border-radius:8px;padding:16px 20px;margin:20px 0;text-align:center;">
                <div style="font-size:13px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Your Score</div>
                <div style="font-size:36px;font-weight:800;color:{result_color};">{ai_score}<span style="font-size:16px;font-weight:500;color:#888;"> / 100</span></div>
                <div style="margin-top:8px;font-weight:700;color:{result_color};font-size:15px;">{result_label}</div>
            </div>"""

        subject = f"Assessment Result — {role_title} | CorpoCalm Gateway"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1a237e,#283593);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Assessment Result</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Dear <strong>{candidate_name}</strong>,</p>
                <p style="margin-top:12px;">Thank you for completing the screening assessment for <strong>{role_title}</strong>.</p>
                {score_block}
                <p style="margin-top:12px;">{result_msg}</p>
                <p style="margin-top:24px;color:#aaa;font-size:12px;">Please do not reply to this email.</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(candidate_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Assessment result email failed: {e}")


def send_shortlisted_email(candidate_name: str, candidate_email: str, role_title: str):
    """Notifies candidate they have been shortlisted."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    try:
        subject = f"Congratulations! You've been Shortlisted — {role_title}"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Shortlisted</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Dear <strong>{candidate_name}</strong>,</p>
                <p style="margin-top:12px;">We are pleased to inform you that you have been <strong style="color:#2e7d32;">shortlisted</strong> for the position of <strong>{role_title}</strong>.</p>
                <p style="margin-top:12px;">Our HR team will reach out to you shortly with the next steps in the hiring process.</p>
                <p style="margin-top:12px;">Congratulations and thank you for your interest!</p>
                <p style="margin-top:24px;color:#aaa;font-size:12px;">Please do not reply to this email.</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(candidate_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Shortlisted email failed: {e}")


def send_rejected_email(candidate_name: str, candidate_email: str, role_title: str):
    """Notifies candidate they have been rejected."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    try:
        subject = f"Application Update — {role_title} | CorpoCalm Gateway"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#37474f,#546e7a);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Application Update</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Dear <strong>{candidate_name}</strong>,</p>
                <p style="margin-top:12px;">Thank you for your interest in the <strong>{role_title}</strong> position and for taking the time to complete our assessment.</p>
                <p style="margin-top:12px;">After careful consideration, we have decided to move forward with other candidates whose profiles more closely match our current requirements.</p>
                <p style="margin-top:12px;">We appreciate the effort you put into the process and encourage you to apply for future opportunities that match your skills.</p>
                <p style="margin-top:12px;">We wish you the very best in your job search.</p>
                <p style="margin-top:24px;color:#aaa;font-size:12px;">Please do not reply to this email.</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(candidate_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Rejection email failed: {e}")


def send_interview_scheduled_email(candidate_name: str, candidate_email: str, role_title: str, interview_details: str):
    """Notifies candidate that an interview has been scheduled."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    try:
        subject = f"Interview Scheduled — {role_title} | CorpoCalm Gateway"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#4a148c,#6a1b9a);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Interview Scheduled</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Dear <strong>{candidate_name}</strong>,</p>
                <p style="margin-top:12px;">Your interview for the position of <strong>{role_title}</strong> has been scheduled.</p>
                <div style="background:#f3e5f5;border-left:4px solid #6a1b9a;padding:14px 16px;border-radius:4px;margin-top:16px;">
                    <strong>Interview Details:</strong><br>
                    <span style="white-space:pre-wrap;">{interview_details}</span>
                </div>
                <p style="margin-top:16px;">Please ensure you are available at the specified time. Reply to this email if you have any questions.</p>
                <p style="margin-top:24px;color:#aaa;font-size:12px;">Best of luck!</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(candidate_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Interview scheduled email failed: {e}")


def send_welcome_email(hr_name: str, hr_email: str, company_name: str):
    """Welcome email sent to HR on signup."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    try:
        subject = "Welcome to CorpoCalm Gateway!"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1a237e,#283593);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Welcome!</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Hi <strong>{hr_name}</strong>,</p>
                <p style="margin-top:12px;">Welcome to <strong>CorpoCalm Gateway</strong>! Your HR account for <strong>{company_name}</strong> has been created successfully.</p>
                <p style="margin-top:12px;">You can now:</p>
                <ul style="margin-top:8px;line-height:1.9;color:#444;">
                    <li>Publish job assessments with AI-generated questions</li>
                    <li>Share portal links with candidates</li>
                    <li>Review AI-scored responses on the dashboard</li>
                    <li>Shortlist, review, or reject candidates</li>
                </ul>
                <p style="margin-top:16px;">Get started by logging in and publishing your first job.</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(hr_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Welcome email failed: {e}")


def send_assessment_assigned_email(candidate_email: str, role_title: str, company_name: str, portal_link: str, time_limit: int):
    """Email to candidate when HR shares the job portal link (called via API)."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    try:
        subject = f"You've Been Invited to Apply — {role_title} at {company_name}"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1a237e,#283593);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Assessment Invitation</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Hello,</p>
                <p style="margin-top:12px;">You have been invited to complete a screening assessment for the position of <strong>{role_title}</strong> at <strong>{company_name}</strong>.</p>
                <div style="background:#e8eaf6;border-left:4px solid #1a237e;padding:14px 16px;border-radius:4px;margin-top:16px;">
                    <strong>Assessment Details:</strong><br>
                    Role: {role_title}<br>
                    Time Limit: {time_limit} minutes<br>
                    Max Attempts: 2
                </div>
                <div style="text-align:center;margin-top:24px;">
                    <a href="{portal_link}" style="background:#1a237e;color:white;padding:12px 28px;border-radius:6px;text-decoration:none;font-weight:600;font-size:15px;">Start Assessment</a>
                </div>
                <p style="margin-top:20px;font-size:13px;color:#888;">If the button doesn't work, copy this link: {portal_link}</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(candidate_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Assessment assigned email failed: {e}")


def _send_email(to_email: str, subject: str, html_body: str):
    """Shared SMTP send helper."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, to_email, msg.as_string())


def send_hr_notification(candidate_name: str, candidate_email: str, role_title: str, score: int, status: str, tab_switches: int):
    """
    Sends an email to HR when a new application is submitted.
    Silently skips if SMTP credentials are not configured.
    """
    if not SMTP_USER or not SMTP_PASSWORD or not HR_EMAIL:
        return  # Email not configured — skip silently

    try:
        subject = f"New Application: {candidate_name} for {role_title}"
        body = f"""
        <h2>New Assessment Submission</h2>
        <table style="font-family:sans-serif;border-collapse:collapse;">
            <tr><td style="padding:8px;font-weight:bold;">Candidate</td><td style="padding:8px;">{candidate_name}</td></tr>
            <tr><td style="padding:8px;font-weight:bold;">Email</td><td style="padding:8px;">{candidate_email}</td></tr>
            <tr><td style="padding:8px;font-weight:bold;">Role</td><td style="padding:8px;">{role_title}</td></tr>
            <tr><td style="padding:8px;font-weight:bold;">Score</td><td style="padding:8px;">{score} / 100</td></tr>
            <tr><td style="padding:8px;font-weight:bold;">Status</td><td style="padding:8px;color:{'green' if status == 'Passed' else 'red'};">{status}</td></tr>
            <tr><td style="padding:8px;font-weight:bold;">Tab Switches</td><td style="padding:8px;">{'[!] ' if tab_switches > 2 else ''}{tab_switches}</td></tr>
        </table>
        <p>Log in to the <a href="#">HR Dashboard</a> to view full answers.</p>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = HR_EMAIL
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, HR_EMAIL, msg.as_string())

    except Exception as e:
        # Don't fail the request if email fails — just log it
        print(f"[Email Warning] Could not send notification: {e}")


# ──────────────────────────────────────────
# 4. ENDPOINTS
# ──────────────────────────────────────────

@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok"}


# ──────────────────────────────────────────
# HR AUTH ENDPOINTS
# ──────────────────────────────────────────

@app.post("/hr/signup", summary="HR Sign Up")
def hr_signup(data: HRSignup):
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if len(data.password) > 72:
        raise HTTPException(status_code=400, detail="Password must be 72 characters or less.")

    # Check if email already exists
    existing = supabase.table("hr_users").select("id").eq("email", data.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    hashed = hash_password(data.password)

    # Create company first
    company_res = supabase.table("companies").insert({"name": data.company_name}).execute()
    if not company_res.data:
        raise HTTPException(status_code=500, detail="Failed to create company.")
    company_id = company_res.data[0]["id"]

    # Create HR user as admin of that company
    res = supabase.table("hr_users").insert({
        "name": data.name,
        "company_name": data.company_name,
        "company_id": company_id,
        "role": "admin",
        "email": data.email,
        "password_hash": hashed
    }).execute()

    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create account.")

    # Send welcome email to new HR
    send_welcome_email(data.name, data.email, data.company_name)

    return {"message": "Account created successfully. Please login."}


@app.post("/hr/login", summary="HR Login")
def hr_login(data: HRLogin):
    res = supabase.table("hr_users").select("*").eq("email", data.email).execute()

    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    hr = res.data[0]

    if not verify_password(data.password, hr["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_token(
        hr_id=str(hr["id"]),
        email=hr["email"],
        company_id=str(hr.get("company_id") or ""),
        role=hr.get("role", "admin")
    )
    return {
        "token": token,
        "name": hr["name"],
        "email": hr["email"],
        "company_name": hr.get("company_name", ""),
        "company_id": str(hr.get("company_id") or ""),
        "role": hr.get("role", "admin")
    }


# ──────────────────────────────────────────
# TEAM MANAGEMENT ENDPOINTS
# ──────────────────────────────────────────

import secrets

class InviteMemberRequest(BaseModel):
    email: EmailStr

class AcceptInviteRequest(BaseModel):
    token: str
    name: str
    password: str = Field(..., min_length=6, max_length=72)

class UpdateMemberRoleRequest(BaseModel):
    role: str  # 'admin' or 'member'


@app.post("/team/invite", summary="Invite a team member (admin only)")
def invite_team_member(data: InviteMemberRequest, hr=Depends(get_current_hr)):
    """Sends an invite to a new HR team member. Only admins can invite."""
    if hr.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can invite team members.")

    company_id = hr.get("company_id")
    if not company_id:
        raise HTTPException(status_code=400, detail="Your account is not linked to a company.")

    # Check if already a member
    existing = supabase.table("hr_users").select("id").eq("email", data.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="This email is already registered.")

    # Check for existing pending invite
    existing_invite = supabase.table("team_invites")\
        .select("id")\
        .eq("company_id", company_id)\
        .eq("invited_email", data.email)\
        .eq("status", "pending")\
        .execute()
    if existing_invite.data:
        raise HTTPException(status_code=400, detail="An invite has already been sent to this email.")

    # Get inviter info
    inviter_res = supabase.table("hr_users").select("id, name").eq("email", hr["email"]).execute()
    inviter_id = inviter_res.data[0]["id"] if inviter_res.data else None

    # Get company name
    company_res = supabase.table("companies").select("name").eq("id", company_id).execute()
    company_name = company_res.data[0]["name"] if company_res.data else "your company"

    # Generate unique invite token
    invite_token = secrets.token_urlsafe(32)

    supabase.table("team_invites").insert({
        "company_id": company_id,
        "invited_email": data.email,
        "invited_by": inviter_id,
        "token": invite_token,
        "status": "pending"
    }).execute()

    # Send invite email
    _send_team_invite_email(
        to_email=data.email,
        inviter_name=inviter_res.data[0]["name"] if inviter_res.data else "HR Admin",
        company_name=company_name,
        invite_token=invite_token
    )

    return {"message": f"Invite sent to {data.email}"}


@app.post("/team/accept-invite", summary="Accept a team invite and create account")
def accept_invite(data: AcceptInviteRequest):
    """Called when an invited member clicks the invite link and signs up."""
    # Validate token
    invite_res = supabase.table("team_invites")\
        .select("*")\
        .eq("token", data.token)\
        .eq("status", "pending")\
        .execute()

    if not invite_res.data:
        raise HTTPException(status_code=400, detail="Invalid or expired invite link.")

    invite = invite_res.data[0]

    # Check expiry
    from datetime import timezone
    expires_at = datetime.fromisoformat(invite["expires_at"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail="This invite link has expired.")

    # Check email not already registered
    existing = supabase.table("hr_users").select("id").eq("email", invite["invited_email"]).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    # Get company info
    company_res = supabase.table("companies").select("name").eq("id", invite["company_id"]).execute()
    company_name = company_res.data[0]["name"] if company_res.data else ""

    hashed = hash_password(data.password)
    res = supabase.table("hr_users").insert({
        "name": data.name,
        "email": invite["invited_email"],
        "password_hash": hashed,
        "company_id": invite["company_id"],
        "company_name": company_name,
        "role": "member"
    }).execute()

    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create account.")

    # Mark invite as accepted
    supabase.table("team_invites").update({"status": "accepted"}).eq("id", invite["id"]).execute()

    send_welcome_email(data.name, invite["invited_email"], company_name)

    return {"message": "Account created successfully. Please login."}


@app.get("/team/members", summary="List all HR team members in the same company")
def list_team_members(hr=Depends(get_current_hr)):
    """Returns all HR users in the same company."""
    company_id = hr.get("company_id")
    if not company_id:
        raise HTTPException(status_code=400, detail="Your account is not linked to a company.")

    res = supabase.table("hr_users")\
        .select("id, name, email, role, created_at")\
        .eq("company_id", company_id)\
        .order("created_at")\
        .execute()

    return res.data or []


@app.get("/team/invites", summary="List pending invites (admin only)")
def list_invites(hr=Depends(get_current_hr)):
    """Returns all pending invites for this company."""
    if hr.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view invites.")

    company_id = hr.get("company_id")
    res = supabase.table("team_invites")\
        .select("id, invited_email, status, created_at, expires_at")\
        .eq("company_id", company_id)\
        .order("created_at", desc=True)\
        .execute()

    return res.data or []


@app.delete("/team/members/{member_id}", summary="Remove a team member (admin only)")
def remove_team_member(member_id: str, hr=Depends(get_current_hr)):
    """Removes a member from the team. Admins cannot remove themselves."""
    if hr.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can remove team members.")

    if hr["sub"] == member_id:
        raise HTTPException(status_code=400, detail="You cannot remove yourself.")

    company_id = hr.get("company_id")

    # Verify member belongs to same company
    member_res = supabase.table("hr_users")\
        .select("id, company_id, role")\
        .eq("id", member_id)\
        .execute()

    if not member_res.data:
        raise HTTPException(status_code=404, detail="Member not found.")

    if str(member_res.data[0].get("company_id")) != str(company_id):
        raise HTTPException(status_code=403, detail="This member does not belong to your company.")

    supabase.table("hr_users").delete().eq("id", member_id).execute()
    return {"message": "Member removed successfully."}


@app.patch("/team/members/{member_id}/role", summary="Change a member's role (admin only)")
def update_member_role(member_id: str, data: UpdateMemberRoleRequest, hr=Depends(get_current_hr)):
    """Promotes or demotes a team member. Only admins can do this."""
    if hr.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can change roles.")

    if data.role not in ("admin", "member"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'member'.")

    company_id = hr.get("company_id")
    member_res = supabase.table("hr_users").select("company_id").eq("id", member_id).execute()
    if not member_res.data or str(member_res.data[0].get("company_id")) != str(company_id):
        raise HTTPException(status_code=404, detail="Member not found in your company.")

    supabase.table("hr_users").update({"role": data.role}).eq("id", member_id).execute()
    return {"message": f"Role updated to {data.role}."}


def _send_team_invite_email(to_email: str, inviter_name: str, company_name: str, invite_token: str):
    """Sends a team invite email with a signup link."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    try:
        # The frontend accept-invite page
        invite_link = f"{os.getenv('FRONTEND_URL', 'http://localhost:5500/frontend')}/accept-invite.html?token={invite_token}"
        subject = f"You're invited to join {company_name} on CorpoCalm Gateway"
        body = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(135deg,#1a237e,#283593);padding:24px;color:white;text-align:center;">
                <h2 style="margin:0;">CorpoCalm Gateway</h2>
                <p style="margin:4px 0 0;opacity:0.85;font-size:13px;">Team Invitation</p>
            </div>
            <div style="padding:28px 24px;">
                <p>Hello,</p>
                <p style="margin-top:12px;"><strong>{inviter_name}</strong> has invited you to join <strong>{company_name}</strong> on CorpoCalm Gateway as an HR team member.</p>
                <div style="text-align:center;margin-top:24px;">
                    <a href="{invite_link}" style="background:#1a237e;color:white;padding:12px 28px;border-radius:6px;text-decoration:none;font-weight:600;font-size:15px;">Accept Invitation</a>
                </div>
                <p style="margin-top:20px;font-size:13px;color:#888;">This link expires in 7 days. If you did not expect this, ignore this email.</p>
            </div>
            <div style="background:#f5f5f5;padding:14px;text-align:center;font-size:11px;color:#aaa;">
                CorpoCalm Gateway &nbsp;|&nbsp; Built by Priti Ganesh Lande
            </div>
        </div>"""
        _send_email(to_email, subject, body)
    except Exception as e:
        print(f"[Email Warning] Team invite email failed: {e}")


# ──────────────────────────────────────────
# AI QUESTION GENERATOR
# ──────────────────────────────────────────

@app.post("/generate-questions", summary="AI generates screening questions")
def generate_questions(data: GenerateQuestionsRequest, hr=Depends(get_current_hr)):
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI not configured. Please add GROQ_API_KEY to .env")

    try:
        skills_text = f"Required skills: {data.skills_required}" if data.skills_required else ""

        prompt = f"""You are an expert technical HR interviewer.

Generate exactly 5 screening questions for a job interview.

Job Title: {data.role_title}
{skills_text}

Rules:
- Questions should be easy to medium difficulty
- Mix of practical and conceptual questions
- Suitable for a written screening test
- Each question on its own line
- No numbering, no bullet points, just the question text

Return ONLY the 5 questions, one per line, nothing else."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )

        content = response.choices[0].message.content.strip()
        questions = [q.strip() for q in content.split('\n') if q.strip() and len(q.strip()) > 10]
        questions = questions[:5]  # max 5

        if not questions:
            raise HTTPException(status_code=500, detail="AI returned empty response.")

        return {"questions": questions}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.post("/generate-description", summary="AI writes a job description")
def generate_description(data: GenerateDescriptionRequest, hr=Depends(get_current_hr)):
    """Uses Groq LLaMA 3 to write a professional job description based on the job title and details."""
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI not configured. Please add GROQ_API_KEY to .env")

    try:
        details = []
        if data.skills_required:      details.append(f"Skills: {data.skills_required}")
        if data.job_type:             details.append(f"Job Type: {data.job_type}")
        if data.experience_required:  details.append(f"Experience: {data.experience_required}")
        if data.location:             details.append(f"Location: {data.location}")
        details_text = "\n".join(details) if details else "No additional details provided."

        prompt = f"""You are an expert HR professional and technical recruiter.

Write a concise, professional job description for the following role.

Job Title: {data.role_title}
{details_text}

Requirements:
- 3 to 5 sentences maximum
- Describe what the role involves and what the team does
- Mention key responsibilities naturally
- Sound professional but approachable
- Do NOT use bullet points or headers
- Do NOT include salary or application instructions
- Plain paragraph text only

Return ONLY the job description text, nothing else."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        description = response.choices[0].message.content.strip()

        if not description:
            raise HTTPException(status_code=500, detail="AI returned empty response.")

        return {"description": description}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.post("/generate-skills", summary="AI suggests required skills for a job role")
def generate_skills(data: GenerateSkillsRequest, hr=Depends(get_current_hr)):
    """Uses Groq LLaMA 3 to suggest a comma-separated list of required skills."""
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI not configured. Please add GROQ_API_KEY to .env")

    try:
        details = []
        if data.job_type:            details.append(f"Job Type: {data.job_type}")
        if data.experience_required: details.append(f"Experience: {data.experience_required}")
        if data.description:         details.append(f"Description: {data.description}")
        details_text = "\n".join(details) if details else ""

        prompt = f"""You are a senior technical recruiter with deep knowledge of tech stacks.

Suggest the most relevant required skills for the following job role.

Job Title: {data.role_title}
{details_text}

Rules:
- Return 6 to 10 skills
- Mix of technical and soft skills relevant to the role
- Each skill is a short phrase (1-3 words max)
- Comma-separated on a single line
- No numbering, no bullets, no explanation
- Example format: Python, FastAPI, PostgreSQL, Docker, REST APIs, Problem Solving

Return ONLY the comma-separated skills list, nothing else."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )

        skills = response.choices[0].message.content.strip()

        # Clean up any accidental newlines or extra spaces
        skills = ", ".join([s.strip() for s in skills.replace("\n", ",").split(",") if s.strip()])

        if not skills:
            raise HTTPException(status_code=500, detail="AI returned empty response.")

        return {"skills": skills}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.post("/generate-full-job", summary="AI generates entire job posting from just a title")
def generate_full_job(data: GenerateFullJobRequest, hr=Depends(get_current_hr)):
    """
    Single call that generates description, skills, questions,
    time limit, passing score, job type, experience and salary range
    — all from just a job title.
    """
    if not groq_client:
        raise HTTPException(status_code=503, detail="AI not configured. Please add GROQ_API_KEY to .env")

    try:
        prompt = f"""You are a senior HR professional and technical recruiter.

Generate a complete job posting for the role below. Return ONLY valid JSON — no markdown, no explanation.

Job Title: {data.role_title}

Return exactly this JSON structure:
{{
  "description": "3-5 sentence professional job description as plain text",
  "skills_required": "comma-separated list of 6-10 relevant skills",
  "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"],
  "job_type": "one of: Full-Time, Part-Time, Contract, Internship",
  "experience_required": "e.g. 2-4 years",
  "salary_range": "e.g. 8-12 LPA",
  "time_limit_minutes": 25,
  "passing_threshold": 65
}}

Rules:
- questions: exactly 5, practical screening questions, mix of conceptual and applied
- time_limit_minutes: integer between 15 and 45 based on complexity
- passing_threshold: integer between 55 and 75
- salary_range: realistic for Indian market in LPA
- No markdown, no code blocks, return raw JSON only"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=900
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown code blocks if model wrapped the JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)

        # Validate required keys are present
        required = ["description", "skills_required", "questions", "job_type",
                    "experience_required", "salary_range", "time_limit_minutes", "passing_threshold"]
        for key in required:
            if key not in result:
                raise ValueError(f"AI response missing field: {key}")

        # Ensure questions is a list of strings
        if not isinstance(result["questions"], list):
            result["questions"] = []

        return result

    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid data: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@app.post("/jobs", summary="Create a job posting (HR login required)")
def create_job(job: JobCreate, hr=Depends(get_current_hr)):
    """Creates a new job and its screening questions. Requires HR login token."""
    if not job.questions:
        raise HTTPException(status_code=400, detail="At least one question is required")

    job_res = supabase.table("jobs").insert({
        "company_id": hr.get("company_id") or None,
        "status": "published",
        "role_title": job.role_title,
        "time_limit_minutes": job.time_limit_minutes,
        "passing_threshold": job.passing_threshold,
        "description": job.description,
        "skills_required": job.skills_required,
        "location": job.location,
        "job_type": job.job_type,
        "salary_range": job.salary_range,
        "experience_required": job.experience_required,
        "expiry_date": job.expiry_date
    }).execute()

    if not job_res.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    new_job_id = job_res.data[0]["id"]

    q_payload = [
        {"job_id": new_job_id, "question_text": q.strip()}
        for q in job.questions if q.strip()
    ]
    supabase.table("screening_questions").insert(q_payload).execute()

    return {
        "job_id": new_job_id,
        "role_title": job.role_title,
        "question_count": len(q_payload)
    }


@app.post("/jobs/draft", summary="Save a job as draft (HR login required)")
def save_draft(job: JobCreate, hr=Depends(get_current_hr)):
    """Saves a job as a draft — not visible to candidates yet. Questions optional."""

    job_res = supabase.table("jobs").insert({
        "company_id": hr.get("company_id") or None,
        "status": "draft",
        "role_title": job.role_title or "Untitled Draft",
        "time_limit_minutes": job.time_limit_minutes,
        "passing_threshold": job.passing_threshold,
        "description": job.description,
        "skills_required": job.skills_required,
        "location": job.location,
        "job_type": job.job_type,
        "salary_range": job.salary_range,
        "experience_required": job.experience_required,
        "expiry_date": job.expiry_date
    }).execute()

    if not job_res.data:
        raise HTTPException(status_code=500, detail="Failed to save draft")

    new_job_id = job_res.data[0]["id"]

    # Save questions if provided
    if job.questions:
        q_payload = [
            {"job_id": new_job_id, "question_text": q.strip()}
            for q in job.questions if q.strip()
        ]
        if q_payload:
            supabase.table("screening_questions").insert(q_payload).execute()

    return {
        "job_id": new_job_id,
        "role_title": job.role_title or "Untitled Draft",
        "status": "draft"
    }


@app.patch("/jobs/{job_id}/publish", summary="Publish a draft job (HR login required)")
def publish_draft(job_id: str, hr=Depends(get_current_hr)):
    """Publishes an existing draft job, making it visible to candidates."""
    job_res = supabase.table("jobs").select("id, status, company_id").eq("id", job_id).execute()
    if not job_res.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_res.data[0]
    if job.get("status") == "published":
        raise HTTPException(status_code=400, detail="Job is already published.")

    # Verify questions exist
    q_res = supabase.table("screening_questions").select("id").eq("job_id", job_id).execute()
    if not q_res.data:
        raise HTTPException(status_code=400, detail="Add at least one screening question before publishing.")

    supabase.table("jobs").update({"status": "published"}).eq("id", job_id).execute()
    return {"job_id": job_id, "status": "published"}


@app.get("/jobs/drafts", summary="List all draft jobs for this HR's company")
def list_drafts(hr=Depends(get_current_hr)):
    """Returns all draft jobs scoped to the HR's company."""
    company_id = hr.get("company_id")
    query = supabase.table("jobs").select("id, role_title, location, job_type, created_at, status")\
        .eq("status", "draft")
    if company_id:
        query = query.eq("company_id", company_id)
    res = query.order("created_at", desc=True).execute()
    return res.data or []


@app.delete("/jobs/{job_id}/draft", summary="Delete a draft job")
def delete_draft(job_id: str, hr=Depends(get_current_hr)):
    """Deletes a draft job. Cannot delete published jobs this way."""
    job_res = supabase.table("jobs").select("id, status").eq("id", job_id).execute()
    if not job_res.data:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_res.data[0].get("status") != "draft":
        raise HTTPException(status_code=400, detail="Only draft jobs can be deleted this way.")
    supabase.table("jobs").delete().eq("id", job_id).execute()
    return {"message": "Draft deleted."}


@app.get("/jobs/{job_id}", summary="Get job info and questions (for candidate page)")
def get_job(job_id: str):
    job_res = supabase.table("jobs").select("*").eq("id", job_id).execute()
    if not job_res.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_res.data[0]

    # Check expiry date
    if job.get("expiry_date"):
        from datetime import date
        expiry = date.fromisoformat(job["expiry_date"])
        if date.today() > expiry:
            raise HTTPException(
                status_code=410,
                detail=f"This job posting has expired on {job['expiry_date']}. Applications are no longer accepted."
            )

    q_res = supabase.table("screening_questions").select("id, question_text").eq("job_id", job_id).execute()

    return {
        "job": job,
        "questions": q_res.data or []
    }


@app.post("/applications", summary="Submit a candidate's assessment")
def submit_application(app_data: ApplicationCreate):
    """
    Saves application + answers, scores the submission, sends HR notification.
    """
    # Validate job exists
    job_res = supabase.table("jobs").select("id, role_title, passing_threshold").eq("id", app_data.job_id).execute()
    if not job_res.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = job_res.data[0]
    passing_threshold = job_info.get("passing_threshold", 70)
    role_title = job_info.get("role_title", "Unknown Role")

    # Check attempt limit — max 2 attempts per candidate per job
    attempt_res = supabase.table("applications")\
        .select("id")\
        .eq("job_id", app_data.job_id)\
        .eq("candidate_email", app_data.candidate_email)\
        .execute()

    attempt_count = len(attempt_res.data) if attempt_res.data else 0

    if attempt_count >= 2:
        raise HTTPException(
            status_code=403,
            detail="You have already used both attempts for this assessment. No further attempts are allowed."
        )

    attempt_number = attempt_count + 1

    # Score the answers
    ai_score, ai_feedback = compute_ai_score(app_data.answers)
    status = determine_status(ai_score, passing_threshold)

    # Save application
    app_res = supabase.table("applications").insert({
        "job_id": app_data.job_id,
        "candidate_name": app_data.candidate_name,
        "candidate_email": app_data.candidate_email,
        "tab_switch_count": app_data.tab_switch_count,
        "face_away_count": app_data.face_away_count,
        "camera_declined": app_data.camera_declined,
        "ai_score": ai_score,
        "ai_feedback": ai_feedback or None,
        "status": status,
        "submit_time": datetime.utcnow().isoformat(),
        "attempt_number": attempt_number,
        "github_url": app_data.github_url or None,
        "portfolio_url": app_data.portfolio_url or None
    }).execute()

    if not app_res.data:
        raise HTTPException(status_code=500, detail="Failed to save application")

    application_id = app_res.data[0]["id"]

    # Save individual answers
    if app_data.answers:
        answers_payload = [
            {
                "application_id": application_id,
                "question_id": a.question_id or None,
                "question_text": a.question_text,
                "candidate_answer": a.candidate_answer
            }
            for a in app_data.answers
        ]
        supabase.table("answers").insert(answers_payload).execute()

    # Send HR email notification
    send_hr_notification(
        candidate_name=app_data.candidate_name,
        candidate_email=app_data.candidate_email,
        role_title=role_title,
        score=ai_score,
        status=status,
        tab_switches=app_data.tab_switch_count
    )

    # Send confirmation email to candidate
    send_candidate_confirmation(
        candidate_name=app_data.candidate_name,
        candidate_email=app_data.candidate_email,
        role_title=role_title,
        ai_score=ai_score,
        status=status
    )

    return {
        "message": "Assessment submitted successfully!",
        "application_id": application_id,
        "ai_score": ai_score,
        "status": status
    }


@app.get("/applications", summary="List applications with search, filters, pagination, sorting")
def list_applications(
    job_id:     Optional[str] = Query(None,  description="Filter by Job ID"),
    name:       Optional[str] = Query(None,  description="Search by candidate name (partial)"),
    email:      Optional[str] = Query(None,  description="Search by candidate email (partial)"),
    status:     Optional[str] = Query(None,  description="Filter by AI status: Passed/Failed"),
    hr_status:  Optional[str] = Query(None,  description="Filter by HR status: Shortlisted/Reviewing/Rejected"),
    min_score:  Optional[int] = Query(None,  description="Minimum AI score"),
    max_score:  Optional[int] = Query(None,  description="Maximum AI score"),
    page:       int           = Query(1,     ge=1,  description="Page number"),
    page_size:  int           = Query(20,    ge=1, le=100, description="Items per page"),
    sort_by:    str           = Query("created_at", description="Sort field: created_at|ai_score|candidate_name"),
    sort_dir:   str           = Query("desc", description="Sort direction: asc|desc"),
    hr=Depends(get_current_hr)
):
    """Search and filter applications with pagination and sorting."""
    allowed_sort = {"created_at", "ai_score", "candidate_name", "submit_time"}
    if sort_by not in allowed_sort:
        sort_by = "created_at"
    ascending = sort_dir.lower() == "asc"

    query = supabase.table("applications").select("*", count="exact")

    if job_id:   query = query.eq("job_id", job_id)
    if name:     query = query.ilike("candidate_name", f"%{name}%")
    if email:    query = query.ilike("candidate_email", f"%{email}%")
    if status:   query = query.eq("status", status)
    if hr_status:query = query.eq("hr_status", hr_status)
    if min_score is not None: query = query.gte("ai_score", min_score)
    if max_score is not None: query = query.lte("ai_score", max_score)

    query = query.order(sort_by, desc=not ascending)

    # Pagination
    offset = (page - 1) * page_size
    query = query.range(offset, offset + page_size - 1)

    res = query.execute()

    return {
        "data":       res.data or [],
        "total":      res.count or 0,
        "page":       page,
        "page_size":  page_size,
        "total_pages": max(1, -(-( res.count or 0) // page_size))
    }


@app.get("/applications/ranking", summary="Ranked leaderboard of candidates by AI score")
def get_ranking(
    job_id:    Optional[str] = Query(None, description="Filter by job ID to rank within a role"),
    hr_status: Optional[str] = Query(None, description="Filter by HR status"),
    limit:     int           = Query(50,   ge=1, le=200, description="Max candidates to return"),
    hr=Depends(get_current_hr)
):
    """
    Returns candidates ranked by AI score (highest first).
    Adds rank number, medal (for top 3), and percentile.
    Best attempt per candidate per job is used (no duplicates).
    """
    query = supabase.table("applications")\
        .select("id, candidate_name, candidate_email, ai_score, status, hr_status, tab_switch_count, job_id, attempt_number, submit_time, created_at")

    if job_id:
        query = query.eq("job_id", job_id)
    if hr_status:
        query = query.eq("hr_status", hr_status)

    res = query.order("ai_score", desc=True).execute()
    apps = res.data or []

    # Deduplicate — keep best attempt per candidate per job
    seen = {}
    for app in apps:
        key = (app["job_id"], app["candidate_email"])
        if key not in seen:
            seen[key] = app
        else:
            existing_score = seen[key].get("ai_score") or 0
            current_score  = app.get("ai_score") or 0
            if current_score > existing_score:
                seen[key] = app

    unique = sorted(seen.values(), key=lambda x: x.get("ai_score") or 0, reverse=True)
    unique = unique[:limit]

    total = len(unique)
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    # Fetch job titles for display
    job_ids = list({a["job_id"] for a in unique if a.get("job_id")})
    job_titles = {}
    if job_ids:
        jobs_res = supabase.table("jobs").select("id, role_title").in_("id", job_ids).execute()
        if jobs_res.data:
            job_titles = {j["id"]: j["role_title"] for j in jobs_res.data}

    ranked = []
    for i, app in enumerate(unique, 1):
        score = app.get("ai_score")
        percentile = round(((total - i) / total) * 100) if total > 1 else 100
        ranked.append({
            "rank":           i,
            "medal":          medals.get(i, ""),
            "id":             app["id"],
            "candidate_name": app["candidate_name"],
            "candidate_email":app["candidate_email"],
            "ai_score":       score,
            "status":         app.get("status", ""),
            "hr_status":      app.get("hr_status", ""),
            "tab_switch_count": app.get("tab_switch_count", 0),
            "job_id":         app.get("job_id", ""),
            "role_title":     job_titles.get(app.get("job_id", ""), "Unknown Role"),
            "percentile":     percentile,
            "submitted_at":   app.get("submit_time") or app.get("created_at", "")
        })

    return {"total": total, "ranking": ranked}


@app.get("/applications/{application_id}/answers", summary="Get answers for an application")
def get_answers(application_id: str):
    res = supabase.table("answers").select("*").eq("application_id", application_id).execute()
    if res.data is None:
        raise HTTPException(status_code=404, detail="No answers found")
    return res.data


@app.get("/applications/{application_id}/profile", summary="Full candidate profile")
def get_candidate_profile(application_id: str, hr=Depends(get_current_hr)):
    """Returns full candidate profile: application + answers + job info."""
    # Get application
    app_res = supabase.table("applications").select("*").eq("id", application_id).execute()
    if not app_res.data:
        raise HTTPException(status_code=404, detail="Application not found")
    application = app_res.data[0]

    # Get answers
    ans_res = supabase.table("answers").select("*").eq("application_id", application_id).execute()

    # Get job info
    job_res = supabase.table("jobs").select("*").eq("id", application["job_id"]).execute()
    job = job_res.data[0] if job_res.data else {}

    # Get all attempts by same candidate for same job (assessment history)
    history_res = supabase.table("applications")\
        .select("id, ai_score, status, attempt_number, submit_time, created_at, tab_switch_count")\
        .eq("job_id", application["job_id"])\
        .eq("candidate_email", application["candidate_email"])\
        .order("attempt_number")\
        .execute()

    return {
        "application": application,
        "answers":     ans_res.data or [],
        "job":         job,
        "history":     history_res.data or []
    }


@app.post("/applications/{application_id}/resume", summary="Upload candidate resume")
async def upload_resume(
    application_id: str,
    file: UploadFile = File(...),
):
    """
    Upload a resume (PDF or DOCX, max 5MB) for a candidate application.
    Stores the file in Supabase Storage and saves the public URL.
    """
    # Validate file type
    allowed_types = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/msword": ".doc"
    }
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and DOCX files are accepted."
        )

    # Validate file size (max 5MB)
    MAX_SIZE = 5 * 1024 * 1024  # 5MB
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 5MB. Your file is {len(content) // (1024*1024):.1f}MB."
        )

    # Validate application exists
    app_res = supabase.table("applications").select("id, candidate_name").eq("id", application_id).execute()
    if not app_res.data:
        raise HTTPException(status_code=404, detail="Application not found")

    # Build storage path: resumes/application_id/filename
    ext = allowed_types[content_type]
    storage_path = f"{application_id}/resume{ext}"

    try:
        # Upload to Supabase Storage bucket 'resumes'
        supabase.storage.from_("resumes").upload(
            path=storage_path,
            file=content,
            file_options={"content-type": content_type, "upsert": "true"}
        )

        # Get public URL
        url_res = supabase.storage.from_("resumes").get_public_url(storage_path)
        public_url = url_res if isinstance(url_res, str) else url_res.get("publicUrl", "")

        # Save URL to applications table
        supabase.table("applications").update({"resume_url": public_url}).eq("id", application_id).execute()

        return {"message": "Resume uploaded successfully", "resume_url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


class HRStatusUpdate(BaseModel):
    hr_status: Optional[str] = None   # Shortlisted / Reviewing / Rejected
    hr_notes: Optional[str] = None    # Internal HR notes


@app.patch("/applications/{application_id}/hr-status", summary="Update HR status and notes")
def update_hr_status(application_id: str, data: HRStatusUpdate, hr=Depends(get_current_hr)):
    update_data = {}
    if data.hr_status is not None:
        update_data["hr_status"] = data.hr_status
    if data.hr_notes is not None:
        update_data["hr_notes"] = data.hr_notes

    if not update_data:
        raise HTTPException(status_code=400, detail="Nothing to update")

    res = supabase.table("applications").update(update_data).eq("id", application_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Application not found")

    # Send shortlisted email if status changed to Shortlisted
    if data.hr_status == "Shortlisted":
        app_info = res.data[0]
        job_res = supabase.table("jobs").select("role_title").eq("id", app_info.get("job_id","")).execute()
        role_title = job_res.data[0]["role_title"] if job_res.data else "the position"
        send_shortlisted_email(
            candidate_name=app_info.get("candidate_name", "Candidate"),
            candidate_email=app_info.get("candidate_email", ""),
            role_title=role_title
        )

    if data.hr_status == "Rejected":
        app_info = res.data[0]
        job_res = supabase.table("jobs").select("role_title").eq("id", app_info.get("job_id","")).execute()
        role_title = job_res.data[0]["role_title"] if job_res.data else "the position"
        send_rejected_email(
            candidate_name=app_info.get("candidate_name", "Candidate"),
            candidate_email=app_info.get("candidate_email", ""),
            role_title=role_title
        )

    return {"message": "Updated successfully"}


# ──────────────────────────────────────────
# ANALYTICS ENDPOINT
# ──────────────────────────────────────────

@app.get("/analytics", summary="Dashboard analytics for HR")
def get_analytics(hr=Depends(get_current_hr)):
    """Returns all analytics data needed for the dashboard."""

    # Total candidates (unique emails in applications)
    all_apps = supabase.table("applications").select("id, candidate_name, candidate_email, ai_score, status, hr_status, created_at, job_id").execute()
    apps = all_apps.data or []

    # Active jobs count
    all_jobs = supabase.table("jobs").select("id, role_title, created_at").execute()
    jobs = all_jobs.data or []

    total_candidates   = len(apps)
    active_jobs        = len(jobs)
    assessments_done   = len([a for a in apps if a.get("status") in ["Passed", "Failed"]])
    shortlisted        = len([a for a in apps if a.get("hr_status") == "Shortlisted"])
    rejected           = len([a for a in apps if a.get("hr_status") == "Rejected"])

    # Average score (only where ai_score is not None)
    scores = [a["ai_score"] for a in apps if a.get("ai_score") is not None]
    avg_score = round(sum(scores) / len(scores)) if scores else 0

    # Monthly applications — last 6 months
    from collections import defaultdict
    from datetime import date
    monthly = defaultdict(int)
    for a in apps:
        try:
            dt = datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
            key = dt.strftime("%b %Y")
            monthly[key] += 1
        except Exception:
            pass

    # Get last 6 months in order
    today = date.today()
    month_labels = []
    month_values = []
    for i in range(5, -1, -1):
        from datetime import timedelta
        # first day of month i months ago
        d = today.replace(day=1)
        for _ in range(i):
            d = (d - timedelta(days=1)).replace(day=1)
        label = d.strftime("%b %Y")
        month_labels.append(d.strftime("%b"))
        month_values.append(monthly.get(label, 0))

    # Top 5 candidates by ai_score
    scored_apps = sorted(
        [a for a in apps if a.get("ai_score") is not None],
        key=lambda x: x["ai_score"],
        reverse=True
    )[:5]

    top_candidates = [
        {
            "name": a.get("candidate_name", "Unknown"),
            "email": a.get("candidate_email", ""),
            "score": a.get("ai_score", 0),
            "status": a.get("status", ""),
            "hr_status": a.get("hr_status", "")
        }
        for a in scored_apps
    ]

    # Score distribution buckets
    score_dist = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for s in scores:
        if s <= 20:   score_dist["0-20"] += 1
        elif s <= 40: score_dist["21-40"] += 1
        elif s <= 60: score_dist["41-60"] += 1
        elif s <= 80: score_dist["61-80"] += 1
        else:         score_dist["81-100"] += 1

    # Pass vs Fail
    passed = len([a for a in apps if a.get("status") == "Passed"])
    failed = len([a for a in apps if a.get("status") == "Failed"])

    return {
        "total_candidates":    total_candidates,
        "active_jobs":         active_jobs,
        "assessments_done":    assessments_done,
        "avg_score":           avg_score,
        "shortlisted":         shortlisted,
        "rejected":            rejected,
        "monthly_labels":      month_labels,
        "monthly_values":      month_values,
        "top_candidates":      top_candidates,
        "score_distribution":  score_dist,
        "passed":              passed,
        "failed":              failed
    }


# ──────────────────────────────────────────
# EMAIL TRIGGER ENDPOINTS
# ──────────────────────────────────────────

class InterviewScheduleRequest(BaseModel):
    application_id: str
    interview_details: str  # e.g. "Date: 25 July 2026\nTime: 3:00 PM IST\nMode: Google Meet"

class AssessmentInviteRequest(BaseModel):
    candidate_email: str
    job_id: str
    portal_base_url: str  # e.g. "http://localhost:3000/"


@app.post("/notifications/schedule-interview", summary="Send interview scheduled email to candidate")
def schedule_interview(data: InterviewScheduleRequest, hr=Depends(get_current_hr)):
    """Sends interview schedule notification to a candidate."""
    app_res = supabase.table("applications").select("candidate_name, candidate_email, job_id").eq("id", data.application_id).execute()
    if not app_res.data:
        raise HTTPException(status_code=404, detail="Application not found")

    app_info = app_res.data[0]
    job_res = supabase.table("jobs").select("role_title").eq("id", app_info["job_id"]).execute()
    role_title = job_res.data[0]["role_title"] if job_res.data else "the position"

    send_interview_scheduled_email(
        candidate_name=app_info["candidate_name"],
        candidate_email=app_info["candidate_email"],
        role_title=role_title,
        interview_details=data.interview_details
    )
    return {"message": f"Interview notification sent to {app_info['candidate_email']}"}


@app.post("/notifications/send-assessment-invite", summary="Send assessment invitation email to a candidate")
def send_assessment_invite(data: AssessmentInviteRequest, hr=Depends(get_current_hr)):
    """Sends assessment invitation email with portal link to a candidate."""
    job_res = supabase.table("jobs").select("role_title, time_limit_minutes").eq("id", data.job_id).execute()
    if not job_res.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_res.data[0]
    hr_res = supabase.table("hr_users").select("company_name").eq("email", hr["email"]).execute()
    company = hr_res.data[0]["company_name"] if hr_res.data else "CorpoCalm Gateway"

    portal_link = f"{data.portal_base_url.rstrip('/')}/job-portal.html?job={data.job_id}"

    send_assessment_assigned_email(
        candidate_email=data.candidate_email,
        role_title=job["role_title"],
        company_name=company,
        portal_link=portal_link,
        time_limit=job["time_limit_minutes"]
    )
    return {"message": f"Assessment invitation sent to {data.candidate_email}"}


# ──────────────────────────────────────────
# RECRUITER NOTES ENDPOINTS
# ──────────────────────────────────────────

class NoteCreate(BaseModel):
    note_text: str = Field(..., min_length=1, description="Note content")

class NoteUpdate(BaseModel):
    note_text: str = Field(..., min_length=1, description="Updated note content")


@app.get("/applications/{application_id}/notes", summary="Get all recruiter notes for an application")
def get_notes(application_id: str, hr=Depends(get_current_hr)):
    res = supabase.table("recruiter_notes")\
        .select("*")\
        .eq("application_id", application_id)\
        .order("created_at", desc=False)\
        .execute()
    return res.data or []


@app.post("/applications/{application_id}/notes", summary="Add a recruiter note")
def add_note(application_id: str, data: NoteCreate, hr=Depends(get_current_hr)):
    # Verify application exists
    app_res = supabase.table("applications").select("id").eq("id", application_id).execute()
    if not app_res.data:
        raise HTTPException(status_code=404, detail="Application not found")

    res = supabase.table("recruiter_notes").insert({
        "application_id": application_id,
        "hr_email": hr["email"],
        "hr_name":  hr.get("email", "HR"),  # will be enriched below
        "note_text": data.note_text
    }).execute()

    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create note")

    # Enrich with actual HR name
    hr_res = supabase.table("hr_users").select("name").eq("email", hr["email"]).execute()
    hr_name = hr_res.data[0]["name"] if hr_res.data else hr["email"]
    supabase.table("recruiter_notes").update({"hr_name": hr_name}).eq("id", res.data[0]["id"]).execute()
    res.data[0]["hr_name"] = hr_name

    return res.data[0]


@app.patch("/applications/{application_id}/notes/{note_id}", summary="Edit a recruiter note")
def edit_note(application_id: str, note_id: str, data: NoteUpdate, hr=Depends(get_current_hr)):
    # Verify note belongs to this HR
    note_res = supabase.table("recruiter_notes")\
        .select("*")\
        .eq("id", note_id)\
        .eq("application_id", application_id)\
        .execute()

    if not note_res.data:
        raise HTTPException(status_code=404, detail="Note not found")

    if note_res.data[0]["hr_email"] != hr["email"]:
        raise HTTPException(status_code=403, detail="You can only edit your own notes")

    res = supabase.table("recruiter_notes").update({
        "note_text":  data.note_text,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", note_id).execute()

    return res.data[0] if res.data else {"message": "Updated"}


@app.delete("/applications/{application_id}/notes/{note_id}", summary="Delete a recruiter note")
def delete_note(application_id: str, note_id: str, hr=Depends(get_current_hr)):
    # Verify note belongs to this HR
    note_res = supabase.table("recruiter_notes")\
        .select("hr_email")\
        .eq("id", note_id)\
        .eq("application_id", application_id)\
        .execute()

    if not note_res.data:
        raise HTTPException(status_code=404, detail="Note not found")

    if note_res.data[0]["hr_email"] != hr["email"]:
        raise HTTPException(status_code=403, detail="You can only delete your own notes")

    supabase.table("recruiter_notes").delete().eq("id", note_id).execute()
    return {"message": "Note deleted successfully"}
