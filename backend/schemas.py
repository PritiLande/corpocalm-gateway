from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime


# ==========================================
# 1. JOBS
# ==========================================
class JobCreate(BaseModel):
    role_title: str = Field("", max_length=255, description="Title of the job role")
    time_limit_minutes: Optional[int] = Field(15, ge=1, description="Test time limit in minutes")
    passing_threshold: Optional[int] = Field(70, ge=0, le=100, description="Passing percentage threshold")
    questions: Optional[List[str]] = Field(default_factory=list, description="List of screening questions")
    # Job portal display fields (all optional)
    description: Optional[str] = Field(None, description="Full job description")
    skills_required: Optional[str] = Field(None, description="Comma-separated skills")
    location: Optional[str] = Field(None, description="Job location e.g. Remote / Mumbai")
    job_type: Optional[str] = Field(None, description="Full-Time, Part-Time, Contract, etc.")
    salary_range: Optional[str] = Field(None, description="Salary range e.g. 8-12 LPA")
    experience_required: Optional[str] = Field(None, description="Experience e.g. 2-4 years")
    expiry_date: Optional[str] = Field(None, description="Last date to apply e.g. 2026-08-01")


class JobResponse(BaseModel):
    id: UUID
    role_title: str
    time_limit_minutes: int
    passing_threshold: int
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# 2. SCREENING QUESTIONS
# ==========================================
class QuestionCreate(BaseModel):
    job_id: UUID
    question_text: str = Field(..., min_length=5, description="The content of the question")


class QuestionResponse(BaseModel):
    id: UUID
    job_id: UUID
    question_text: str
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# 3. ANSWERS (one row per question per application)
# ==========================================
class AnswerSubmit(BaseModel):
    question_id: Optional[str] = Field(None, description="UUID of the screening question")
    question_text: str = Field(..., description="The question text (stored for reference)")
    candidate_answer: str = Field("", description="Candidate's answer")


class AnswerResponse(BaseModel):
    id: UUID
    application_id: UUID
    question_id: Optional[UUID]
    question_text: str
    candidate_answer: str
    created_at: datetime

    class Config:
        from_attributes = True


# ==========================================
# 4. APPLICATIONS
# ==========================================
class ApplicationCreate(BaseModel):
    job_id: str = Field(..., description="UUID of the job")
    candidate_name: str = Field(..., min_length=2, max_length=255, description="Full name of the candidate")
    candidate_email: EmailStr = Field(..., description="Email address (validated format)")
    tab_switch_count: int = Field(0, ge=0, description="Anti-cheat: number of tab switches detected")
    face_away_count: int = Field(0, ge=0, description="Proctoring: number of times face not detected")
    camera_declined: bool = Field(False, description="Proctoring: whether candidate denied camera access")
    answers: List[AnswerSubmit] = Field(default_factory=list, description="List of answers")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio_url: Optional[str] = Field(None, description="Portfolio or personal website URL")


class ApplicationResponse(BaseModel):
    id: UUID
    job_id: UUID
    candidate_name: str
    candidate_email: str
    tab_switch_count: int
    ai_score: Optional[int] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
