"""
CorpoCalm Gateway — Feature Regression Tests
=============================================
Confirms these features still work after all security fixes:
  1.  HR Signup
  2.  HR Login
  3.  Job Creation
  4.  Candidate Application Submission
  5.  AI Generation (questions, description, skills, full job)
  6.  Dashboard (GET /applications)
  7.  Analytics (GET /analytics)
  8.  Emails (send functions called without crashing)
  9.  Draft Save & Publish
  10. Candidate Profile

All Supabase and external calls are mocked — no real DB or network required.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone

os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-supabase-key")
os.environ.setdefault("JWT_SECRET",   "test-secret-key-for-unit-tests-only")

from fastapi.testclient import TestClient

with patch("supabase.create_client"), patch("groq.Groq"):
    from main import app, create_token

client = TestClient(app, raise_server_exceptions=False)

# ──────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────

def _hr(hr_id="hr-1", email="hr@acme.com", company_id="co-1", role="admin"):
    r = MagicMock()
    r.data = [{
        "id": hr_id, "email": email, "name": "HR User",
        "role": role, "company_id": company_id,
        "company_name": "Acme Corp", "password_hash": "x"
    }]
    return r

def _ok(data=None):
    r = MagicMock()
    r.data = data or [{"id": "new-id"}]
    r.count = len(r.data)
    return r

def _empty():
    r = MagicMock()
    r.data = []
    r.count = 0
    return r

def token(hr_id="hr-1", email="hr@acme.com") -> str:
    return create_token(hr_id=hr_id, email=email)

def auth(hr_id="hr-1", email="hr@acme.com") -> dict:
    return {"Authorization": f"Bearer {token(hr_id, email)}"}

def mock_hr_lookup(mock_sb, company_id="co-1", role="admin"):
    """Make every hr_users lookup return a valid HR row."""
    mock_sb.table.return_value.select.return_value\
        .eq.return_value.execute.return_value = _hr(
            company_id=company_id, role=role)


# ──────────────────────────────────────────────────────────────
# 1. HR SIGNUP
# ──────────────────────────────────────────────────────────────

class TestHRSignup:

    def test_signup_creates_company_and_user(self):
        company_res = _ok([{"id": "co-new"}])
        user_res    = _ok([{"id": "hr-new", "name": "Priti", "email": "priti@test.com"}])
        no_existing = _empty()

        with patch("main.supabase") as sb, \
             patch("main.send_welcome_email") as mock_mail:

            sb.table.return_value.select.return_value\
                .eq.return_value.execute.return_value = no_existing
            sb.table.return_value.insert.return_value\
                .execute.side_effect = [company_res, user_res]

            res = client.post("/hr/signup", json={
                "name": "Priti",
                "company_name": "Acme",
                "email": "priti@test.com",
                "password": "secure123"
            })

        assert res.status_code == 200
        assert "created" in res.json()["message"].lower()
        mock_mail.assert_called_once()

    def test_signup_duplicate_email_returns_400(self):
        with patch("main.supabase") as sb:
            sb.table.return_value.select.return_value\
                .eq.return_value.execute.return_value = _ok([{"id": "existing"}])

            res = client.post("/hr/signup", json={
                "name": "Priti",
                "company_name": "Acme",
                "email": "taken@test.com",
                "password": "secure123"
            })

        assert res.status_code == 400

    def test_signup_short_password_returns_400(self):
        res = client.post("/hr/signup", json={
            "name": "Priti", "company_name": "Acme",
            "email": "priti@test.com", "password": "abc"
        })
        assert res.status_code == 400  # endpoint-level check (< 6 chars)


# ──────────────────────────────────────────────────────────────
# 2. HR LOGIN
# ──────────────────────────────────────────────────────────────

class TestHRLogin:

    def test_valid_login_returns_token_and_user_info(self):
        import bcrypt
        hashed = bcrypt.hashpw(b"mypassword", bcrypt.gensalt()).decode()
        hr_row = _ok([{
            "id": "hr-1", "email": "hr@acme.com", "name": "HR",
            "password_hash": hashed, "role": "admin",
            "company_id": "co-1", "company_name": "Acme"
        }])

        with patch("main.supabase") as sb:
            sb.table.return_value.select.return_value\
                .eq.return_value.execute.return_value = hr_row

            res = client.post("/hr/login", json={
                "email": "hr@acme.com", "password": "mypassword"
            })

        assert res.status_code == 200
        body = res.json()
        assert "token" in body
        assert body["email"] == "hr@acme.com"
        assert body["role"] == "admin"
        assert body["company_id"] == "co-1"

    def test_wrong_password_returns_401(self):
        import bcrypt
        hashed = bcrypt.hashpw(b"correct", bcrypt.gensalt()).decode()
        hr_row = _ok([{
            "id": "hr-1", "email": "hr@acme.com", "name": "HR",
            "password_hash": hashed, "role": "admin",
            "company_id": "co-1", "company_name": "Acme"
        }])

        with patch("main.supabase") as sb:
            sb.table.return_value.select.return_value\
                .eq.return_value.execute.return_value = hr_row

            res = client.post("/hr/login", json={
                "email": "hr@acme.com", "password": "wrong"
            })

        assert res.status_code == 401


# ──────────────────────────────────────────────────────────────
# 3. JOB CREATION
# ──────────────────────────────────────────────────────────────

class TestJobCreation:

    def test_create_published_job_returns_job_id(self):
        job_res = _ok([{"id": "job-123", "role_title": "Python Dev"}])
        q_res   = _ok([{"id": "q-1"}])

        with patch("main.supabase") as sb:
            mock_hr_lookup(sb)
            sb.table.return_value.insert.return_value\
                .execute.side_effect = [job_res, q_res]

            res = client.post("/jobs", json={
                "role_title": "Python Developer",
                "time_limit_minutes": 30,
                "passing_threshold": 70,
                "questions": ["What is Python?", "Explain OOP"]
            }, headers=auth())

        assert res.status_code == 200
        assert "job_id" in res.json()

    def test_create_job_without_questions_returns_400(self):
        with patch("main.supabase") as sb:
            mock_hr_lookup(sb)

            res = client.post("/jobs", json={
                "role_title": "Python Developer",
                "time_limit_minutes": 30,
                "questions": []
            }, headers=auth())

        assert res.status_code == 400

    def test_save_draft_returns_draft_status(self):
        draft_res = _ok([{"id": "draft-1", "role_title": "Draft Job"}])

        with patch("main.supabase") as sb:
            mock_hr_lookup(sb)
            sb.table.return_value.insert.return_value\
                .execute.return_value = draft_res

            res = client.post("/jobs/draft", json={
                "role_title": "Draft Job",
                "time_limit_minutes": 20
            }, headers=auth())

        assert res.status_code == 200
        assert res.json()["status"] == "draft"


# ──────────────────────────────────────────────────────────────
# 4. CANDIDATE APPLICATION SUBMISSION
# ──────────────────────────────────────────────────────────────

class TestCandidateApplication:

    def test_submit_application_returns_score_and_status(self):
        job_res  = _ok([{"id": "job-1", "role_title": "Python Dev", "passing_threshold": 70}])
        app_res  = _ok([{"id": "app-1"}])
        ans_res  = _ok([])

        with patch("main.supabase") as sb, \
             patch("main.compute_ai_score", return_value=(80, "Good answers")), \
             patch("main.send_hr_notification"), \
             patch("main.send_candidate_confirmation"):

            def table_side(name):
                t = MagicMock()
                if name == "jobs":
                    t.select.return_value.eq.return_value\
                        .execute.return_value = job_res
                elif name == "applications":
                    # First call: attempt check (empty), second: insert
                    t.select.return_value.select.return_value\
                        .eq.return_value.eq.return_value\
                        .execute.return_value = _empty()
                    t.select.return_value.eq.return_value\
                        .eq.return_value.execute.return_value = _empty()
                    t.insert.return_value.execute.return_value = app_res
                elif name == "answers":
                    t.insert.return_value.execute.return_value = ans_res
                return t

            sb.table.side_effect = table_side

            res = client.post("/applications", json={
                "job_id": "job-1",
                "candidate_name": "Test Candidate",
                "candidate_email": "test@candidate.com",
                "tab_switch_count": 0,
                "face_away_count": 0,
                "camera_declined": False,
                "answers": [
                    {"question_text": "What is Python?", "candidate_answer": "A language"}
                ]
            })

        assert res.status_code == 200
        body = res.json()
        assert "application_id" in body
        assert "ai_score" in body
        assert "status" in body

    def test_answer_too_long_returns_400(self):
        with patch("main.supabase") as sb, \
             patch("main.compute_ai_score", return_value=(50, "")), \
             patch("main.send_hr_notification"), \
             patch("main.send_candidate_confirmation"):

            sb.table.return_value.select.return_value\
                .eq.return_value.execute.return_value = _ok([
                    {"id": "job-1", "role_title": "Dev", "passing_threshold": 70}
                ])
            sb.table.return_value.select.return_value\
                .eq.return_value.eq.return_value\
                .execute.return_value = _empty()

            res = client.post("/applications", json={
                "job_id": "job-1",
                "candidate_name": "Test",
                "candidate_email": "t@t.com",
                "tab_switch_count": 0,
                "face_away_count": 0,
                "camera_declined": False,
                "answers": [
                    {"question_text": "Q?", "candidate_answer": "x" * 5001}
                ]
            })

        assert res.status_code in (400, 422)  # rejected by Pydantic schema or endpoint guard


# ──────────────────────────────────────────────────────────────
# 5. AI GENERATION ENDPOINTS
# ──────────────────────────────────────────────────────────────

class TestAIGeneration:

    def _mock_groq_response(self, text: str):
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_generate_questions_returns_list(self):
        questions_text = "What is Python?\nExplain OOP\nWhat is a REST API?"

        with patch("main.supabase") as sb, \
             patch("main.groq_client") as mock_groq:
            mock_hr_lookup(sb)
            mock_groq.chat.completions.create.return_value = \
                self._mock_groq_response(questions_text)

            res = client.post("/generate-questions", json={
                "role_title": "Python Developer"
            }, headers=auth())

        assert res.status_code == 200
        assert "questions" in res.json()
        assert len(res.json()["questions"]) > 0

    def test_generate_description_returns_text(self):
        desc = "We are looking for a skilled Python developer to join our team."

        with patch("main.supabase") as sb, \
             patch("main.groq_client") as mock_groq:
            mock_hr_lookup(sb)
            mock_groq.chat.completions.create.return_value = \
                self._mock_groq_response(desc)

            res = client.post("/generate-description", json={
                "role_title": "Python Developer"
            }, headers=auth())

        assert res.status_code == 200
        assert "description" in res.json()

    def test_generate_skills_returns_comma_separated(self):
        skills = "Python, FastAPI, PostgreSQL, Docker, REST APIs"

        with patch("main.supabase") as sb, \
             patch("main.groq_client") as mock_groq:
            mock_hr_lookup(sb)
            mock_groq.chat.completions.create.return_value = \
                self._mock_groq_response(skills)

            res = client.post("/generate-skills", json={
                "role_title": "Python Developer"
            }, headers=auth())

        assert res.status_code == 200
        assert "skills" in res.json()

    def test_generate_full_job_returns_all_fields(self):
        full_job = '{"description":"Great role","skills_required":"Python,FastAPI","questions":["Q1","Q2","Q3","Q4","Q5"],"job_type":"Full-Time","experience_required":"2-4 years","salary_range":"8-12 LPA","time_limit_minutes":25,"passing_threshold":65}'

        with patch("main.supabase") as sb, \
             patch("main.groq_client") as mock_groq:
            mock_hr_lookup(sb)
            mock_groq.chat.completions.create.return_value = \
                self._mock_groq_response(full_job)

            res = client.post("/generate-full-job", json={
                "role_title": "Python Developer"
            }, headers=auth())

        assert res.status_code == 200
        body = res.json()
        for field in ["description", "skills_required", "questions",
                      "job_type", "experience_required", "salary_range",
                      "time_limit_minutes", "passing_threshold"]:
            assert field in body, f"Missing field: {field}"


# ──────────────────────────────────────────────────────────────
# 6. DASHBOARD (GET /applications)
# ──────────────────────────────────────────────────────────────

class TestDashboard:

    def test_list_applications_returns_paginated_data(self):
        apps = [
            {"id": "app-1", "candidate_name": "Alice", "candidate_email": "a@a.com",
             "ai_score": 80, "status": "Passed", "tab_switch_count": 0, "created_at": "2026-01-01T00:00:00"},
            {"id": "app-2", "candidate_name": "Bob",   "candidate_email": "b@b.com",
             "ai_score": 45, "status": "Failed", "tab_switch_count": 1, "created_at": "2026-01-02T00:00:00"},
        ]
        jobs_res = _ok([{"id": "job-1"}])
        apps_res = MagicMock()
        apps_res.data = apps
        apps_res.count = 2

        with patch("main.supabase") as sb:
            def table_side(name):
                t = MagicMock()
                if name == "hr_users":
                    t.select.return_value.eq.return_value\
                        .execute.return_value = _hr()
                elif name == "jobs":
                    t.select.return_value.eq.return_value\
                        .execute.return_value = jobs_res
                elif name == "applications":
                    q = MagicMock()
                    q.execute.return_value = apps_res
                    q.in_.return_value = q
                    q.ilike.return_value = q
                    q.eq.return_value = q
                    q.gte.return_value = q
                    q.lte.return_value = q
                    q.order.return_value = q
                    q.range.return_value = q
                    t.select.return_value = q
                return t
            sb.table.side_effect = table_side

            res = client.get("/applications", headers=auth())

        assert res.status_code == 200
        body = res.json()
        assert "data" in body
        assert "total" in body
        assert body["total"] == 2
        assert len(body["data"]) == 2


# ──────────────────────────────────────────────────────────────
# 7. ANALYTICS
# ──────────────────────────────────────────────────────────────

class TestAnalytics:

    def test_analytics_returns_expected_keys(self):
        jobs = [{"id": "job-1", "role_title": "Dev", "created_at": "2026-01-01T00:00:00"}]
        apps = [
            {"id": "a1", "candidate_name": "Alice", "candidate_email": "a@a.com",
             "ai_score": 80, "status": "Passed", "hr_status": "Shortlisted",
             "created_at": "2026-01-01T00:00:00", "job_id": "job-1"},
            {"id": "a2", "candidate_name": "Bob", "candidate_email": "b@b.com",
             "ai_score": 40, "status": "Failed", "hr_status": None,
             "created_at": "2026-01-02T00:00:00", "job_id": "job-1"},
        ]

        with patch("main.supabase") as sb:
            def table_side(name):
                t = MagicMock()
                if name == "hr_users":
                    t.select.return_value.eq.return_value\
                        .execute.return_value = _hr()
                elif name == "jobs":
                    t.select.return_value.eq.return_value\
                        .execute.return_value = _ok(jobs)
                elif name == "applications":
                    q = MagicMock()
                    q.execute.return_value = _ok(apps)
                    q.in_.return_value = q
                    t.select.return_value = q
                return t
            sb.table.side_effect = table_side

            res = client.get("/analytics", headers=auth())

        assert res.status_code == 200
        body = res.json()
        for key in ["total_candidates", "active_jobs", "assessments_done",
                    "avg_score", "shortlisted", "rejected",
                    "monthly_labels", "monthly_values",
                    "top_candidates", "score_distribution",
                    "passed", "failed"]:
            assert key in body, f"Missing analytics key: {key}"

        assert body["total_candidates"] == 2
        assert body["active_jobs"] == 1
        assert body["passed"] == 1
        assert body["failed"] == 1
        assert body["shortlisted"] == 1


# ──────────────────────────────────────────────────────────────
# 8. EMAIL FUNCTIONS
# ──────────────────────────────────────────────────────────────

class TestEmails:
    """
    Email functions should silently skip when SMTP is not configured.
    They must not raise exceptions — they are fire-and-forget.
    """

    def test_candidate_confirmation_skips_gracefully_without_smtp(self):
        from main import send_candidate_confirmation
        # No SMTP configured — should not raise
        send_candidate_confirmation(
            candidate_name="Alice",
            candidate_email="alice@test.com",
            role_title="Python Developer",
            ai_score=75,
            status="Passed"
        )

    def test_welcome_email_skips_gracefully_without_smtp(self):
        from main import send_welcome_email
        send_welcome_email(
            hr_name="Priti",
            hr_email="priti@test.com",
            company_name="Acme Corp"
        )

    def test_shortlisted_email_skips_gracefully_without_smtp(self):
        from main import send_shortlisted_email
        send_shortlisted_email(
            candidate_name="Alice",
            candidate_email="alice@test.com",
            role_title="Python Developer"
        )

    def test_rejected_email_skips_gracefully_without_smtp(self):
        from main import send_rejected_email
        send_rejected_email(
            candidate_name="Bob",
            candidate_email="bob@test.com",
            role_title="Python Developer"
        )

    def test_hr_notification_skips_gracefully_without_smtp(self):
        from main import send_hr_notification
        send_hr_notification(
            candidate_name="Alice",
            candidate_email="alice@test.com",
            role_title="Python Developer",
            score=80,
            status="Passed",
            tab_switches=0
        )
