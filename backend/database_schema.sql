-- ============================================================
-- CorpoCalm Gateway - Full Database Schema
-- Run this in Supabase SQL Editor (Dashboard > SQL Editor)
-- ============================================================

-- Step 0a: Companies Table
CREATE TABLE IF NOT EXISTS companies (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         VARCHAR(255) NOT NULL,
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 0b: HR Users Table (login/signup)
-- role: 'admin' = company owner, 'member' = invited team member
CREATE TABLE IF NOT EXISTS hr_users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           VARCHAR(255) NOT NULL,
    email          VARCHAR(255) UNIQUE NOT NULL,
    password_hash  TEXT NOT NULL,
    company_id     UUID REFERENCES companies(id) ON DELETE SET NULL,
    company_name   VARCHAR(255),
    role           VARCHAR(50) NOT NULL DEFAULT 'admin',
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 0c: Team Invites Table (pending invitations)
CREATE TABLE IF NOT EXISTS team_invites (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id     UUID REFERENCES companies(id) ON DELETE CASCADE,
    invited_email  VARCHAR(255) NOT NULL,
    invited_by     UUID REFERENCES hr_users(id) ON DELETE SET NULL,
    token          VARCHAR(255) UNIQUE NOT NULL,
    status         VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending / accepted
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days')
);

-- ── Migration: add company_id + role to existing hr_users ──
-- ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS company_id UUID REFERENCES companies(id) ON DELETE SET NULL;
-- ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS role VARCHAR(50) NOT NULL DEFAULT 'admin';
-- ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS company_name VARCHAR(255);

-- Step 1: Jobs Table (includes job portal display fields)
CREATE TABLE IF NOT EXISTS jobs (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id           UUID REFERENCES companies(id) ON DELETE SET NULL,
    status               VARCHAR(20) NOT NULL DEFAULT 'published',  -- 'draft' or 'published'
    role_title           VARCHAR(255) NOT NULL,
    time_limit_minutes   INT NOT NULL DEFAULT 15,
    passing_threshold    INT NOT NULL DEFAULT 70,
    -- Job portal display fields
    description          TEXT,
    skills_required      TEXT,
    location             VARCHAR(255),
    job_type             VARCHAR(100),
    salary_range         VARCHAR(100),
    experience_required  VARCHAR(100),
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 2: Screening Questions Table
CREATE TABLE IF NOT EXISTS screening_questions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id        UUID REFERENCES jobs(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 3: Applications Table
CREATE TABLE IF NOT EXISTS applications (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id             UUID REFERENCES jobs(id) ON DELETE CASCADE,
    candidate_name     VARCHAR(255) NOT NULL DEFAULT '',
    candidate_email    VARCHAR(255) NOT NULL,
    tab_switch_count   INT DEFAULT 0,
    ai_score           INT,
    status             VARCHAR(50) DEFAULT 'Submitted',
    attempt_number     INT NOT NULL DEFAULT 1,
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT check_attempt_limit CHECK (attempt_number <= 2),
    CONSTRAINT unique_job_candidate_attempt UNIQUE (job_id, candidate_email, attempt_number)
);

-- Step 4: Answers Table
CREATE TABLE IF NOT EXISTS answers (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id   UUID REFERENCES applications(id) ON DELETE CASCADE,
    question_id      UUID REFERENCES screening_questions(id) ON DELETE SET NULL,
    question_text    TEXT NOT NULL,
    candidate_answer TEXT DEFAULT '',
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- IF YOU ALREADY HAVE THE JOBS TABLE, run this to add new columns:
-- ============================================================
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS description TEXT;
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS skills_required TEXT;
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS location VARCHAR(255);
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS job_type VARCHAR(100);
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS salary_range VARCHAR(100);
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS experience_required VARCHAR(100);
-- ALTER TABLE applications ADD COLUMN IF NOT EXISTS candidate_name VARCHAR(255) NOT NULL DEFAULT '';
-- ALTER TABLE applications DROP COLUMN IF EXISTS feedback;
-- ALTER TABLE applications ADD COLUMN IF NOT EXISTS github_url TEXT;
-- ALTER TABLE applications ADD COLUMN IF NOT EXISTS portfolio_url TEXT;
-- ALTER TABLE applications ADD COLUMN IF NOT EXISTS ai_feedback TEXT;
-- ALTER TABLE applications ADD COLUMN IF NOT EXISTS face_away_count INT DEFAULT 0;
-- ALTER TABLE applications ADD COLUMN IF NOT EXISTS camera_declined BOOLEAN DEFAULT FALSE;
-- ============================================================

-- ============================================================
-- Migration 002: Multi-HR Team Support
-- Run these if you already have the tables above:
-- ============================================================
-- ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS company_id UUID REFERENCES companies(id) ON DELETE SET NULL;
-- ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS role VARCHAR(50) NOT NULL DEFAULT 'admin';
-- ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS company_name VARCHAR(255);
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS company_id UUID REFERENCES companies(id) ON DELETE SET NULL;
-- ALTER TABLE jobs ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'published';
-- CREATE TABLE IF NOT EXISTS team_invites (
--     id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--     company_id     UUID REFERENCES companies(id) ON DELETE CASCADE,
--     invited_email  VARCHAR(255) NOT NULL,
--     invited_by     UUID REFERENCES hr_users(id) ON DELETE SET NULL,
--     token          VARCHAR(255) UNIQUE NOT NULL,
--     status         VARCHAR(50) NOT NULL DEFAULT 'pending',
--     created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
--     expires_at     TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days')
-- );
-- ============================================================

-- ============================================================
-- Migration 003: Password Reset Tokens
-- ============================================================
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hr_id       UUID REFERENCES hr_users(id) ON DELETE CASCADE UNIQUE,
    token       VARCHAR(255) UNIQUE NOT NULL,
    expires_at  TIMESTAMP WITH TIME ZONE NOT NULL,
    used        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
-- ============================================================
