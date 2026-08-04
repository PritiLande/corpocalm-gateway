-- ============================================================
-- Migration 001: Search, Candidate Profile & Resume Upload
-- Run this in Supabase SQL Editor
-- ============================================================

-- Add resume_url to applications
ALTER TABLE applications ADD COLUMN IF NOT EXISTS resume_url TEXT;

-- Add submit_time if missing
ALTER TABLE applications ADD COLUMN IF NOT EXISTS submit_time TIMESTAMP WITH TIME ZONE;

-- Add hr_status if missing
ALTER TABLE applications ADD COLUMN IF NOT EXISTS hr_status VARCHAR(50);

-- Add hr_notes if missing
ALTER TABLE applications ADD COLUMN IF NOT EXISTS hr_notes TEXT;

-- Add company_name to hr_users if missing
ALTER TABLE hr_users ADD COLUMN IF NOT EXISTS company_name VARCHAR(255);

-- Add expiry_date to jobs if missing
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS expiry_date DATE;

-- Indexes for search performance
CREATE INDEX IF NOT EXISTS idx_applications_candidate_name  ON applications(candidate_name);
CREATE INDEX IF NOT EXISTS idx_applications_candidate_email ON applications(candidate_email);
CREATE INDEX IF NOT EXISTS idx_applications_status          ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_hr_status       ON applications(hr_status);
CREATE INDEX IF NOT EXISTS idx_applications_ai_score        ON applications(ai_score);
CREATE INDEX IF NOT EXISTS idx_applications_job_id          ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_applications_created_at      ON applications(created_at DESC);

-- ============================================================
-- Supabase Storage: Create 'resumes' bucket
-- Run this separately or do it from the Supabase Storage UI:
-- Storage → New Bucket → Name: resumes → Public: ON
-- ============================================================
