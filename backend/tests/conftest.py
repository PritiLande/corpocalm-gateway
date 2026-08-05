"""
Pytest configuration for CorpoCalm Gateway backend tests.
Sets required environment variables before any test module is imported.
"""
import os

# Must be set before importing main
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-supabase-key")
os.environ.setdefault("JWT_SECRET",   "test-secret-key-for-unit-tests-only")
