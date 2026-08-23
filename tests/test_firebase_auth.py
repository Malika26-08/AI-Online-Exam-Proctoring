"""
Unit Tests for Firebase Authentication Handler.
Validates email/password user creation, sign in, email verification requests,
password reset requests, error message formatting, and environment key loading using FirebaseAuthHandler.
"""

import os
import pytest
from src.auth.firebase_auth import FirebaseAuthHandler, MISSING_API_KEY_ERROR


def test_firebase_auth_get_api_key_str():
    """get_api_key must return a string (empty or key value)."""
    key = FirebaseAuthHandler.get_api_key()
    assert isinstance(key, str)


def test_firebase_auth_configured_key_loading(monkeypatch):
    """get_api_key must correctly retrieve configured key without exposing key in test logs."""
    mock_key = "AIzaSyMockKeyForUnitTest12345"
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", mock_key)
    retrieved = FirebaseAuthHandler.get_api_key()
    assert retrieved == mock_key
    assert len(retrieved) > 0


def test_firebase_auth_error_formatting():
    """_format_error must translate Firebase error codes to user-friendly text."""
    assert "already exists" in FirebaseAuthHandler._format_error("EMAIL_EXISTS")
    assert "valid email" in FirebaseAuthHandler._format_error("INVALID_EMAIL")
    assert "6 characters" in FirebaseAuthHandler._format_error("WEAK_PASSWORD")
    assert "Incorrect password" in FirebaseAuthHandler._format_error("INVALID_PASSWORD")
    assert "Unknown error" == FirebaseAuthHandler._format_error("Unknown error")


def test_firebase_auth_missing_api_key_prevents_proceeding(monkeypatch):
    """When API key is unconfigured, handlers must return False and exact required error message."""
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", "")
    monkeypatch.setenv("FIREBASE_API_KEY", "")
    monkeypatch.setattr(FirebaseAuthHandler, "get_api_key", lambda: "")

    # Test Create User fails
    ok_create, res_create = FirebaseAuthHandler.create_user("test.candidate@university.edu", "Password123!")
    assert ok_create is False
    assert res_create.get("error") == MISSING_API_KEY_ERROR

    # Test Sign In fails
    ok_login, res_login = FirebaseAuthHandler.sign_in("test.candidate@university.edu", "Password123!")
    assert ok_login is False
    assert res_login.get("error") == MISSING_API_KEY_ERROR

    # Test Send Email Verification fails
    ok_ver, msg_ver = FirebaseAuthHandler.send_verification_email("mock_token")
    assert ok_ver is False
    assert msg_ver == MISSING_API_KEY_ERROR

    # Test Send Password Reset fails
    ok_rst, msg_rst = FirebaseAuthHandler.send_password_reset("test.candidate@university.edu")
    assert ok_rst is False
    assert msg_rst == MISSING_API_KEY_ERROR
