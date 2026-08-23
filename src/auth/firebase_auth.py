"""
Firebase Authentication Handler.
Uses official Firebase Identity Toolkit REST API for Email/Password authentication,
account creation, email verification links, and password resets.
Does not require private service keys in repository — uses FIREBASE_WEB_API_KEY from environment or .env file.
"""

import os
from pathlib import Path
import requests
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger("firebase_auth")

# Explicitly locate and load .env file from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

FIREBASE_AUTH_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"
MISSING_API_KEY_ERROR = "Firebase Authentication is not configured. Please configure FIREBASE_WEB_API_KEY."


class FirebaseAuthHandler:
    """
    Handles Firebase Email/Password Authentication & Email Verification
    via Firebase Identity Toolkit REST API.
    """

    @staticmethod
    def get_api_key() -> str:
        """Retrieves Firebase Web API Key from environment variables or .env file."""
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=False)
        else:
            load_dotenv(override=False)

        api_key = os.getenv("FIREBASE_WEB_API_KEY", os.getenv("FIREBASE_API_KEY", "") or "")
        return api_key.strip()

    @classmethod
    def create_user(cls, email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Creates a new candidate account using Firebase Authentication.
        """
        api_key = cls.get_api_key()
        if not api_key:
            logger.warning("FIREBASE_WEB_API_KEY is missing. Failing account creation.")
            return False, {"error": MISSING_API_KEY_ERROR}

        url = f"{FIREBASE_AUTH_BASE}:signUp?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            res = requests.post(url, json=payload, timeout=10)
            data = res.json()
            if res.status_code == 200:
                logger.info(f"Firebase account created for '{email}' (UID: {data.get('localId')})")
                return True, data
            else:
                err_code = data.get("error", {}).get("message", "Registration failed.")
                logger.warning(f"Firebase signUp error for '{email}': {err_code}")
                return False, {"error": cls._format_error(err_code)}
        except Exception as exc:
            logger.error(f"Firebase signUp exception: {exc}")
            return False, {"error": f"Connection error: {exc}"}

    @classmethod
    def sign_in(cls, email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Signs in an existing candidate using Firebase Authentication and checks emailVerified status.
        """
        api_key = cls.get_api_key()
        if not api_key:
            logger.warning("FIREBASE_WEB_API_KEY is missing. Failing sign in.")
            return False, {"error": MISSING_API_KEY_ERROR}

        url = f"{FIREBASE_AUTH_BASE}:signInWithPassword?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            res = requests.post(url, json=payload, timeout=10)
            data = res.json()
            if res.status_code == 200:
                # Fetch fresh lookup information to get up-to-date emailVerified status
                ok_lookup, lookup_info = cls.get_user_info(data.get("idToken", ""))
                if ok_lookup:
                    data["emailVerified"] = lookup_info.get("emailVerified", False)
                else:
                    data["emailVerified"] = False

                logger.info(f"Firebase user '{email}' signed in (emailVerified: {data.get('emailVerified')})")
                return True, data
            else:
                err_code = data.get("error", {}).get("message", "Sign in failed.")
                logger.warning(f"Firebase signIn error for '{email}': {err_code}")
                return False, {"error": cls._format_error(err_code)}
        except Exception as exc:
            logger.error(f"Firebase signIn exception: {exc}")
            return False, {"error": f"Connection error: {exc}"}

    @classmethod
    def send_verification_email(cls, id_token: str) -> Tuple[bool, str]:
        """
        Sends a Firebase Email Verification link to the user's email address.
        """
        api_key = cls.get_api_key()
        if not api_key:
            return False, MISSING_API_KEY_ERROR

        url = f"{FIREBASE_AUTH_BASE}:sendOobCode?key={api_key}"
        payload = {"requestType": "VERIFY_EMAIL", "idToken": id_token}
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                logger.info("Sent Firebase verification email.")
                return True, "Verification email sent successfully."
            else:
                err = res.json().get("error", {}).get("message", "Failed to send email.")
                return False, cls._format_error(err)
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def send_password_reset(cls, email: str) -> Tuple[bool, str]:
        """
        Sends a Firebase Password Reset email to the candidate's email address.
        """
        api_key = cls.get_api_key()
        if not api_key:
            return False, MISSING_API_KEY_ERROR

        url = f"{FIREBASE_AUTH_BASE}:sendOobCode?key={api_key}"
        payload = {"requestType": "PASSWORD_RESET", "email": email}
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                logger.info(f"Sent password reset email to '{email}'")
                return True, f"Password reset email sent to '{email}'."
            else:
                err = res.json().get("error", {}).get("message", "Failed to send reset email.")
                return False, cls._format_error(err)
        except Exception as exc:
            return False, str(exc)

    @classmethod
    def get_user_info(cls, id_token: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieves user account info (including emailVerified) from Firebase using idToken.
        """
        api_key = cls.get_api_key()
        if not api_key:
            return False, {}

        url = f"{FIREBASE_AUTH_BASE}:lookup?key={api_key}"
        payload = {"idToken": id_token}
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                users = res.json().get("users", [])
                if users:
                    return True, users[0]
            return False, {}
        except Exception:
            return False, {}

    @staticmethod
    def _format_error(msg: str) -> str:
        """Maps Firebase error string codes to user-friendly messages."""
        mapping = {
            "EMAIL_EXISTS": "An account with this email address already exists.",
            "INVALID_EMAIL": "Please enter a valid email address.",
            "WEAK_PASSWORD": "Password should be at least 6 characters.",
            "EMAIL_NOT_FOUND": "No candidate account found with this email address.",
            "INVALID_PASSWORD": "Incorrect password. Please check your credentials.",
            "USER_DISABLED": "This candidate account has been disabled by the administrator.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many failed attempts. Please try again later."
        }
        return mapping.get(msg, msg)
