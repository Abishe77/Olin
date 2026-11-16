
import os
import hmac
import hashlib
import secrets
import jwt

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from argon2 import PasswordHasher

from argon2.exceptions import (
    VerifyMismatchError , 
    VerificationError,
    )

password_hasher = PasswordHasher() #argon2 is hashing library


def validate_password(password: str) -> list[str]:
    errors = []

    if len(password) < 8:
        errors.append("Password must contain at least 8 characters.")

    if len(password) > 128:
        errors.append("Password must not exceed 128 characters.")    

    if not any(char.isupper() for char in password):
        errors.append("Password must contain at least one uppercase letter.")

    if not any(char.isdigit() for char in password):
        errors.append("Password must contain at least one digit.")

    if not any(not char.isalnum() for char in password):
        errors.append("Password must contain at least one special character.")

    return errors


def hash_password(password : str) -> str:
    return password_hasher.hash(password)

def verify_password(password : str , password_hash : str) -> bool:
    try:
        return password_hasher.verify(password_hash,password)

    except (VerifyMismatchError , VerificationError):
        return False

def needs_rehash(password_hash : str) -> bool:
    return password_hasher.check_needs_rehash(password_hash)


def generate_otp() -> str:
    return f"{secrets.randbelow(1000000):06d}"

def hash_otp(otp : str) -> str:
    otp_secret = os.getenv("OTP_SECRET")

    if not otp_secret:
        raise RuntimeError("OTP _SECRET is not configured.")

    return hmac.new(
        otp_secret.encode(),
        otp.encode(),
        hashlib.sha256,
    ).hexdigest()

def verify_otp(otp: str , otp_hash : str) -> bool:
    calculated_hash = hash_otp(otp)

    return hmac.compare_digest(calculated_hash,otp_hash)

def create_access_token(user_id : int) ->str:
    jwt_secret = os.getenv("JWT_SECRET")

    if not jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured.")

    token_minutes = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES" , "30"))
    now = datetime.now(timezone.utc)

    payload = {
        "sub" : str(user_id),
        "iat" : now,
        "exp" : now + timedelta(minutes=token_minutes),
        "iss" : "olin-api",
        "aud" : "olin-client",
        "type" : "access"
    }

    return jwt.encode(
        payload,
        jwt_secret,
        algorithm = "HS256",
    )

def decode_access_token(token : str) -> dict:
    jwt_secret = os.getenv("JWT_SECRET")

    if not jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured.")

    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms = ["HS256"],
            issuer = "olin-api",
            audience = "olin-client",
        )

        if payload.get("type") != "access":
            raise ValueError("Invalid token type.")

        user_id = int(payload["sub"])

        if user_id <=0:
            raise ValueError("Invalid token subject.")

        return payload
    except (jwt.InvalidTokenError, KeyError , TypeError , ValueError):
        raise ValueError("Invalid or expired access token.") from None

def generate_refresh_token() -> str:
    return secrets.token_urlsafe(64)

def hash_refresh_token(token : str) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest() 
    
