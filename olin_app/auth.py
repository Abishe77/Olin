import os
import hmac

from psycopg2 import IntegrityError
from email_validator import validate_email, EmailNotValidError

from datetime import datetime, timedelta, timezone

from olin_app.database import get_db_cursor
from olin_app.email_service import send_otp_email

from olin_app.security import (
    hash_password,
    validate_password,
    verify_password,
    generate_otp,
    hash_otp,
    verify_otp as verify_otp_hash,
    generate_refresh_token,
    hash_refresh_token,
)

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token



# Email validation and normalization


def normalize_email(email: str) -> str:
    try:
        validated_email = validate_email(
            email.strip(),
            check_deliverability=False,
        )

        normalized_email = validated_email.normalized

    except EmailNotValidError:
        raise ValueError("Invalid email address.")

    if len(normalized_email) > 255:
        raise ValueError("Email address must not exceed 255 characters.")

    return normalized_email

def verify_google_id_token(token : str , nonce: str) -> dict:
    client_id = os.getenv("GOOGLE_CLIENT_ID")

    if not client_id:
        raise RuntimeError("Google OAuth is not configured.")

    try:
        claims = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            audience = client_id,
        )
    except ValueError as error:
        raise ValueError("Invalid Google ID token.") from None

    #Require a verified gmail
    if claims.get("email_verified") is not True:
        raise ValueError("Google email address is not verified.")

    google_sub = claims.get("sub")
    email = claims.get("email")
    name = claims.get("name")
    token_nonce = claims.get("nonce") #ensure token belongs to login attempt

    if not token_nonce or not hmac.compare_digest(token_nonce, nonce):
        raise ValueError("Google nonce validation failed.")

    if not google_sub or not email:
        raise ValueError("Google account information is incomplete.")

    return{
        "google_sub" : google_sub,
        "email" : email,
        "name" : name,
    }


def login_or_create_google_user(google_identity: dict) -> dict:
    google_sub = google_identity["google_sub"]
    email = normalize_email(google_identity["email"])
    name = (google_identity.get("name") or "").strip()

    if not name:
        name = email.split("@")[0]

    name = name[:150]

    with get_db_cursor() as cursor:

        # Serialize requests for the same Google identity and email.
        lock_keys = sorted([
            f"google:{google_sub}",
            f"email:{email}",
        ])

        for lock_key in lock_keys:
            cursor.execute(
                """
                SELECT pg_advisory_xact_lock(hashtext(%s)::bigint);
                """,
                (lock_key,),
            )

        # Check whether this Google identity is already linked.
        cursor.execute(
            """
            SELECT u.id, u.name, u.email
            FROM auth_accounts aa
            JOIN users u ON u.id = aa.user_id
            WHERE aa.provider = 'google'
              AND aa.provider_user_id = %s;
            """,
            (google_sub,),
        )

        user = cursor.fetchone()

        if user is not None:
            return {
                "id": user[0],
                "name": user[1],
                "email": user[2],
            }

        # Try creating a new verified Olin account.
        cursor.execute(
            """
            INSERT INTO users (
                name,
                email,
                password_hash,
                email_verified
            )
            VALUES (%s, %s, NULL, TRUE)
            ON CONFLICT (email) DO NOTHING
            RETURNING id, name, email;
            """,
            (name, email),
        )

        user = cursor.fetchone()

        if user is not None:
            # A new account was created successfully.
            user_id, user_name, user_email = user

        else:
            # An Olin account with this email already exists.
            cursor.execute(
                """
                SELECT id, name, email, email_verified
                FROM users
                WHERE email = %s;
                """,
                (email,),
            )

            existing_user = cursor.fetchone()

            if existing_user is None:
                raise RuntimeError("Unable to retrieve Olin account.")

            user_id, user_name, user_email, email_verified = existing_user

            if not email_verified:
                raise ValueError(
                    "An unverified Olin account already uses this email. "
                    "Please verify it before linking Google."
                )

        # Link the verified Google identity to the Olin account.
        cursor.execute(
            """
            INSERT INTO auth_accounts (
                user_id,
                provider,
                provider_user_id
            )
            VALUES (%s, 'google', %s)
            ON CONFLICT (provider, provider_user_id) DO NOTHING
            RETURNING user_id;
            """,
            (user_id, google_sub),
        )

        linked_user = cursor.fetchone()

        if linked_user is None:
            raise ValueError(
                "This Google account is already linked to another Olin account."
            )

        return {
            "id": user_id,
            "name": user_name,
            "email": user_email,
        }
        


# User registration


def register_user(name: str, email: str, password: str):

    name = name.strip()

    if not name:
        raise ValueError("Name cannot be empty.")

    if len(name) > 150:
        raise ValueError("Name must not exceed 150 characters.")

    email = normalize_email(email)

    password_errors = validate_password(password)

    if password_errors:
        raise ValueError("; ".join(password_errors))

    password_hash = hash_password(password)

    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id, name, email, email_verified, created_at;
                """,
                (name, email, password_hash),
            )

            user = cursor.fetchone()

    except IntegrityError as error:
        if error.diag.constraint_name == "users_email_key":
            raise ValueError(
                "An account with this email already exists."
            ) from error

        raise

    user_data = {
        "id": user[0],
        "name": user[1],
        "email": user[2],
        "email_verified": user[3],
        "created_at": user[4],
    }

    # Generate and store verification OTP
    otp = create_otp(
        user_id=user_data["id"],
        purpose="email_verification",
    )

    # Send verification email
    try:
        send_otp_email(
            email=user_data["email"],
            otp=otp,
            purpose="email_verification",
        )

    except Exception:
        invalidate_otp(
            user_id=user_data["id"],
            purpose="email_verification",
        )

        raise RuntimeError(
            "Your account was created, but the verification email "
            "could not be sent. Please try again later."
        ) from None

    return user_data



# OTP generation and database storage


def create_otp(user_id: int, purpose: str):

    if purpose not in ("email_verification", "password_reset"):
        raise ValueError("Invalid OTP purpose.")

    otp = generate_otp()
    otp_hash = hash_otp(otp)

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    with get_db_cursor() as cursor:

        # Invalidate previous active OTPs
        cursor.execute(
            """
            UPDATE otp_codes
            SET consumed_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
              AND purpose = %s
              AND consumed_at IS NULL;
            """,
            (user_id, purpose),
        )

        # Store the new OTP hash
        cursor.execute(
            """
            INSERT INTO otp_codes (
                user_id,
                purpose,
                otp_hash,
                expires_at
            )
            VALUES (%s, %s, %s, %s);
            """,
            (user_id, purpose, otp_hash, expires_at),
        )

    return otp



# OTP invalidation


def invalidate_otp(user_id: int, purpose: str):

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            UPDATE otp_codes
            SET consumed_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
              AND purpose = %s
              AND consumed_at IS NULL;
            """,
            (user_id, purpose),
        )



# OTP verification


def verify_otp_code(user_id: int, purpose: str, otp: str):

    if purpose not in ("email_verification", "password_reset"):
        raise ValueError("Invalid OTP purpose.")

    with get_db_cursor() as cursor:

        cursor.execute(
            """
            SELECT id, otp_hash, attempts
            FROM otp_codes
            WHERE user_id = %s
              AND purpose = %s
              AND consumed_at IS NULL
              AND expires_at > CURRENT_TIMESTAMP
            ORDER BY created_at DESC
            LIMIT 1
            FOR UPDATE;
            """,
            (user_id, purpose),
        )

        record = cursor.fetchone()

        if record is None:
            raise ValueError("OTP is invalid or expired.")

        otp_id, stored_hash, attempts = record

        if attempts >= 5:
            raise ValueError("Maximum OTP attempts exceeded.")

        # Check submitted OTP
        if not verify_otp_hash(otp, stored_hash):

            cursor.execute(
                """
                UPDATE otp_codes
                SET attempts = attempts + 1
                WHERE id = %s;
                """,
                (otp_id,),
            )

            # Return normally so the failed-attempt counter commits
            return False

        # Consume successfully verified OTP
        cursor.execute(
            """
            UPDATE otp_codes
            SET consumed_at = CURRENT_TIMESTAMP
            WHERE id = %s;
            """,
            (otp_id,),
        )

        # Verify user email
        if purpose == "email_verification":

            cursor.execute(
                """
                UPDATE users
                SET email_verified = TRUE
                WHERE id = %s;
                """,
                (user_id,),
            )

        return True



# Resend email verification OTP


def resend_verification_email(email: str):

    email = normalize_email(email)

    with get_db_cursor() as cursor:

        cursor.execute(
            """
            SELECT id, email_verified
            FROM users
            WHERE email = %s;
            """,
            (email,),
        )

        user = cursor.fetchone()

    if not user:
        # Do not reveal whether an account exists
        return

    user_id, email_verified = user

    if email_verified:
        return

    # Generate and store a fresh OTP
    otp = create_otp(
        user_id=user_id,
        purpose="email_verification",
    )

    # Send the newly generated OTP
    try:
        send_otp_email(
            email=email,
            otp=otp,
            purpose="email_verification",
        )

    except Exception:

        invalidate_otp(
            user_id=user_id,
            purpose="email_verification",
        )

        raise RuntimeError(
            "Unable to send verification email. Please try again later."
        ) from None

def login_user(email: str , password: str):
    email = normalize_email(email)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
             SELECT id , name , email , password_hash , email_verified
             FROM users 
             WHERE email = %s;
            """,
            (email,),
        )

        user = cursor.fetchone()

        #Rejecting unknown acc without pass
    if user is None or user[3] is None:
        raise ValueError("Invalid email or password.")

    user_id , name , email , password_hash , email_verified = user

    #Verify pass
    if not verify_password(password, password_hash):
        raise ValueError("Invalid email or password.")

    #Require verification
    if not email_verified:
        raise ValueError("Please verify your email before logging in.")

    return{
        "id" : user_id,
        "name" : name,
        "email" : email,
    }

# Request password reset

def request_password_reset(email: str):

    email = normalize_email(email)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = %s
              AND email_verified = TRUE;
            """,
            (email,),
        )

        user = cursor.fetchone()

    # Do not reveal whether an account exists
    if user is None:
        return

    user_id = user[0]

    # Generate password-reset OTP
    otp = create_otp(
        user_id=user_id,
        purpose="password_reset",
    )

    try:
        send_otp_email(
            email=email,
            otp=otp,
            purpose="password_reset",
        )

    except Exception:
        invalidate_otp(
            user_id=user_id,
            purpose="password_reset",
        )

        raise RuntimeError(
            "Unable to send password reset email. Please try again later."
        ) from None


def verify_password_reset_otp(email: str, otp: str) -> bool:

    email = normalize_email(email)

    with get_db_cursor() as cursor:

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = %s
              AND email_verified = TRUE;
            """,
            (email,),
        )

        user = cursor.fetchone()

        if user is None:
            raise ValueError("Invalid or expired OTP.")

        user_id = user[0]

        cursor.execute(
            """
            SELECT id, otp_hash, attempts
            FROM otp_codes
            WHERE user_id = %s
              AND purpose = 'password_reset'
              AND consumed_at IS NULL
              AND expires_at > CURRENT_TIMESTAMP
            ORDER BY created_at DESC
            LIMIT 1
            FOR UPDATE;
            """,
            (user_id,),
        )

        record = cursor.fetchone()

        if record is None:
            raise ValueError("Invalid or expired OTP.")

        otp_id, stored_hash, attempts = record

        if attempts >= 5:
            raise ValueError("Maximum OTP attempts exceeded.")

        if not verify_otp_hash(otp, stored_hash):

            cursor.execute(
                """
                UPDATE otp_codes
                SET attempts = attempts + 1
                WHERE id = %s;
                """,
                (otp_id,),
            )

            return False

        # OTP is valid.
        # Do not consume it here.
        # The final password-reset operation will consume it.

        return True
    

# Reset password

def reset_password(email: str, otp: str, new_password: str):

    email = normalize_email(email)

    password_errors = validate_password(new_password)

    if password_errors:
        raise ValueError("; ".join(password_errors))

    new_password_hash = hash_password(new_password)

    with get_db_cursor() as cursor:

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = %s
              AND email_verified = TRUE;
            """,
            (email,),
        )

        user = cursor.fetchone()

        if user is None:
            raise ValueError("Invalid or expired OTP.")

        user_id = user[0]

        cursor.execute(
            """
            SELECT id, otp_hash, attempts
            FROM otp_codes
            WHERE user_id = %s
              AND purpose = 'password_reset'
              AND consumed_at IS NULL
              AND expires_at > CURRENT_TIMESTAMP
            ORDER BY created_at DESC
            LIMIT 1
            FOR UPDATE;
            """,
            (user_id,),
        )

        record = cursor.fetchone()

        if record is None:
            raise ValueError("Invalid or expired OTP.")

        otp_id, stored_hash, attempts = record

        if attempts >= 5:
            raise ValueError("Maximum OTP attempts exceeded.")

        if not verify_otp_hash(otp, stored_hash):

            cursor.execute(
                """
                UPDATE otp_codes
                SET attempts = attempts + 1
                WHERE id = %s;
                """,
                (otp_id,),
            )

            return False

        # Consume the OTP
        cursor.execute(
            """
            UPDATE otp_codes
            SET consumed_at = CURRENT_TIMESTAMP
            WHERE id = %s;
            """,
            (otp_id,),
        )

                # Update the password
        cursor.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s;
            """,
            (new_password_hash, user_id),
        )

        # Revoke all existing refresh sessions
        cursor.execute(
            """
            UPDATE refresh_sessions
            SET revoked_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
              AND revoked_at IS NULL;
            """,
            (user_id,),
        )

        return True


def create_refresh_session(user_id : int) -> str:
    refresh_token = generate_refresh_token()
    token_hash = hash_refresh_token(refresh_token)

    expires_at = datetime.now(timezone.utc) + timedelta(days=30)


    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO refresh_sessions(
            user_id,
            token_hash,
            expires_at
            )
            VALUES (%s , %s , %s)
            """,
            (user_id, token_hash ,expires_at)
        )
    return refresh_token



def rotate_refresh_session(refresh_token: str) -> tuple[int, str]:
    token_hash = hash_refresh_token(refresh_token)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, user_id
            FROM refresh_sessions
            WHERE token_hash = %s
            AND revoked_at IS NULL
            AND expires_at > CURRENT_TIMESTAMP
            FOR UPDATE
            """,
            (token_hash,)
        )

        session = cursor.fetchone()

        if session is None:
            raise ValueError("Invalid or expired refresh token.")

        session_id, user_id = session

        # Revoke the old refresh-token session.
        cursor.execute(
            """
            UPDATE refresh_sessions
            SET revoked_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (session_id,)
        )

        # Generate a replacement refresh token.
        new_refresh_token = generate_refresh_token()
        new_token_hash = hash_refresh_token(new_refresh_token)

        new_expires_at = (
            datetime.now(timezone.utc) + timedelta(days=30)
        )

        cursor.execute(
            """
            INSERT INTO refresh_sessions (
                user_id,
                token_hash,
                expires_at
            )
            VALUES (%s, %s, %s)
            """,
            (user_id, new_token_hash, new_expires_at)
        )

    return user_id, new_refresh_token



def revoke_refresh_session(refresh_token: str) -> bool:
    token_hash = hash_refresh_token(refresh_token)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            UPDATE refresh_sessions
            SET revoked_at = CURRENT_TIMESTAMP
            WHERE token_hash = %s
            AND revoked_at IS NULL
            """,
            (token_hash,)
        )

        return cursor.rowcount > 0



   


    

