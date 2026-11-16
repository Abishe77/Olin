import os
import secrets
import hashlib
import base64
import requests
import json
from pathlib import Path

from urllib.parse import urlencode


from contextlib import asynccontextmanager


from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Cookie,
    Response,
    UploadFile,
    File,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse , RedirectResponse , FileResponse
from fastapi.middleware.cors import CORSMiddleware

from olin_app.database import (
    get_db_cursor,
    initialize_connection_pool,
    close_connection_pool,
    check_database_connection,
)

from olin_app.auth import (
    register_user, 
    verify_otp_code,
    resend_verification_email,
    login_user,
    reset_password,
    request_password_reset,
    create_refresh_session,
    rotate_refresh_session,
    revoke_refresh_session,
    verify_google_id_token,
    login_or_create_google_user,
)

from olin_app.schemas import (
    RegisterRequest,
    VerifyEmailRequest,
    ResendVerificationRequest,
    LoginRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    RefreshRequest,
)

from olin_app.security import(
    create_access_token,
    decode_access_token,
)
from olin_app.rate_limit import enforce_rate_limit

#Scieitific related impirts
from starlette.concurrency import run_in_threadpool
from psycopg2.extras import Json
from inference.config import MODEL_VERSION
from inference.pipeline import run_pipeline
from inference.report import generate_pdf
from PIL import Image , UnidentifiedImageError




@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_connection_pool()

    yield

    close_connection_pool()


app = FastAPI(
    title="Olin API",
    description="Gravitational Lensing Inference Application",
    version="1.0.0",
    lifespan=lifespan,
)

#For reports
ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = ROOT / "private_storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True , exist_ok= True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024 # 10mb

ALLOWED_UPLOAD_FORMATS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

frontend_origin = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5500",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

def validate_frontend_origin(request: Request):
    origin = request.headers.get("origin")

    if origin != frontend_origin:
        raise HTTPException(
            status_code=403,
            detail="Invalid request origin.",
        )

bearer_scheme = HTTPBearer(auto_error = False)
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):

    if credentials is None:
        raise HTTPException(
            status_code = 401,
            detail = "Authentication required.",
            headers = {"WWW-Authenticate" : "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])

    except ValueError:
        raise HTTPException(
            status_code = 401,
            detail = "Invalid or expired access token.",
            headers = {"WWW-Authenticate" : "Bearer"},
        )

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id , name , email 
            FROM users 
            WHERE id = %s
            AND email_verified = TRUE;
            """,
            (user_id,),
        ) 
        user = cursor.fetchone()

    if user is None:
        raise HTTPException(
            status_code = 401,
            detail = "User account is unavailable.",
        )

    return {
        "id" : user[0],
        "name" : user[1],
        "email" : user[2],
    }


def clear_google_oauth_cookies(response: JSONResponse, cookie_secure: bool):
    for cookie_name in (
        "olin_google_state",
        "olin_google_nonce",
        "olin_google_verifier",
    ):
        response.delete_cookie(
            key=cookie_name,
            path="/auth/google/callback",
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
        )

async def save_uploaded_image(upload_file : UploadFile , destination: Path):
    suffix = Path(upload_file.filename or "").suffix.lower()

    if suffix not in ALLOWED_UPLOAD_FORMATS:
        raise HTTPException(
            status_code = 400,
            detail = "Unsupported image format."
        )
    total_size = 0
    try:
        with open(destination , "wb") as file:
            while True:
                chunk = await upload_file.read(1024 * 1024)

                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code = 413,
                        detail = "Image exceeds the 10 MB upload limit."
                    ) 
                file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code = 400,
                detail = "Uploaded image is empty."
            )
        format_by_extension = {
            ".png" : "PNG",
            ".jpg" : "JPEG",
            ".jpeg" : "JPEG",
            ".webp" : "WEBP",
        }
        with Image.open(destination) as image:
            actual_format = image.format
            image.verify()

        if actual_format != format_by_extension[suffix]:
            raise HTTPException(
                status_code = 400,
                detail = "File extension does not match its image format."
            )
    except (UnidentifiedImageError , OSError , Image.DecompressionBombError):
        destination.unlink(missing_ok = True)

        raise HTTPException(
            status_code = 400,
            detail = "Uploaded file is not a valid supported image."
        ) 
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()


    return destination 

def mark_analysis_failed(analysis_id, user_id):
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE analyses
                SET status = %s
                WHERE id = %s AND user_id = %s
                AND status != %s;
                """,
                (
                    "failed",
                    analysis_id,
                    user_id,
                    "completed"
                )
            )

    except Exception:
        # Preserve the original processing error.
        print(
            f"WARNING: Could not mark analysis {analysis_id} as failed."
        )                    


    
@app.get("/")
def root():
    return {
        "application": "Olin",
        "message": "Gravitational Lensing Inference API is running",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    if check_database_connection():
        return {
            "status": "healthy",
            "database": "connected",
            "database_name": "olin_db",
        }

    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy",
            "database": "disconnected",
        },
    )


@app.post("/auth/register", status_code=201)
def register(
    request: RegisterRequest,
    http_request: Request,
):
    client_ip = (
        http_request.client.host
        if http_request.client
        else "unknown"
    )

    enforce_rate_limit(
        scope="register_ip",
        identifier=client_ip,
        limit=5,
        window_seconds=3600,
    )

    try:
        user = register_user(
            name=request.name,
            email=request.email,
            password=request.password,
        )

        return {
            "message": "Registration Successful, Please verify your email.",
            "user": user,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )


@app.post("/auth/verify-email")
def verify_email(request: VerifyEmailRequest):
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT id, email_verified FROM users WHERE email = %s;",
                (request.email.strip().lower(),)
            )
            user = cursor.fetchone()

        if not user:
            raise HTTPException(status_code=400, detail="Invalid email or OTP.")

        user_id, email_verified = user

        if email_verified:
            return {"message": "Email is already verified."}

        verified = verify_otp_code(
            user_id=user_id,
            purpose="email_verification",
            otp=request.otp
        )

        if not verified:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

        return {"message": "Email verified successfully."}

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) 


@app.post("/auth/resend-verification")
def resend_verification(request: ResendVerificationRequest):
    normalized_email = request.email.strip().lower()

    enforce_rate_limit(
        scope="resend_verification_60s",
        identifier=normalized_email,
        limit=1,
        window_seconds=60,
    )

    enforce_rate_limit(
        scope="resend_verification_hour",
        identifier=normalized_email,
        limit=5,
        window_seconds=3600,
    )

    try:
        resend_verification_email(request.email)

        return {
            "message": "If the account exists and requires verification, "
                       "a verification email has been sent."
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Unable to send verification email."
        )




@app.post("/auth/login")
def login(
    request: LoginRequest,
    response: Response,
    http_request : Request,
):
    client_ip = (
        http_request.client.host
        if http_request.client
        else "unknown"
    )
    normalized_email = request.email.strip().lower()

    # Limit by both IP address and email.
    enforce_rate_limit(
        scope="login_ip",
        identifier=client_ip,
        limit=5,
        window_seconds=900,
    )

    enforce_rate_limit(
        scope="login_email",
        identifier=normalized_email,
        limit=5,
        window_seconds=900,
    )

    try:
        user = login_user(
            email=request.email,
            password=request.password,
        )

        access_token = create_access_token(user["id"])
        refresh_token = create_refresh_session(user["id"])

        cookie_secure = (
            os.getenv("COOKIE_SECURE", "true").lower() == "true"
        )

        response.set_cookie(
            key="olin_refresh_token",
            value=refresh_token,
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
            max_age=30 * 24 * 60 * 60,
            path="/auth",
        )

        return {
            "message": "Login successful.",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 1800,
            "user": {
                "name": user["name"],
                "email": user["email"],
            },
        }

    except ValueError as error:
        raise HTTPException(
            status_code=401,
            detail=str(error),
        )


@app.post(
    "/auth/refresh",
    dependencies=[Depends(validate_frontend_origin)],
)
def refresh(
    response: Response,
    refresh_token: str | None = Cookie(
        default=None,
        alias="olin_refresh_token",
    ),
):
    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="Refresh token is missing.",
        )

    try:
        user_id, new_refresh_token = rotate_refresh_session(
            refresh_token
        )

        new_access_token = create_access_token(user_id)

        cookie_secure = (
            os.getenv("COOKIE_SECURE", "true").lower() == "true"
        )

        response.set_cookie(
            key="olin_refresh_token",
            value=new_refresh_token,
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
            max_age=30 * 24 * 60 * 60,
            path="/auth",
        )

        return {
            "message": "Tokens refreshed successfully.",
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": 1800,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=401,
            detail=str(error),
        )



@app.post(
    "/auth/logout",
    dependencies=[Depends(validate_frontend_origin)],
)
def logout(
    response: Response,
    refresh_token: str | None = Cookie(
        default=None,
        alias="olin_refresh_token",
    ),
):
    if refresh_token:
        revoke_refresh_session(refresh_token)

    cookie_secure = (
        os.getenv("COOKIE_SECURE", "true").lower() == "true"
    )

    response.delete_cookie(
        key="olin_refresh_token",
        path="/auth",
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
    )

    return {
        "message": "Logout successful."
    }



@app.post("/auth/forgot-password")
def forgot_password(request: ForgotPasswordRequest):
    normalized_email = request.email.strip().lower()

    enforce_rate_limit(
        scope="forgot_password_60s",
        identifier=normalized_email,
        limit=1,
        window_seconds=60,
    )

    enforce_rate_limit(
        scope="forgot_password_hour",
        identifier=normalized_email,
        limit=5,
        window_seconds=3600,
    )

    try:
        request_password_reset(request.email)

        return {
            "message": "If the account exists and is verified, "
                       "a password reset email has been sent."
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Unable to send password reset email.",
        )


@app.post("/auth/reset-password")
def reset_password_endpoint(request : ResetPasswordRequest):
    try:
        success = reset_password(
            email= request.email,
            otp = request.otp,
            new_password = request.new_password,
        )

        if not success:
            raise HTTPException(
                status_code = 400,
                detail = "Invalid or expired OTP.",
            )
        return{
            "message" : "Password reset successfully."
        } 
    except ValueError as error:
        raise HTTPException(
            status_code = 400,
            detail = str(error),
        )

@app.get("/auth/me")
def get_me(
    current_user : dict = Depends(get_current_user),):

    return{
        "message" : "Authentication successful.",
        "user" : current_user,
    }



@app.get("/auth/google/login")
def google_login(request: Request):

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    enforce_rate_limit(
        scope="google_login_ip",
        identifier=client_ip,
        limit=10,
        window_seconds=900,
    )

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

    if not client_id or not redirect_uri:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured.",
        )

    cookie_secure = (
        os.getenv("COOKIE_SECURE", "true").lower() == "true"
    )

    # Generate values for this login attempt.
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)

    # Generate PKCE verifier and challenge.
    code_verifier = secrets.token_urlsafe(64)

    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(
            code_verifier.encode("ascii")
        ).digest()
    ).rstrip(b"=").decode("ascii")

    # Construct Google's authorization URL.
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }

    authorization_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urlencode(params)
    )

    response = RedirectResponse(
        url=authorization_url,
        status_code=302,
    )

    # Store temporary OAuth values in HttpOnly cookies.
    cookie_settings = {
        "httponly": True,
        "secure": cookie_secure,
        "samesite": "lax",
        "max_age": 600,
        "path": "/auth/google/callback",
    }

    response.set_cookie(
        key="olin_google_state",
        value=state,
        **cookie_settings,
    )

    response.set_cookie(
        key="olin_google_nonce",
        value=nonce,
        **cookie_settings,
    )

    response.set_cookie(
        key="olin_google_verifier",
        value=code_verifier,
        **cookie_settings,
    )

    return response




@app.get("/auth/google/callback")
def google_callback(request: Request):

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

    cookie_secure = (
        os.getenv("COOKIE_SECURE", "true").lower() == "true"
    )

    def error_response(message: str, status_code: int):
        response = JSONResponse(
            status_code=status_code,
            content={"detail": message},
        )
        clear_google_oauth_cookies(response, cookie_secure)
        return response

    if not client_id or not client_secret or not redirect_uri:
        return error_response(
            "Google OAuth is not configured.",
            500,
        )

    # Retrieve temporary values from the login-start request.
    stored_state = request.cookies.get("olin_google_state")
    nonce = request.cookies.get("olin_google_nonce")
    code_verifier = request.cookies.get("olin_google_verifier")

    returned_state = request.query_params.get("state")
    authorization_code = request.query_params.get("code")

    # Handle user cancellation or Google authorization errors.
    if request.query_params.get("error"):
        return error_response(
            "Google sign-in was cancelled or denied.",
            400,
        )

    # Validate OAuth state.
    if (
        not stored_state
        or not returned_state
        or not secrets.compare_digest(stored_state, returned_state)
    ):
        return error_response(
             "OAuth state validation failed.",
            400,
        )

    if not nonce or not code_verifier or not authorization_code:
        return error_response(
            "Google login information is incomplete.",
            400,
        )

    # Exchange authorization code for Google tokens.
    try:
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": authorization_code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": code_verifier,
            },
            timeout=10,
        )

        token_response.raise_for_status()
        token_data = token_response.json()

    except (requests.RequestException, ValueError):
        return error_response(
            "Unable to complete Google authentication.",
            502,
        )

    google_id_token = token_data.get("id_token")

    if not google_id_token:
        return error_response(
            "Google did not return an identity token.",
            401,
        )

    # Verify Google's identity token and nonce.
    try:
        google_identity = verify_google_id_token(
            google_id_token,
            nonce,
        )

    except ValueError:
        return error_response(
            "Google identity verification failed.",
            401,
        )

    # Find or create the corresponding Olin account.
    try:
        user = login_or_create_google_user(google_identity)

    except ValueError as error:
        return error_response(str(error), 409)

    # Issue Olin's existing authentication tokens.
    access_token = create_access_token(user["id"])
    refresh_token = create_refresh_session(user["id"])

    response = JSONResponse(
        content={
            "message": "Google login successful.",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 1800,
            "user": {
                "name": user["name"],
                "email": user["email"],
            },
        }
    )

    # Keep the refresh token inaccessible to JavaScript.
    response.set_cookie(
        key="olin_refresh_token",
        value=refresh_token,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=30 * 24 * 60 * 60,
        path="/auth",
    )

    clear_google_oauth_cookies(response, cookie_secure)

    return response



@app.post("/analyses", status_code=201)
async def create_analysis(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    #adding rate limits to this
    enforce_rate_limit(
        scope = "analysis_creation",
        identifier= str(current_user["id"]),
        limit = 5,
        window_seconds=3600
    )
    # Create the initial analysis record
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO analyses (
                user_id,
                model_version,
                status,
                reproducibility_metadata
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (
                current_user["id"],
                MODEL_VERSION,
                "pending",
                Json({"model_version": MODEL_VERSION})
            )
        )

        analysis_id = cursor.fetchone()[0]

    try:
        # Create a private directory for this analysis
        analysis_dir = UPLOAD_DIR / str(analysis_id)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "").suffix.lower()
        image_path = analysis_dir / f"original{suffix}"

        # Validate and save the uploaded image
        await save_uploaded_image(file, image_path)

        # Mark the analysis as processing
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE analyses
                SET status = %s
                WHERE id = %s AND user_id = %s;
                """,
                (
                    "processing",
                    analysis_id,
                    current_user["id"]
                )
            )

        # Execute the existing scientific pipeline
        report = await run_in_threadpool(
            run_pipeline,
            image_path,
            analysis_id,
            False
        )

        #Verify that scientific pipeline generated its op's
        reconstruction_path = Path(report["outputs"]["reconstructed_image"])
        residual_path = Path(report["outputs"]["residual_image"])

        if not reconstruction_path.is_file() or not residual_path.is_file():
            raise RuntimeError("Analysis output files were not generated.")

        # Separate predictions from uncertainty estimates
        predictions = {
            name: values["prediction"]
            for name, values in report["lens_parameters"].items()
        }

        uncertainty = {
            name: {
                "raw": values["raw_uncertainty"],
                "calibrated": values["calibrated_uncertainty"]
            }
            for name, values in report["lens_parameters"].items()
        }

        # Store reproducibility information
        metadata = {
            "model_version": MODEL_VERSION,
            "model": report["model"],
            "mc_passes": report["mc_passes"],
            "input_type": report["input_type"],
            "evaluation_type": report["evaluation_type"],
            "image_size": [160, 160],
            "image_channels": 1,
            "input_image_extension": suffix
        }

        # Save the completed analysis
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE analyses
                SET
                    status = %s,
                    predictions = %s,
                    uncertainty = %s,
                    reproducibility_metadata = %s,
                    reconstruction_path = %s,
                    residual_path = %s
                WHERE id = %s AND user_id = %s;
                """,
                (
                    "completed",
                    Json(predictions),
                    Json(uncertainty),
                    Json(metadata),
                    str(reconstruction_path),
                    str(residual_path),
                    analysis_id,
                    current_user["id"]
                )
            )

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "model_version": MODEL_VERSION,
            "predictions": predictions,
            "uncertainty": uncertainty,
            "evaluation_type": report["evaluation_type"]
        }

    except HTTPException:
        mark_analysis_failed(analysis_id, current_user["id"])
        raise

    except Exception:
        mark_analysis_failed(analysis_id, current_user["id"])
        raise HTTPException(
            status_code=500,
            detail="Analysis processing failed."
        )


@app.get("/analyses")
async def list_analyses(
    limit : int = 20,
    offset : int = 0,
    current_user : dict = Depends(get_current_user)
):

    #Validate pagination params
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code = 400,
            detail = "Limit must be between 1 and 100."
        )
    if offset < 0:
        raise HTTPException(
            status_code = 400,
            detail = "Offset cannot be negative."
        )
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT 
            id,
            model_version,
            status,
            created_at
            FROM analyses
            WHERE user_id = %s
            ORDER BY created_at DESC , id DESC
            LIMIT %s OFFSET %s
            """,
            (
                current_user["id"],
                limit,
                offset
            )
        )
        rows = cursor.fetchall()

    analyses = [
        {
            "analysis_id" : row[0],
            "model_version" : row[1],
            "status" : row[2],
            "created_at" : row[3].isoformat()
        }
        for row in rows
    ]

    return{
        "analysis": analyses,
        "limit" : limit,
        "offset" : offset,
        "count" : len(analyses)
    }



@app.get("/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    current_user: dict = Depends(get_current_user)
):
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                model_version,
                status,
                predictions,
                uncertainty,
                reproducibility_metadata,
                reconstruction_path,
                residual_path,
                created_at
            FROM analyses
            WHERE id = %s AND user_id = %s;
            """,
            (
                analysis_id,
                current_user["id"]
            )
        )

        row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    return {
        "analysis_id": row[0],
        "model_version": row[1],
        "status": row[2],
        "predictions": row[3],
        "uncertainty": row[4],
        "reproducibility_metadata": row[5],
        "outputs": {
            "reconstruction_available": (
                row[6] is not None and Path(row[6]).is_file()
            ),
            "residual_available": (
                row[7] is not None and Path(row[7]).is_file()
            )
        },
        "created_at": row[8].isoformat()
    }



@app.post("/analyses/{analysis_id}/report", status_code=201)
async def generate_analysis_report(
    analysis_id: int,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]

    try:
        with get_db_cursor() as cursor:

            # Lock report generation for this analysis and user
            lock_id = (analysis_id * 1000003 + user_id) % (2**63)
            cursor.execute(
                """
                SELECT pg_advisory_xact_lock(%s);
                """,
                (lock_id,)
            )

            # Verify analysis ownership and completion
            cursor.execute(
                """
                SELECT status, reproducibility_metadata
                FROM analyses
                WHERE id = %s AND user_id = %s;
                """,
                (analysis_id, user_id)
            )

            row = cursor.fetchone()

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail="Analysis not found."
                )

            status, metadata = row

            if status != "completed":
                raise HTTPException(
                    status_code=409,
                    detail="A report can only be generated for a completed analysis."
                )

            # Check for an existing report
            cursor.execute(
                """
                SELECT id, file_path
                FROM reports
                WHERE analysis_id = %s AND user_id = %s;
                """,
                (analysis_id, user_id)
            )

            existing_report = cursor.fetchone()

            # Reuse an existing report if its PDF is available
            if existing_report is not None:
                existing_report_id, existing_file_path = existing_report

                existing_pdf = Path(existing_file_path)

                if existing_pdf.is_file() and existing_pdf.stat().st_size > 0:
                    return {
                        "report_id": existing_report_id,
                        "analysis_id": analysis_id,
                        "status": "generated",
                        "file_name": "report.pdf"
                    }

            # Locate the original uploaded image
            suffix = metadata.get("input_image_extension")

            if suffix not in ALLOWED_UPLOAD_FORMATS:
                raise HTTPException(
                    status_code=500,
                    detail="Original image metadata is unavailable."
                )

            image_path = (
                UPLOAD_DIR / str(analysis_id) / f"original{suffix}"
            )

            result_path = (
                ROOT / "inference" / "results" /
                str(analysis_id) / "result.json"
            )

            if not image_path.is_file() or not result_path.is_file():
                raise HTTPException(
                    status_code=500,
                    detail="Required analysis files are unavailable."
                )

            # Load scientific report data
            try:
                with open(result_path, "r", encoding="utf-8") as file:
                    report_data = json.load(file)
            except (OSError, json.JSONDecodeError):
                raise HTTPException(
                    status_code=500,
                    detail="Could not load analysis report data."
                )

            # Generate PDF
            report_dir = (
                ROOT / "private_storage" / "reports" /
                str(analysis_id)
            )
            report_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = report_dir / "report.pdf"

            await run_in_threadpool(
                generate_pdf,
                image_path,
                report_data,
                pdf_path
            )

            if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
                raise RuntimeError("PDF generation failed.")

            # Update an existing record or insert a new one
            if existing_report is not None:
                cursor.execute(
                    """
                    UPDATE reports
                    SET file_path = %s,
                        file_name = %s
                    WHERE id = %s AND user_id = %s;
                    """,
                    (
                        str(pdf_path),
                        "report.pdf",
                        existing_report_id,
                        user_id
                    )
                )

                report_id = existing_report_id

            else:
                cursor.execute(
                    """
                    INSERT INTO reports (
                        analysis_id,
                        user_id,
                        file_path,
                        file_name
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        analysis_id,
                        user_id,
                        str(pdf_path),
                        "report.pdf"
                    )
                )

                report_id = cursor.fetchone()[0]

        return {
            "report_id": report_id,
            "analysis_id": analysis_id,
            "status": "generated",
            "file_name": "report.pdf"
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Report generation failed."
        )


@app.get("/reports")
async def list_reports(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100."
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset cannot be negative."
        )

    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    analysis_id,
                    file_name,
                    created_at
                FROM reports
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s;
                """,
                (user_id, limit, offset)
            )

            rows = cursor.fetchall()

        reports = [
            {
                "report_id": row[0],
                "analysis_id": row[1],
                "file_name": row[2],
                "created_at": row[3]
            }
            for row in rows
        ]

        return {
            "reports": reports,
            "limit": limit,
            "offset": offset
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve reports."
        )


@app.get("/reports/{report_id}/download")
async def download_report(
    report_id: int,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]

    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT file_path, file_name
                FROM reports
                WHERE id = %s AND user_id = %s;
                """,
                (report_id, user_id)
            )

            report = cursor.fetchone()

        if report is None:
            raise HTTPException(
                status_code=404,
                detail="Report not found."
            )

        file_path, file_name = report
        pdf_path = Path(file_path)

        if not pdf_path.is_file():
            raise HTTPException(
                status_code=410,
                detail="The report file is no longer available."
            )

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=file_name
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not download report."
        )
    










    
  
