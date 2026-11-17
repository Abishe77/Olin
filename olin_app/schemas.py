from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    email: str = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)


class VerifyEmailRequest(BaseModel):
    email : str = Field(max_length = 255)
    otp : str = Field(min_length = 6 , max_length = 6 , pattern=r"^\d{6}$")

class ResendVerificationRequest(BaseModel):
    email: str = Field(max_length=255)

class LoginRequest(BaseModel):
    email : str = Field(max_length = 255)
    password : str = Field(min_length =1 , max_length = 128)

class ForgotPasswordRequest(BaseModel):
    email: str = Field(max_length=255)


class VerifyResetOTPRequest(BaseModel):
    email: str = Field(max_length=255)
    otp: str = Field(
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
    )



class ResetPasswordRequest(BaseModel):
    email: str = Field(max_length=255)
    otp: str = Field(
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
    )
    new_password: str = Field(
        min_length=8,
        max_length=128,
    )

class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=256)                