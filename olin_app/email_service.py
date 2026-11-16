
import os
import resend

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RESEND_API_KEY")
sender_email = os.getenv("RESEND_FROM_EMAIL")

if not api_key:
    raise RuntimeError("RESEND_API_KEY is not configured.")

if not sender_email:
    raise RuntimeError("RESEND_FROM_EMAIL is not configured.")

resend.api_key = api_key


def send_otp_email(email: str, otp: str, purpose: str):
    if purpose == "email_verification":
        subject = "Verify your Olin account"
        message = (
            f"Your Olin verification code is: {otp}\n\n"
            "This code expires in 10 minutes. "
            "If you did not request this, ignore this email."
        )

    elif purpose == "password_reset":
        subject = "Reset your Olin password"
        message = (
            f"Your Olin password reset code is: {otp}\n\n"
            "This code expires in 10 minutes. "
            "If you did not request this, ignore this email."
        )

    else:
        raise ValueError("Invalid email purpose.")

    try:
        result = resend.Emails.send({
            "from": f"Olin <{sender_email}>",
            "to": [email],
            "subject": subject,
            "text": message,
        })

        return result["id"]

    except Exception as error:
        raise RuntimeError(
            "Unable to send email through Resend."
        ) from error




        