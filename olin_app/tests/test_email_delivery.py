
from olin_app.email_service import send_otp_email

try:
    recipient = input("Enter recipient email: ").strip()

    message_id = send_otp_email(
        email=recipient,
        otp="123456",
        purpose="email_verification",
    )

    print("Resend accepted the email request.")
    print("Message ID:", message_id)

except Exception as error:
    print("Email delivery test failed:", error)
