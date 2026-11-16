from olin_app.database import (
    initialize_connection_pool,
    close_connection_pool,
)

from olin_app.auth import verify_otp_code


initialize_connection_pool()

try:
    otp = input("Enter your OTP: ").strip()

    result = verify_otp_code(
        user_id=2,
        purpose="email_verification",
        otp=otp,
    )

    if result:
        print("OTP verification successful")
    else:
        print("Incorrect OTP")

except ValueError as error:
    print("Verification failed:", error)

finally:
    close_connection_pool()