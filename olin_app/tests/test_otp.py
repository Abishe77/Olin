from olin_app.database import (
    initialize_connection_pool,
    close_connection_pool,
)

from olin_app.auth import create_otp


initialize_connection_pool()

try:
    otp = create_otp(
        user_id=2,
        purpose="email_verification",
    )

    print("OTP created successfully")
    print("OTP:", otp)
    print("OTP length:", len(otp))

except Exception as error:
    print("OTP creation failed:", error)

finally:
    close_connection_pool()