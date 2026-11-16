
from olin_app.database import (
    initialize_connection_pool,
    close_connection_pool,
)
from olin_app.auth import register_user


initialize_connection_pool()

try:
    user = register_user(
        name="Olin Test User",
        email="olin.test@example.com",
        password="OlinTestPassword123!",
    )

    print("Registration successful")
    print("User ID:", user["id"])
    print("Name:", user["name"])
    print("Email:", user["email"])
    print("Email verified:", user["email_verified"])

except Exception as error:
    print("Registration failed:", error)

finally:
    close_connection_pool()
