from olin_app.security import (
    hash_password,
    verify_password,
    validate_password,
)


passwords = [
    "olintest",
    "Olintest1",
    "OlinTest1!",
    "OLIN1234@",
]

for password in passwords:
    errors = validate_password(password)

    print(f"\nPassword: {password}")
    print("Valid:", len(errors) == 0)

    for error in errors:
        print("-", error)


password = "OlinTestPassword123!"
hashed_password = hash_password(password)

print("\nHashing test")
print("Correct password:", verify_password(password, hashed_password))
print("Incorrect password:", verify_password("WrongPassword123!", hashed_password))