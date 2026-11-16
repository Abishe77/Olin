
import asyncio
import sys
from pathlib import Path

from fastapi import HTTPException

# Add the project root (D:\Olin) to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from olin_app.main import app
from olin_app.rate_limit import enforce_rate_limit


async def test_rate_limit():
    async with app.router.lifespan_context(app):

        print("\nTesting PostgreSQL rate limiter...\n")

        for attempt in range(1, 5):
            try:
                enforce_rate_limit(
                    scope="test_scope",
                    identifier="test_user_file",
                    limit=3,
                    window_seconds=60,
                )

                print(f"Attempt {attempt}: Allowed")

            except HTTPException as error:
                print(
                    f"Attempt {attempt}: "
                    f"HTTP {error.status_code} - {error.detail}"
                )


if __name__ == "__main__":
    asyncio.run(test_rate_limit())

