
import os
import hmac
import hashlib

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from olin_app.database import get_db_cursor


def create_rate_limit_key(scope: str, identifier: str) -> str:
    secret = os.getenv("RATE_LIMIT_SECRET")

    if not secret:
        raise RuntimeError("Rate limiting is not configured.")

    raw_key = f"{scope}:{identifier}".encode("utf-8")

    return hmac.new(
        secret.encode("utf-8"),
        raw_key,
        hashlib.sha256,
    ).hexdigest()


def enforce_rate_limit(
    scope: str,
    identifier: str,
    limit: int,
    window_seconds: int,
):
    if limit < 1 or window_seconds < 1:
        raise ValueError("Invalid rate limit configuration.")

    rate_key = create_rate_limit_key(scope, identifier)

    now = datetime.now(timezone.utc)

    # Calculate the beginning of the current fixed window.
    timestamp = int(now.timestamp())
    window_timestamp = (
        timestamp // window_seconds
    ) * window_seconds

    window_start = datetime.fromtimestamp(
        window_timestamp,
        tz=timezone.utc,
    )

    window_end = window_start + timedelta(
        seconds=window_seconds
    )

    with get_db_cursor() as cursor:

        # Remove expired rate-limit records in batches.
        cursor.execute(
            """
            DELETE FROM rate_limits
            WHERE id IN (
                SELECT id
                FROM rate_limits
                WHERE window_start < %s
                ORDER BY window_start
                LIMIT 1000
            );
            """,
            (now - timedelta(days=2),),
        )
        cursor.execute(
            """
            INSERT INTO rate_limits (
                rate_key,
                window_start,
                attempts
            )
            VALUES (%s, %s, 1)

            ON CONFLICT (rate_key, window_start)
            DO UPDATE SET
                attempts = LEAST(
                    rate_limits.attempts + 1,
                    %s
                )

            RETURNING attempts;
            """,
            (rate_key, window_start, limit + 1),
        )

        attempts = cursor.fetchone()[0]

    if attempts > limit:
        retry_after = max(
            1,
            int((window_end - now).total_seconds()) + 1,
        )

        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
            headers={
                "Retry-After": str(retry_after),
            },
        )
