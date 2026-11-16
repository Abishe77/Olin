
import os
from pathlib import Path
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_connection_pool = None


def initialize_connection_pool():
    global _connection_pool

    if _connection_pool is not None:
        return

    required_variables = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:
        raise RuntimeError(
            f"Missing database configuration: {', '.join(missing_variables)}"
        )

    _connection_pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def get_db_connection():
    if _connection_pool is None:
        raise RuntimeError("Database connection pool is not initialized")

    return _connection_pool.getconn()


def release_db_connection(connection):
    if connection is None:
        return

    try:
        connection.rollback()

    except psycopg2.Error:
        _connection_pool.putconn(connection,close = True) #discards a connection in a pool, if smth breaks
        return
    _connection_pool.putconn(connection)   
    


@contextmanager
def get_db_cursor():
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        yield cursor

        connection.commit()

    except Exception:
        if connection is not None:
            connection.rollback()

        raise

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            release_db_connection(connection)
                  


def close_connection_pool():
    global _connection_pool

    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


def check_database_connection():
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT 1;")
        result = cursor.fetchone()

        return result[0] == 1

    except psycopg2.Error:
        return False

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            release_db_connection(connection)
   
