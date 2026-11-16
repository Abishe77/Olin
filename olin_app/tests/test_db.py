from database import get_db_connection

try:
    connection = get_db_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT current_database(), current_user;")

    result = cursor.fetchone()

    print("DB connection successful")
    print("DB: " , result[0])
    print("User: ", result[1])

except Exception as error:
    print("DB connection failed")    
    print(error)