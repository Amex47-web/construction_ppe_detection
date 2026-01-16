from database.database import get_connection

def test_connection():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM workers")
        count = cur.fetchone()[0]
        print(f"Connection Successful! Workers count: {count}")
        conn.close()
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_connection()
