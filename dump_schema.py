import psycopg2
from database.database import get_connection

def dump_schema():
    conn = get_connection()
    cur = conn.cursor()
    
    print("\nDatabase Schema Dump")
    print("====================\n")
    
    tables = ['workers', 'violations']
    
    for table in tables:
        print(f"Table: {table}")
        print("-" * (len(table) + 7))
        
        # Header
        print(f"{'Column Name':<20} | {'Type':<20} | {'Nullable':<8} | {'Default'}")
        print("-" * 80)
        
        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        
        rows = cur.fetchall()
        for row in rows:
            col_name = row[0]
            dtype = row[1]
            nullable = row[2]
            default = str(row[3]) if row[3] else "NULL"
            
            # Simplified type mapping for display matching the screenshot somewhat
            if dtype == 'ARRAY':
                dtype = 'ARRAY'
            elif dtype == 'user-defined': # UUID usually
                dtype = 'uuid' 
            
            print(f"{col_name:<20} | {dtype:<20} | {nullable:<8} | {default}")
        
        print("\n")

    cur.close()
    conn.close()

if __name__ == "__main__":
    dump_schema()
