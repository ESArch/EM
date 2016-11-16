import psycopg2

class Queries:

    def __init__(self):
        self.conn = ""
        self.query = ""

    def connect(self):
        self.conn = psycopg2.connect("dbname=postgis_22_sample user=postgres password=postgres")
        print("Connected to database")

    def disconnect(self):
        self.conn.close()
        print("Disconnected from database")

    def execute_query(self):
        self.connect()
        cur = self.conn.cursor()
        cur.execute(self.query)
        result = cur.fetchall()
        cur.close()
        self.disconnect()
        return result


    def avg_distance_and_interval_detail(self):
        self.query = "SELECT AVG(twe_min_distancia), MAX(twe_fecha_creacion)-MIN(twe_fecha_creacion)\n"
        self.query += "FROM tweet\n"
        self.query += "GROUP BY twe_usuario HAVING COUNT(*) >= 5 AND MAX(twe_fecha_creacion)-MIN(twe_fecha_creacion) <= 50\n"
        return self.execute_query()

    def avg_distance_and_interval(self):
        self.query = "SELECT AVG(twe_min_distancia), MAX(twe_fecha_creacion)-MIN(twe_fecha_creacion)\n"
        self.query += "FROM tweet\n"
        self.query += "GROUP BY twe_usuario HAVING COUNT(*) >= 5\n"
        return self.execute_query()