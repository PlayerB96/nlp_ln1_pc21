import mysql.connector
from config import Config

class Database:
    @staticmethod
    def get_connection():
        return mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )
