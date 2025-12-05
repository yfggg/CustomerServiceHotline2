import os
from typing import Any, Dict, Iterable, List, Optional
from mysql.connector import pooling


class DBPool:
    """简单的 MySQL 连接池包装。"""

    def __init__(self) -> None:
        self._pool: Optional[pooling.MySQLConnectionPool] = None

    @staticmethod
    def _build_db_config() -> Dict[str, Any]:
        return {
            "host": "localhost",
            "port": "3306",
            "user": "root",
            "password": "root",
            "database": "lgpt",
        }

    def _get_pool(self) -> pooling.MySQLConnectionPool:
        if self._pool is None:
            pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
            self._pool = pooling.MySQLConnectionPool(
                pool_name="lgpt_pool",
                pool_size=pool_size,
                **self._build_db_config(),
            )
        return self._pool

    def get_connection(self):
        return self._get_pool().get_connection()

    def fetch_all(self, sql: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute(sql, params or ())
                return cur.fetchall()
        finally:
            conn.close()

    def fetch_one(self, sql: str, params: Optional[Iterable[Any]] = None) -> Optional[Dict[str, Any]]:
        rows = self.fetch_all(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> int:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                conn.commit()
                return cur.rowcount
        finally:
            conn.close()


# 单例实例，方便直接导入使用
db_pool = DBPool()


if __name__ == "__main__":
    # 简单自测
    try:
        for row in db_pool.fetch_all("SELECT * FROM pay_order LIMIT 1"):
            print(row)
    except Exception as exc:
        print("DB self-test failed:", exc)
