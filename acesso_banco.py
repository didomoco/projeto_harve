import os
from typing import List

try:
    import pandas as pd
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text
    import pymysql  # noqa: F401 - valida o driver instalado
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variavel de ambiente ausente: {name}")
    return value


def create_mysql_engine():
    load_dotenv()
    host = _get_env("DB_HOST")
    port = _get_env("DB_PORT")
    user = _get_env("DB_USER")
    password = _get_env("DB_PASSWORD")
    db_name = _get_env("DB_NAME")

    # Usa pymysql como driver do MySQL
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(url)


ALLOWED_TABLES = [
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_datase",
]


def list_tables(engine) -> List[str]:
    query = text("SHOW TABLES;")
    with engine.connect() as conn:
        result = conn.execute(query).fetchall()
    available = {row[0] for row in result}
    return [name for name in ALLOWED_TABLES if name in available]


def preview_table(engine, table_name: str, limit: int = 5) -> pd.DataFrame:
    query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
    return pd.read_sql(query, engine)


def demo_tables(limit: int = 5) -> None:
    engine = create_mysql_engine()
    tables = list_tables(engine)
    if not tables:
        print("Nenhuma das tabelas esperadas foi encontrada no banco.")
        return

    if not tables:
        print("Nenhuma tabela encontrada no banco.")
        return

    print("Tabelas encontradas:")
    for name in tables:
        print(f"- {name}")

    print("\nAmostra de dados:")
    for name in tables:
        try:
            df = preview_table(engine, name, limit=limit)
            print(f"\nTabela: {name}")
            print(df)
        except Exception as exc:
            print(f"\nTabela: {name}")
            print(f"Erro ao ler dados: {exc}")


if __name__ == "__main__":
    demo_tables(limit=5)
