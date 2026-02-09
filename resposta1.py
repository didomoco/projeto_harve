import os

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc


def load_dataframe(path: str = "dados_banco.xlsx") -> "pd.DataFrame":
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    return pd.read_excel(path)


def calcular_tempo_entrega(df: "pd.DataFrame") -> tuple[float, float, int]:
    if "order_approved_at" not in df.columns or "order_delivered_customer_date" not in df.columns:
        raise KeyError("Colunas esperadas nao encontradas no dataframe.")

    approved = pd.to_datetime(df["order_approved_at"], errors="coerce")
    delivered = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")

    mask = approved.notna() & delivered.notna()
    delta_days = (delivered[mask] - approved[mask]).dt.total_seconds() / 86400

    mean_days = int(round(float(delta_days.mean())))
    median_days = int(round(float(delta_days.median())))
    count = int(delta_days.shape[0])
    return mean_days, median_days, count


def main() -> None:
    df = load_dataframe()
    mean_days, median_days, count = calcular_tempo_entrega(df)

    print("Tempo de entrega (da aprovacao ate a entrega):")
    print(f"- Registros validos: {count}")
    print(f"- Media (dias): {mean_days}")
    print(f"- Mediana (dias): {median_days}")


if __name__ == "__main__":
    main()
