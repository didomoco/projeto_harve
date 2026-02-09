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


MESES_PT = {
    1: "janeiro",
    2: "fevereiro",
    3: "marco",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}


def mes_com_mais_pedidos(df: "pd.DataFrame") -> tuple[int, int]:
    if "order_purchase_timestamp" not in df.columns:
        raise KeyError("Coluna order_purchase_timestamp nao encontrada no dataframe.")

    compras = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df_valid = df[compras.notna()].copy()
    df_valid["mes"] = compras.dt.month
    df_valid["mes_nome"] = df_valid["mes"].map(MESES_PT)

    counts = df_valid["mes"].value_counts()
    if counts.empty:
        raise ValueError("Nao ha dados validos de compras.")

    top_mes = int(counts.idxmax())
    top_qtd = int(counts.max())
    return top_mes, top_qtd


def mes_com_maior_pagamento(df: "pd.DataFrame") -> tuple[int, float]:
    if "order_purchase_timestamp" not in df.columns:
        raise KeyError("Coluna order_purchase_timestamp nao encontrada no dataframe.")
    if "total_payment" not in df.columns:
        raise KeyError("Coluna total_payment nao encontrada no dataframe.")

    compras = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    pagamentos = pd.to_numeric(df["total_payment"], errors="coerce")

    df_valid = df[compras.notna() & pagamentos.notna()].copy()
    df_valid["mes"] = compras.dt.month
    df_valid["total_payment"] = pagamentos

    soma = df_valid.groupby("mes")["total_payment"].sum()
    if soma.empty:
        raise ValueError("Nao ha dados validos de pagamentos.")

    top_mes = int(soma.idxmax())
    top_total = float(soma.max())
    return top_mes, top_total


def main() -> None:
    df = load_dataframe()
    mes, qtd = mes_com_mais_pedidos(df)
    mes_nome = MESES_PT.get(mes, "desconhecido")

    mes_pag, total_pag = mes_com_maior_pagamento(df)
    mes_pag_nome = MESES_PT.get(mes_pag, "desconhecido")

    print("Vendas por mes (maior quantidade de pedidos, independente do ano):")
    print(f"- Mes (numero): {mes}")
    print(f"- Mes (nome): {mes_nome}")
    print(f"- Quantidade de pedidos: {qtd}")

    print("\nMes com maiores pagamentos (soma total, independente do ano):")
    print(f"- Mes (numero): {mes_pag}")
    print(f"- Mes (nome): {mes_pag_nome}")
    total_formatado = f"R$ {total_pag:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    print(f"- Total de pagamentos: {total_formatado}")


if __name__ == "__main__":
    main()
