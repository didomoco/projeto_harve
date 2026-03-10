try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc

from df_bancodados import load_dataframe


def normalizar_nota(series: "pd.Series") -> "pd.Series":
    notas = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    return notas.where(notas.between(1, 5))


def pick_first_existing(df: "pd.DataFrame", candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_customer_key(df: "pd.DataFrame") -> tuple["pd.Series", str]:
    # Prefer identificadores reais se existirem.
    col = pick_first_existing(
        df,
        [
            "customer_unique_id",
            "customer_id",
            "id_cliente",
            "cliente_id",
        ],
    )
    if col:
        return df[col].astype("string"), f"coluna:{col}"

    # Fallback: proxy por localizacao (nao garante unicidade; serve para estimativa).
    parts = []
    for c in ["cep_cliente_prefixo", "cidade_cliente", "estado_cliente", "cliente_lat", "cliente_lng"]:
        if c in df.columns:
            parts.append(df[c].astype("string").fillna(""))
    if not parts:
        raise KeyError(
            "Nao encontrei identificador de cliente nem colunas de localizacao para criar proxy."
        )
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    key = key.str.strip()
    key = key.replace("", pd.NA)
    return key, "proxy:cep+cidade+estado+lat+lng (estimativa)"


def build_order_key(df: "pd.DataFrame", customer_key: "pd.Series") -> tuple["pd.Series", str]:
    col = pick_first_existing(df, ["order_id", "id_pedido", "pedido_id"])
    if col:
        return df[col].astype("string"), f"coluna:{col}"

    # Fallback: cliente + timestamp de compra (aproxima pedido; pode colidir em casos raros).
    if "data_compra_pedido" not in df.columns:
        raise KeyError("Nao encontrei coluna de pedido nem data_compra_pedido para criar proxy.")

    ts = pd.to_datetime(df["data_compra_pedido"], errors="coerce").astype("string")
    ok = customer_key.fillna("") + "|" + ts.fillna("")
    ok = ok.replace("|", pd.NA) if False else ok  # no-op, mantem compatibilidade ascii
    ok = ok.replace("", pd.NA)
    return ok, "proxy:cliente+data_compra_pedido (estimativa)"


def to_order_level(df: "pd.DataFrame", order_key: "pd.Series") -> "pd.DataFrame":
    work = df.copy()
    work["__order_key__"] = order_key
    work = work[work["__order_key__"].notna()].copy()

    # Flags e campos usados nas analises
    entrega = pd.to_datetime(work.get("data_entrega_cliente"), errors="coerce")
    estimada = pd.to_datetime(work.get("data_estimada_entrega"), errors="coerce")
    work["__antes_ou_no_prazo__"] = (entrega <= estimada)

    if "nota_avaliacao" in work.columns:
        work["__nota__"] = normalizar_nota(work["nota_avaliacao"])
    else:
        work["__nota__"] = pd.Series([pd.NA] * len(work), dtype="Int64")

    cols_first = [c for c in [
        "estado_cliente",
        "cidade_cliente",
        "tipo_pagamento",
        "max_parcelas",
        "status_pedido",
    ] if c in work.columns]

    # Agrega por pedido, para nao duplicar por item
    agg = {c: "first" for c in cols_first}
    agg.update(
        {
            "__antes_ou_no_prazo__": "max",
            "__nota__": "median",
        }
    )
    if "pagamento_total" in work.columns:
        agg["pagamento_total"] = "max"

    orders = work.groupby("__order_key__", dropna=True).agg(agg).reset_index()
    orders = orders.rename(columns={"__order_key__": "order_key"})
    return orders


def top_counts(series: "pd.Series", top_n: int = 10) -> "pd.DataFrame":
    s = series.dropna().astype("string").str.strip().replace("", pd.NA).dropna()
    return (
        s.value_counts()
        .head(top_n)
        .rename_axis("valor")
        .reset_index(name="quantidade")
    )


def main() -> None:
    df = load_dataframe()

    customer_key, customer_key_src = build_customer_key(df)
    order_key, order_key_src = build_order_key(df, customer_key)

    df_work = df.copy()
    df_work["__customer_key__"] = customer_key
    df_work["__order_key__"] = order_key
    df_work = df_work[df_work["__customer_key__"].notna() & df_work["__order_key__"].notna()].copy()

    # Conta pedidos por cliente
    pedidos_por_cliente = (
        df_work.groupby("__customer_key__")["__order_key__"].nunique().rename("qtd_pedidos")
    )
    recompra_keys = pedidos_por_cliente[pedidos_por_cliente >= 2].index

    orders = to_order_level(df_work, df_work["__order_key__"])
    orders["__customer_key__"] = (
        df_work.groupby("__order_key__")["__customer_key__"].first().reindex(orders["order_key"]).values
    )
    orders["grupo_cliente"] = orders["__customer_key__"].isin(recompra_keys).map(
        {True: "recompra", False: "primeira_compra"}
    )

    print("9. Recompra: padrao de clientes que recompraram\n")
    print(f"- Chave cliente: {customer_key_src}")
    print(f"- Chave pedido: {order_key_src}")
    print(f"- Total clientes (estimado): {int(pedidos_por_cliente.shape[0])}")
    print(f"- Clientes com recompra (>=2 pedidos): {int(len(recompra_keys))}")

    # Localizacao (por pedido)
    if "estado_cliente" in orders.columns:
        print("\nLocalizacao (estado) - recompra:")
        print(top_counts(orders.loc[orders["grupo_cliente"] == "recompra", "estado_cliente"]).to_string(index=False))
        print("\nLocalizacao (estado) - primeira_compra:")
        print(top_counts(orders.loc[orders["grupo_cliente"] == "primeira_compra", "estado_cliente"]).to_string(index=False))

    if "cidade_cliente" in orders.columns:
        print("\nLocalizacao (cidade) - recompra:")
        print(top_counts(orders.loc[orders["grupo_cliente"] == "recompra", "cidade_cliente"]).to_string(index=False))

    # Metodo de pagamento / parcelas
    if "tipo_pagamento" in orders.columns:
        print("\nMetodo de pagamento - recompra:")
        print(top_counts(orders.loc[orders["grupo_cliente"] == "recompra", "tipo_pagamento"]).to_string(index=False))
        print("\nMetodo de pagamento - primeira_compra:")
        print(top_counts(orders.loc[orders["grupo_cliente"] == "primeira_compra", "tipo_pagamento"]).to_string(index=False))

    if "max_parcelas" in orders.columns:
        parcelas = pd.to_numeric(orders["max_parcelas"], errors="coerce")
        orders["__parcelas__"] = parcelas
        parc = orders.groupby("grupo_cliente")["__parcelas__"].agg(["count", "mean", "median"]).round(2).reset_index()
        parc = parc.rename(columns={"count": "qtd_pedidos_com_parcelas", "mean": "parcelas_media", "median": "parcelas_mediana"})
        print("\nParcelas (por pedido):")
        print(parc.to_string(index=False))

    # Entrega antes/no prazo
    if "__antes_ou_no_prazo__" in orders.columns:
        prazo = (
            orders.groupby("grupo_cliente")["__antes_ou_no_prazo__"]
            .agg(total="count", antes_ou_no_prazo="sum")
            .reset_index()
        )
        prazo["antes_ou_no_prazo_pct"] = (prazo["antes_ou_no_prazo"] / prazo["total"] * 100).round(2)
        print("\nEntrega antes/no prazo (por pedido):")
        print(prazo.to_string(index=False))

    # Notas
    if "__nota__" in orders.columns:
        notas = orders.groupby("grupo_cliente")["__nota__"].agg(["count", "mean", "median"]).round(2).reset_index()
        notas = notas.rename(columns={"count": "qtd_pedidos_com_nota", "mean": "media_nota", "median": "mediana_nota"})
        print("\nSatisfacao (nota por pedido):")
        print(notas.to_string(index=False))

    # Tipos de produtos (por item, dentro de pedidos de clientes que recompraram)
    if "categoria_produto" in df_work.columns:
        cats = df_work[["__customer_key__", "categoria_produto"]].copy()
        cats["categoria_produto"] = cats["categoria_produto"].astype("string").str.strip().replace("", pd.NA)
        cats = cats[cats["categoria_produto"].notna()]
        cats["grupo_cliente"] = cats["__customer_key__"].isin(recompra_keys).map(
            {True: "recompra", False: "primeira_compra"}
        )

        print("\nCategorias de produto (por item) - recompra (top 15):")
        print(
            top_counts(cats.loc[cats["grupo_cliente"] == "recompra", "categoria_produto"], top_n=15).to_string(index=False)
        )
        print("\nCategorias de produto (por item) - primeira_compra (top 15):")
        print(
            top_counts(cats.loc[cats["grupo_cliente"] == "primeira_compra", "categoria_produto"], top_n=15).to_string(index=False)
        )

    print(
        "\nObservacao: se a chave de cliente/pedido for 'proxy', os numeros sao aproximacoes (depende das colunas disponiveis no seu arquivo)."
    )


if __name__ == "__main__":
    main()

