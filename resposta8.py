try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc

from df_bancodados import load_dataframe


def preparar_base(df: "pd.DataFrame") -> "pd.DataFrame":
    required = [
        "data_entrega_cliente",
        "data_estimada_entrega",
        "estado_cliente",
        "estado_vendedor",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no dataframe: {', '.join(missing)}")

    base = df[required].copy()
    base["data_entrega_cliente"] = pd.to_datetime(
        base["data_entrega_cliente"], errors="coerce"
    )
    base["data_estimada_entrega"] = pd.to_datetime(
        base["data_estimada_entrega"], errors="coerce"
    )

    base["estado_cliente"] = (
        base["estado_cliente"].astype("string").str.strip().replace("", pd.NA)
    )
    base["estado_vendedor"] = (
        base["estado_vendedor"].astype("string").str.strip().replace("", pd.NA)
    )

    base = base.dropna(
        subset=[
            "data_entrega_cliente",
            "data_estimada_entrega",
            "estado_cliente",
            "estado_vendedor",
        ]
    )

    base["dias_atraso"] = (
        base["data_entrega_cliente"] - base["data_estimada_entrega"]
    ).dt.days
    base["atrasado"] = base["dias_atraso"] > 0
    base["estados_diferentes"] = base["estado_vendedor"] != base["estado_cliente"]
    return base


def resumo_atrasos_por_estado(base: "pd.DataFrame") -> "pd.DataFrame":
    def montar(grupo: "pd.DataFrame") -> dict:
        total = int(grupo.shape[0])
        dif = int(grupo["estados_diferentes"].sum())
        iguais = total - dif
        pct_dif = (dif / total * 100) if total else 0.0
        pct_iguais = (iguais / total * 100) if total else 0.0
        return {
            "total": total,
            "estados_diferentes": dif,
            "estados_diferentes_pct": round(pct_dif, 2),
            "mesmo_estado": iguais,
            "mesmo_estado_pct": round(pct_iguais, 2),
        }

    atrasado = montar(base[base["atrasado"]])
    no_prazo = montar(base[~base["atrasado"]])

    return pd.DataFrame(
        [
            {"situacao_entrega": "atrasado", **atrasado},
            {"situacao_entrega": "antes_ou_no_prazo", **no_prazo},
        ]
    )


def top_rotas_atrasadas(base: "pd.DataFrame", top_n: int = 10) -> "pd.DataFrame":
    atrasos = base[base["atrasado"] & base["estados_diferentes"]]
    if atrasos.empty:
        return pd.DataFrame(columns=["estado_vendedor", "estado_cliente", "quantidade"])

    return (
        atrasos.groupby(["estado_vendedor", "estado_cliente"])
        .size()
        .reset_index(name="quantidade")
        .sort_values("quantidade", ascending=False)
        .head(top_n)
    )


def main() -> None:
    df = load_dataframe()
    base = preparar_base(df)

    resumo = resumo_atrasos_por_estado(base)
    top_rotas = top_rotas_atrasadas(base, top_n=10)

    print("8. Atrasos x Estados")
    print("Pergunta: entregas atrasadas aconteceram entre estados diferentes (vendedor x cliente)?\n")

    print("Resumo (proporcao de estados diferentes vs mesmo estado):")
    print(resumo.to_string(index=False))

    atrasado_row = resumo[resumo["situacao_entrega"] == "atrasado"]
    if not atrasado_row.empty:
        pct = float(atrasado_row["estados_diferentes_pct"].iloc[0])
        print(
            f"\nConclusao: entre as entregas atrasadas, {pct:.2f}% ocorreram entre estados diferentes."
        )

    if top_rotas.empty:
        print("\nTop rotas UF->UF com atraso: sem dados suficientes.")
    else:
        print("\nTop 10 rotas UF->UF (vendedor->cliente) com mais entregas atrasadas:")
        print(top_rotas.to_string(index=False))


if __name__ == "__main__":
    main()

