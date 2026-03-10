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


def analisar_prazo_satisfacao(df: "pd.DataFrame") -> pd.DataFrame:
    required = ["data_entrega_cliente", "data_estimada_entrega", "nota_avaliacao"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes: {', '.join(missing)}")

    entrega = pd.to_datetime(df["data_entrega_cliente"], errors="coerce")
    estimada = pd.to_datetime(df["data_estimada_entrega"], errors="coerce")
    nota = normalizar_nota(df["nota_avaliacao"])

    base = pd.DataFrame(
        {
            "entrega": entrega,
            "estimada": estimada,
            "nota": nota,
        }
    ).dropna()

    base["dias_atraso"] = (base["entrega"] - base["estimada"]).dt.days
    base["status_prazo"] = base["dias_atraso"].apply(
        lambda d: "depois_do_prazo" if d > 0 else "antes_ou_no_prazo"
    )

    resumo = (
        base.groupby("status_prazo")["nota"]
        .agg(
            total="count",
            media_nota_avaliacao="mean",
            mediana_nota_avaliacao="median",
        )
        .reset_index()
    )
    resumo["media_nota_avaliacao"] = resumo["media_nota_avaliacao"].round(2)
    resumo["mediana_nota_avaliacao"] = resumo["mediana_nota_avaliacao"].round(2)
    return resumo


def main() -> None:
    df = load_dataframe()
    resumo = analisar_prazo_satisfacao(df)

    print("4. Prazo x Satisfacao (nota_avaliacao):")
    print(resumo.to_string(index=False))

    if resumo.shape[0] == 2:
        antes = resumo.loc[resumo["status_prazo"] == "antes_ou_no_prazo"]
        depois = resumo.loc[resumo["status_prazo"] == "depois_do_prazo"]

        if not antes.empty and not depois.empty:
            media_antes = float(antes["media_nota_avaliacao"].iloc[0])
            media_depois = float(depois["media_nota_avaliacao"].iloc[0])
            med_antes = float(antes["mediana_nota_avaliacao"].iloc[0])
            med_depois = float(depois["mediana_nota_avaliacao"].iloc[0])

            print("\nConclusao:")
            if media_antes > media_depois and med_antes >= med_depois:
                print(
                    "- Entregas antes/no prazo tendem a ter maior satisfacao do que entregas depois do prazo."
                )
            elif media_depois > media_antes and med_depois >= med_antes:
                print(
                    "- Entregas depois do prazo tendem a ter maior satisfacao (resultado atipico; verifique dados)."
                )
            else:
                print(
                    "- Nao ha um padrao forte (media e mediana nao apontam para a mesma direcao)."
                )


if __name__ == "__main__":
    main()
