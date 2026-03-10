try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc

from df_bancodados import load_dataframe


def normalizar_nota(series: "pd.Series") -> "pd.Series":
    notas = pd.to_numeric(series, errors="coerce").round()
    notas = notas.astype("Int64")
    # Mantem apenas o range esperado de notas (1 a 5)
    notas = notas.where(notas.between(1, 5))
    return notas


def avaliar_notas(df: "pd.DataFrame") -> pd.Series:
    if "nota_avaliacao" not in df.columns:
        raise KeyError("Coluna nota_avaliacao nao encontrada no dataframe.")

    notas = normalizar_nota(df["nota_avaliacao"]).dropna()
    distribuicao = notas.value_counts(dropna=True).sort_index()
    return distribuicao


def avaliar_comentarios_por_nota(df: "pd.DataFrame") -> pd.DataFrame:
    if "nota_avaliacao" not in df.columns or "comentario_avaliacao" not in df.columns:
        raise KeyError("Colunas nota_avaliacao ou comentario_avaliacao nao encontradas.")

    analise = df[["nota_avaliacao", "comentario_avaliacao"]].copy()
    analise["nota_avaliacao"] = normalizar_nota(analise["nota_avaliacao"])
    analise = analise[analise["nota_avaliacao"].notna()].copy()
    analise["nota_avaliacao"] = analise["nota_avaliacao"].astype(int)

    comentarios = analise["comentario_avaliacao"].fillna("").astype(str).str.strip()
    analise["tem_comentario"] = comentarios.ne("")

    resumo = (
        analise.groupby("nota_avaliacao")["tem_comentario"]
        .agg(
            total_avaliacoes="count",
            com_comentario="sum",
        )
        .reset_index()
    )
    resumo["sem_comentario"] = (
        resumo["total_avaliacoes"] - resumo["com_comentario"]
    )
    resumo["perc_com_comentario"] = (
        resumo["com_comentario"] / resumo["total_avaliacoes"] * 100
    ).round(2)
    resumo["perfil_nota"] = resumo["nota_avaliacao"].apply(classificar_nota)

    return resumo.sort_values("nota_avaliacao")


def classificar_nota(nota: int) -> str:
    if nota >= 4:
        return "nota_alta"
    if nota == 3:
        return "nota_media"
    return "nota_baixa"


def main() -> None:
    df = load_dataframe()

    distribuicao = avaliar_notas(df)
    comentarios = avaliar_comentarios_por_nota(df)

    print("3.1 Notas - quantidade por nota:")
    for nota, quantidade in distribuicao.items():
        print(f"- Nota {int(nota)}: {int(quantidade)}")

    print("\n3.2 Comentarios por nota:")
    print(comentarios.to_string(index=False))

    altas = comentarios[comentarios["perfil_nota"] == "nota_alta"]
    baixas = comentarios[comentarios["perfil_nota"] == "nota_baixa"]

    if not altas.empty:
        media_altas = altas["perc_com_comentario"].mean()
        print(
            f"\n- Notas altas (4 e 5): {media_altas:.2f}% das avaliacoes possuem comentario."
        )
    if not baixas.empty:
        media_baixas = baixas["perc_com_comentario"].mean()
        print(
            f"- Notas baixas (1 e 2): {media_baixas:.2f}% das avaliacoes possuem comentario."
        )

    if not altas.empty and not baixas.empty:
        if media_baixas > media_altas:
            print("- Clientes tendem a comentar mais quando a nota e baixa.")
        elif media_altas > media_baixas:
            print("- Clientes tendem a comentar mais quando a nota e alta.")
        else:
            print("- A proporcao de comentarios e semelhante entre notas altas e baixas.")


if __name__ == "__main__":
    main()
