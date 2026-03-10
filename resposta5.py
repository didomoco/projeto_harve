try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc

from df_bancodados import load_dataframe


def normalizar_inteiro(series: "pd.Series") -> "pd.Series":
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def preparar_base(df: "pd.DataFrame") -> "pd.DataFrame":
    required = [
        "categoria_produto",
        "preco",
        "qtd_fotos_produto",
        "peso_produto_g",
        "comprimento_produto_cm",
        "altura_produto_cm",
        "largura_produto_cm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes: {', '.join(missing)}")

    base = df[required].copy()
    base["categoria_produto"] = base["categoria_produto"].astype("string").str.strip()
    base = base[base["categoria_produto"].notna() & (base["categoria_produto"] != "")]

    base["preco"] = pd.to_numeric(base["preco"], errors="coerce")
    base["qtd_fotos_produto"] = normalizar_inteiro(base["qtd_fotos_produto"])
    base["peso_produto_g"] = pd.to_numeric(base["peso_produto_g"], errors="coerce")
    base["comprimento_produto_cm"] = pd.to_numeric(
        base["comprimento_produto_cm"], errors="coerce"
    )
    base["altura_produto_cm"] = pd.to_numeric(base["altura_produto_cm"], errors="coerce")
    base["largura_produto_cm"] = pd.to_numeric(
        base["largura_produto_cm"], errors="coerce"
    )
    return base


def sumarizar_por_categoria(base: "pd.DataFrame") -> "pd.DataFrame":
    resumo = (
        base.groupby("categoria_produto")
        .agg(
            itens_vendidos=("categoria_produto", "size"),
            preco_medio=("preco", "mean"),
            preco_mediano=("preco", "median"),
            fotos_media=("qtd_fotos_produto", "mean"),
            peso_medio_g=("peso_produto_g", "mean"),
            comprimento_medio_cm=("comprimento_produto_cm", "mean"),
            altura_media_cm=("altura_produto_cm", "mean"),
            largura_media_cm=("largura_produto_cm", "mean"),
        )
        .reset_index()
    )
    # Arredondamento para leitura no terminal
    resumo["preco_medio"] = resumo["preco_medio"].round(2)
    resumo["preco_mediano"] = resumo["preco_mediano"].round(2)
    resumo["fotos_media"] = resumo["fotos_media"].round(2)
    resumo["peso_medio_g"] = resumo["peso_medio_g"].round(2)
    resumo["comprimento_medio_cm"] = resumo["comprimento_medio_cm"].round(2)
    resumo["altura_media_cm"] = resumo["altura_media_cm"].round(2)
    resumo["largura_media_cm"] = resumo["largura_media_cm"].round(2)
    return resumo


def correlacao_categoria(resumo: "pd.DataFrame") -> dict[str, float]:
    # Correlacao entre volume (itens) e caracteristicas, por categoria.
    # Spearman e mais robusto para nao-linearidade / outliers.
    corr = {}
    for col in ["preco_mediano", "preco_medio", "fotos_media"]:
        valid = resumo[["itens_vendidos", col]].dropna()
        if len(valid) >= 3:
            x = valid["itens_vendidos"].rank(method="average")
            y = valid[col].rank(method="average")
            # Pearson nos ranks (Spearman), sem precisar de scipy.
            r = x.corr(y, method="pearson")
            if pd.notna(r):
                corr[col] = float(r)
    return corr


def impacto_fotos_item(base: "pd.DataFrame") -> "pd.DataFrame":
    # Analise simples por quantidade de fotos: quantos itens vendidos em cada valor de fotos.
    b = base[base["qtd_fotos_produto"].notna()].copy()
    b["qtd_fotos_produto"] = b["qtd_fotos_produto"].astype(int)
    tabela = (
        b.groupby("qtd_fotos_produto")
        .agg(itens_vendidos=("qtd_fotos_produto", "size"), preco_mediano=("preco", "median"))
        .reset_index()
        .sort_values("qtd_fotos_produto")
    )
    tabela["preco_mediano"] = tabela["preco_mediano"].round(2)
    return tabela


def main() -> None:
    df = load_dataframe()
    base = preparar_base(df)
    resumo = sumarizar_por_categoria(base)

    top = resumo.sort_values("itens_vendidos", ascending=False).head(10)
    bottom = resumo.sort_values("itens_vendidos", ascending=True).head(10)

    print("5. Categorias (por volume de itens vendidos):")
    print("\nMais vendidas (top 10):")
    print(top.to_string(index=False))
    print("\nMenos vendidas (bottom 10):")
    print(bottom.to_string(index=False))

    corr = correlacao_categoria(resumo)
    print("\nRelacao com preco (por categoria, correlacao Spearman com itens_vendidos):")
    if not corr:
        print("- Sem dados suficientes para calcular correlacao.")
    else:
        for k, v in corr.items():
            print(f"- itens_vendidos vs {k}: {v:.3f}")

    fotos_tab = impacto_fotos_item(base)
    print("\nQuantidade de fotos impacta nas vendas? (itens por qtd_fotos_produto):")
    print(fotos_tab.to_string(index=False))

    # Conclusao objetiva baseada na correlacao por categoria
    if "preco_mediano" in corr:
        if corr["preco_mediano"] < -0.1:
            preco_msg = "Categorias com preco mediano maior tendem a vender menos (tendencia negativa)."
        elif corr["preco_mediano"] > 0.1:
            preco_msg = "Categorias com preco mediano maior tendem a vender mais (tendencia positiva)."
        else:
            preco_msg = "Preco mediano por categoria nao mostra tendencia forte com volume."
        print(f"\nConclusao (preco): {preco_msg}")

    if "fotos_media" in corr:
        if corr["fotos_media"] < -0.1:
            fotos_msg = "Categorias com mais fotos em media tendem a vender menos (tendencia negativa)."
        elif corr["fotos_media"] > 0.1:
            fotos_msg = "Categorias com mais fotos em media tendem a vender mais (tendencia positiva)."
        else:
            fotos_msg = "Numero medio de fotos por categoria nao mostra tendencia forte com volume."
        print(f"Conclusao (fotos): {fotos_msg}")


if __name__ == "__main__":
    main()
