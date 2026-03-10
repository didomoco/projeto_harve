try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc

from df_bancodados import load_dataframe


def spearman_sem_scipy(x: "pd.Series", y: "pd.Series") -> float:
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_xy) < 3:
        return float("nan")
    xr = df_xy["x"].rank(method="average")
    yr = df_xy["y"].rank(method="average")
    r = xr.corr(yr, method="pearson")
    return float(r) if pd.notna(r) else float("nan")


def preparar_base(df: "pd.DataFrame") -> "pd.DataFrame":
    required = [
        "valor_frete",
        "peso_produto_g",
        "comprimento_produto_cm",
        "altura_produto_cm",
        "largura_produto_cm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes: {', '.join(missing)}")

    base = df[required].copy()
    base["valor_frete"] = pd.to_numeric(base["valor_frete"], errors="coerce")
    base["peso_produto_g"] = pd.to_numeric(base["peso_produto_g"], errors="coerce")
    base["comprimento_produto_cm"] = pd.to_numeric(
        base["comprimento_produto_cm"], errors="coerce"
    )
    base["altura_produto_cm"] = pd.to_numeric(base["altura_produto_cm"], errors="coerce")
    base["largura_produto_cm"] = pd.to_numeric(
        base["largura_produto_cm"], errors="coerce"
    )

    # Volume aproximado em cm^3
    base["volume_cm3"] = (
        base["comprimento_produto_cm"]
        * base["altura_produto_cm"]
        * base["largura_produto_cm"]
    )
    return base


def main() -> None:
    df = load_dataframe()
    base = preparar_base(df)

    r_peso = spearman_sem_scipy(base["peso_produto_g"], base["valor_frete"])
    r_volume = spearman_sem_scipy(base["volume_cm3"], base["valor_frete"])

    print("6. Frete: impacto de peso e volume no valor do frete")
    print(f"- Registros com dados: {int(base.dropna().shape[0])}")
    print(f"- Correlacao (Spearman) peso_produto_g vs valor_frete: {r_peso:.3f}")
    print(f"- Correlacao (Spearman) volume_cm3 vs valor_frete: {r_volume:.3f}")

    def forca(r: float) -> str:
        if pd.isna(r):
            return "sem dados suficientes"
        ar = abs(r)
        if ar < 0.1:
            return "muito fraca"
        if ar < 0.3:
            return "fraca"
        if ar < 0.5:
            return "moderada"
        return "forte"

    def direcao(r: float) -> str:
        if pd.isna(r):
            return "indefinida"
        return "positiva" if r > 0 else ("negativa" if r < 0 else "nula")

    def resumo_quartis(var_col: str, label: str) -> tuple[float, float, float]:
        tmp = base[[var_col, "valor_frete"]].dropna().copy()
        if len(tmp) < 50:
            return float("nan"), float("nan"), float("nan")
        # qcut pode falhar com muitos empates; duplicates="drop" evita erro.
        tmp["q"] = pd.qcut(tmp[var_col], q=4, labels=False, duplicates="drop")
        if tmp["q"].nunique(dropna=True) < 2:
            return float("nan"), float("nan"), float("nan")
        med = tmp.groupby("q")["valor_frete"].median().sort_index()
        q_min = float(med.iloc[0])
        q_max = float(med.iloc[-1])
        diff = q_max - q_min
        return q_min, q_max, diff

    print("\nConclusao:")
    print(
        f"- Peso vs frete: correlacao {direcao(r_peso)} e {forca(r_peso)} (Spearman={r_peso:.3f})."
    )
    print(
        f"- Volume vs frete: correlacao {direcao(r_volume)} e {forca(r_volume)} (Spearman={r_volume:.3f})."
    )

    p1, p4, pdiff = resumo_quartis("peso_produto_g", "peso")
    v1, v4, vdiff = resumo_quartis("volume_cm3", "volume")

    if pd.notna(pdiff):
        print(
            f"- Mediana do frete (peso): 1o quartil={p1:.2f} -> 4o quartil={p4:.2f} (diff={pdiff:.2f})."
        )
    if pd.notna(vdiff):
        print(
            f"- Mediana do frete (volume): 1o quartil={v1:.2f} -> 4o quartil={v4:.2f} (diff={vdiff:.2f})."
        )

    if pd.notna(r_peso) and pd.notna(r_volume):
        if r_peso > 0 and r_volume > 0:
            msg_dir = "Em geral, produtos mais pesados e/ou volumosos tendem a ter frete maior."
        elif r_peso < 0 and r_volume < 0:
            msg_dir = "Resultado atipico: aumento de peso/volume associado a frete menor; verifique dados."
        else:
            msg_dir = "Peso e volume nao apontam na mesma direcao; pode haver outros fatores dominando o frete."

        if abs(r_peso) > abs(r_volume) + 0.05:
            msg_forca = "Nesta base, peso tem relacao mais forte com o frete do que volume."
        elif abs(r_volume) > abs(r_peso) + 0.05:
            msg_forca = "Nesta base, volume tem relacao mais forte com o frete do que peso."
        else:
            msg_forca = "Nesta base, peso e volume tem relacao de intensidade parecida com o frete."

        print(f"- Interpretacao: {msg_dir}")
        print(f"- Comparacao: {msg_forca}")


if __name__ == "__main__":
    main()
