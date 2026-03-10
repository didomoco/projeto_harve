try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc

from df_bancodados import load_dataframe


def concentracao_por_estado_cidade(
    df: "pd.DataFrame", estado_col: str, cidade_col: str, top_n: int = 10
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    estado = (
        df[estado_col]
        .dropna()
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    cidade = (
        df[cidade_col]
        .dropna()
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )

    top_estados = (
        estado.value_counts()
        .head(top_n)
        .rename_axis("estado")
        .reset_index(name="quantidade")
    )
    top_cidades = (
        cidade.value_counts()
        .head(top_n)
        .rename_axis("cidade")
        .reset_index(name="quantidade")
    )
    return top_estados, top_cidades


def concentracao_grade(
    df: "pd.DataFrame",
    lat_col: str,
    lng_col: str,
    precisao_decimais: int = 1,
    top_n: int = 15,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lng = pd.to_numeric(df[lng_col], errors="coerce")
    base = pd.DataFrame({"lat": lat, "lng": lng}).dropna()

    # Arredonda para criar bins. 1 decimal ~ 11km (aprox.), 2 decimais ~ 1.1km.
    base["lat_bin"] = base["lat"].round(precisao_decimais)
    base["lng_bin"] = base["lng"].round(precisao_decimais)

    grade = (
        base.groupby(["lat_bin", "lng_bin"])
        .size()
        .reset_index(name="quantidade")
        .sort_values("quantidade", ascending=False)
    )
    top_grade = grade.head(top_n).copy()
    return grade, top_grade


def main() -> None:
    df = load_dataframe()

    required = [
        "estado_cliente",
        "cidade_cliente",
        "cliente_lat",
        "cliente_lng",
        "estado_vendedor",
        "cidade_vendedor",
        "vendedor_lat",
        "vendedor_lng",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no dataframe: {', '.join(missing)}")

    print("7. Geografia: concentracao de clientes e vendedores\n")

    top_est_cli, top_cid_cli = concentracao_por_estado_cidade(
        df, "estado_cliente", "cidade_cliente", top_n=10
    )
    top_est_ven, top_cid_ven = concentracao_por_estado_cidade(
        df, "estado_vendedor", "cidade_vendedor", top_n=10
    )

    print("Clientes - Top estados:")
    print(top_est_cli.to_string(index=False))
    print("\nClientes - Top cidades:")
    print(top_cid_cli.to_string(index=False))

    print("\nVendedores - Top estados:")
    print(top_est_ven.to_string(index=False))
    print("\nVendedores - Top cidades:")
    print(top_cid_ven.to_string(index=False))

    # Grade geografica para "visualizacao" simples (pode plotar em Excel/PowerBI)
    precisao = 1
    grade_cli, top_grade_cli = concentracao_grade(
        df, "cliente_lat", "cliente_lng", precisao_decimais=precisao, top_n=15
    )
    grade_ven, top_grade_ven = concentracao_grade(
        df, "vendedor_lat", "vendedor_lng", precisao_decimais=precisao, top_n=15
    )

    print(f"\nClientes - Top bins (lat/lng arredondado a {precisao} decimal):")
    print(top_grade_cli.to_string(index=False))
    print(f"\nVendedores - Top bins (lat/lng arredondado a {precisao} decimal):")
    print(top_grade_ven.to_string(index=False))


if __name__ == "__main__":
    main()
