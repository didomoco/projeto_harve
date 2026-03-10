import math

import pandas as pd
import streamlit as st

from df_bancodados import load_dataframe


st.set_page_config(page_title="Projeto Olist - Analises", layout="wide")


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


def normalizar_nota(series: pd.Series) -> pd.Series:
    notas = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    return notas.where(notas.between(1, 5))


def spearman_sem_scipy(x: pd.Series, y: pd.Series) -> float:
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_xy) < 3:
        return float("nan")
    xr = df_xy["x"].rank(method="average")
    yr = df_xy["y"].rank(method="average")
    r = xr.corr(yr, method="pearson")
    return float(r) if pd.notna(r) else float("nan")


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_customer_key(df: pd.DataFrame) -> tuple[pd.Series, str]:
    col = pick_first_existing(df, ["customer_unique_id", "customer_id", "id_cliente", "cliente_id"])
    if col:
        return df[col].astype("string"), f"coluna:{col}"

    parts = []
    for c in ["cep_cliente_prefixo", "cidade_cliente", "estado_cliente", "cliente_lat", "cliente_lng"]:
        if c in df.columns:
            parts.append(df[c].astype("string").fillna(""))
    if not parts:
        raise KeyError("Sem chave de cliente e sem colunas de localizacao para proxy.")
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    key = key.str.strip().replace("", pd.NA)
    return key, "proxy:cep+cidade+estado+lat+lng (estimativa)"


def build_order_key(df: pd.DataFrame, customer_key: pd.Series) -> tuple[pd.Series, str]:
    col = pick_first_existing(df, ["order_id", "id_pedido", "pedido_id"])
    if col:
        return df[col].astype("string"), f"coluna:{col}"

    if "data_compra_pedido" not in df.columns:
        # Sem order_id e sem timestamp; cai para linha como unidade
        return pd.Series(df.index, dtype="Int64"), "linha (sem order_id/data)"

    ts = pd.to_datetime(df["data_compra_pedido"], errors="coerce").astype("string")
    ok = customer_key.fillna("").astype("string") + "|" + ts.fillna("")
    ok = ok.str.strip().replace("", pd.NA)
    return ok, "proxy:cliente+data_compra_pedido (estimativa)"


def to_order_level(df: pd.DataFrame, order_key: pd.Series, customer_key: pd.Series) -> tuple[pd.DataFrame, str]:
    work = df.copy()
    work["__order_key__"] = order_key
    work["__customer_key__"] = customer_key
    work = work[work["__order_key__"].notna() & work["__customer_key__"].notna()].copy()

    # Garante tipos consistentes para agregacoes (min/max) e comparacoes.
    for c in ["data_compra_pedido", "data_aprovacao_pedido", "data_entrega_cliente", "data_estimada_entrega"]:
        if c in work.columns:
            work[c] = pd.to_datetime(work[c], errors="coerce")

    entrega = pd.to_datetime(work.get("data_entrega_cliente"), errors="coerce")
    estimada = pd.to_datetime(work.get("data_estimada_entrega"), errors="coerce")
    work["__antes_ou_no_prazo__"] = entrega.le(estimada)

    if "nota_avaliacao" in work.columns:
        work["__nota__"] = normalizar_nota(work["nota_avaliacao"])
    else:
        work["__nota__"] = pd.Series([pd.NA] * len(work), dtype="Int64")

    agg: dict[str, str] = {"__customer_key__": "first", "__antes_ou_no_prazo__": "max", "__nota__": "median"}
    for c in ["estado_cliente", "cidade_cliente", "estado_vendedor", "cidade_vendedor", "tipo_pagamento", "max_parcelas", "status_pedido"]:
        if c in work.columns:
            agg[c] = "first"
    if "pagamento_total" in work.columns:
        agg["pagamento_total"] = "max"
    if "data_compra_pedido" in work.columns:
        agg["data_compra_pedido"] = "min"
    if "data_aprovacao_pedido" in work.columns:
        agg["data_aprovacao_pedido"] = "min"
    if "data_entrega_cliente" in work.columns:
        agg["data_entrega_cliente"] = "max"
    if "data_estimada_entrega" in work.columns:
        agg["data_estimada_entrega"] = "max"

    orders = work.groupby("__order_key__", dropna=True).agg(agg).reset_index()
    orders = orders.rename(columns={"__order_key__": "order_key", "__customer_key__": "customer_key"})
    return orders, "order_level"


@st.cache_data(show_spinner=False)
def load_df_cached() -> pd.DataFrame:
    return load_dataframe()


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info: dict[str, object] = {}
    df2 = df

    if "data_compra_pedido" in df2.columns:
        compras = pd.to_datetime(df2["data_compra_pedido"], errors="coerce")
        min_d = pd.to_datetime(compras.min()).date() if compras.notna().any() else None
        max_d = pd.to_datetime(compras.max()).date() if compras.notna().any() else None
        if min_d and max_d:
            start, end = st.sidebar.date_input(
                "Periodo (data_compra_pedido)",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )
            mask = compras.dt.date.between(start, end)
            df2 = df2[mask]
            info["periodo"] = (start, end)

    if "status_pedido" in df2.columns:
        status_vals = sorted(pd.Series(df2["status_pedido"]).dropna().astype("string").unique().tolist())
        selected = st.sidebar.multiselect("Status do pedido", status_vals, default=status_vals)
        if selected:
            df2 = df2[df2["status_pedido"].isin(selected)]
            info["status"] = selected

    return df2, info


st.title("Projeto Olist - Analises (Pandas)")
st.caption("Base: `df_bancodados.load_dataframe()` (colunas em pt-br, sem acentos).")

df_raw = load_df_cached()
df, filter_info = apply_filters(df_raw)

with st.expander("Resumo da base", expanded=False):
    st.write({"linhas": int(len(df)), "colunas": int(len(df.columns)), **filter_info})
    st.dataframe(pd.DataFrame({"coluna": df.columns}))


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "1. Tempo Entrega",
        "2. Vendas por Mes",
        "3. Satisfacao",
        "4. Prazo x Satisfacao",
        "5. Categorias",
        "6. Frete",
        "7. Geografia",
        "8. Atrasos x Estados",
        "9. Recompra",
    ]
)


with tab1:
    st.subheader("1. Tempo de entrega (aprovacao -> entrega)")
    if "data_aprovacao_pedido" not in df.columns or "data_entrega_cliente" not in df.columns:
        st.error("Colunas necessarias ausentes: `data_aprovacao_pedido`, `data_entrega_cliente`.")
    else:
        approved = pd.to_datetime(df["data_aprovacao_pedido"], errors="coerce")
        delivered = pd.to_datetime(df["data_entrega_cliente"], errors="coerce")
        mask = approved.notna() & delivered.notna()
        delta_days = (delivered[mask] - approved[mask]).dt.total_seconds() / 86400
        st.write(
            {
                "registros_validos": int(delta_days.shape[0]),
                "media_dias": int(round(float(delta_days.mean()))) if len(delta_days) else None,
                "mediana_dias": int(round(float(delta_days.median()))) if len(delta_days) else None,
            }
        )
        st.line_chart(delta_days.sample(min(2000, len(delta_days)), random_state=42).reset_index(drop=True))


with tab2:
    st.subheader("2. Vendas por mes (independente do ano)")
    if "data_compra_pedido" not in df.columns:
        st.error("Coluna necessaria ausente: `data_compra_pedido`.")
    else:
        customer_key, customer_key_src = build_customer_key(df)
        order_key, order_key_src = build_order_key(df, customer_key)
        orders, _ = to_order_level(df, order_key, customer_key)
        st.caption(f"Unidade de analise: pedidos ({order_key_src}); cliente ({customer_key_src}).")

        compras = pd.to_datetime(orders.get("data_compra_pedido"), errors="coerce")
        orders_valid = orders[compras.notna()].copy()
        orders_valid["mes"] = compras.dt.month
        count_mes = orders_valid["mes"].value_counts().sort_index()
        st.dataframe(
            pd.DataFrame(
                {"mes": count_mes.index.astype(int), "mes_nome": [MESES_PT.get(int(m), "") for m in count_mes.index], "qtd_pedidos": count_mes.values}
            )
        )
        top_mes = int(count_mes.idxmax()) if not count_mes.empty else None
        if top_mes:
            st.write({"mes_maior_volume": top_mes, "mes_nome": MESES_PT[top_mes], "qtd_pedidos": int(count_mes.max())})

        st.subheader("2b. Mes com maiores pagamentos (soma total, independente do ano)")
        if "pagamento_total" not in orders_valid.columns:
            st.info("Coluna `pagamento_total` nao encontrada para calcular maiores pagamentos.")
        else:
            orders_valid["pagamento_total"] = pd.to_numeric(orders_valid["pagamento_total"], errors="coerce")
            pay = orders_valid.dropna(subset=["pagamento_total"]).groupby("mes")["pagamento_total"].sum().sort_index()
            st.bar_chart(pay)
            if not pay.empty:
                m = int(pay.idxmax())
                total = float(pay.max())
                total_formatado = f"R$ {total:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                st.write({"mes_maior_pagamento": m, "mes_nome": MESES_PT[m], "total_pagamentos": total_formatado})


with tab3:
    st.subheader("3. Satisfacao")
    if "nota_avaliacao" not in df.columns:
        st.error("Coluna necessaria ausente: `nota_avaliacao`.")
    else:
        notas = normalizar_nota(df["nota_avaliacao"]).dropna()
        dist = notas.value_counts().sort_index()
        st.subheader("3.1 Quantidade por nota")
        st.dataframe(pd.DataFrame({"nota": dist.index.astype(int), "quantidade": dist.values}))
        st.bar_chart(dist)

        st.subheader("3.2 Comentarios por nota (notas altas x baixas)")
        if "comentario_avaliacao" not in df.columns:
            st.info("Coluna `comentario_avaliacao` nao encontrada.")
        else:
            tmp = df[["nota_avaliacao", "comentario_avaliacao"]].copy()
            tmp["nota"] = normalizar_nota(tmp["nota_avaliacao"])
            tmp = tmp[tmp["nota"].notna()].copy()
            comentarios = tmp["comentario_avaliacao"].fillna("").astype(str).str.strip()
            tmp["tem_comentario"] = comentarios.ne("")
            resumo = (
                tmp.groupby(tmp["nota"].astype(int))["tem_comentario"]
                .agg(total="count", com_comentario="sum")
                .reset_index()
                .rename(columns={"nota": "nota"})
            )
            resumo["sem_comentario"] = resumo["total"] - resumo["com_comentario"]
            resumo["perc_com_comentario"] = (resumo["com_comentario"] / resumo["total"] * 100).round(2)
            st.dataframe(resumo)


with tab4:
    st.subheader("4. Prazo x Satisfacao")
    required = ["data_entrega_cliente", "data_estimada_entrega", "nota_avaliacao"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        st.error(f"Colunas necessarias ausentes: {', '.join(miss)}")
    else:
        entrega = pd.to_datetime(df["data_entrega_cliente"], errors="coerce")
        estimada = pd.to_datetime(df["data_estimada_entrega"], errors="coerce")
        nota = normalizar_nota(df["nota_avaliacao"])
        base = pd.DataFrame({"entrega": entrega, "estimada": estimada, "nota": nota}).dropna()
        base["dias_atraso"] = (base["entrega"] - base["estimada"]).dt.days
        base["status_prazo"] = base["dias_atraso"].apply(lambda d: "depois_do_prazo" if d > 0 else "antes_ou_no_prazo")
        resumo = (
            base.groupby("status_prazo")["nota"]
            .agg(total="count", media_nota_avaliacao="mean", mediana_nota_avaliacao="median")
            .round(2)
            .reset_index()
        )
        st.dataframe(resumo)


with tab5:
    st.subheader("5. Categorias: mais/menos vendidos e relacoes")
    required = [
        "categoria_produto",
        "preco",
        "qtd_fotos_produto",
        "peso_produto_g",
        "comprimento_produto_cm",
        "altura_produto_cm",
        "largura_produto_cm",
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        st.error(f"Colunas necessarias ausentes: {', '.join(miss)}")
    else:
        base = df[required].copy()
        base["categoria_produto"] = base["categoria_produto"].astype("string").str.strip()
        base = base[base["categoria_produto"].notna() & (base["categoria_produto"] != "")]
        base["preco"] = pd.to_numeric(base["preco"], errors="coerce")
        base["qtd_fotos_produto"] = pd.to_numeric(base["qtd_fotos_produto"], errors="coerce")
        base["peso_produto_g"] = pd.to_numeric(base["peso_produto_g"], errors="coerce")
        for c in ["comprimento_produto_cm", "altura_produto_cm", "largura_produto_cm"]:
            base[c] = pd.to_numeric(base[c], errors="coerce")

        resumo = (
            base.groupby("categoria_produto")
            .agg(
                itens_vendidos=("categoria_produto", "size"),
                preco_medio=("preco", "mean"),
                preco_mediano=("preco", "median"),
                fotos_media=("qtd_fotos_produto", "mean"),
                peso_medio_g=("peso_produto_g", "mean"),
            )
            .reset_index()
        )
        resumo["preco_medio"] = resumo["preco_medio"].round(2)
        resumo["preco_mediano"] = resumo["preco_mediano"].round(2)
        resumo["fotos_media"] = resumo["fotos_media"].round(2)
        resumo["peso_medio_g"] = resumo["peso_medio_g"].round(2)

        colA, colB = st.columns(2)
        with colA:
            st.write("Top 10 categorias (itens vendidos)")
            st.dataframe(resumo.sort_values("itens_vendidos", ascending=False).head(10))
        with colB:
            st.write("Bottom 10 categorias (itens vendidos)")
            st.dataframe(resumo.sort_values("itens_vendidos", ascending=True).head(10))

        st.write("Relacao (Spearman por categoria, sem scipy)")
        corr = {}
        for c in ["preco_mediano", "preco_medio", "fotos_media"]:
            valid = resumo[["itens_vendidos", c]].dropna()
            if len(valid) >= 3:
                x = valid["itens_vendidos"].rank(method="average")
                y = valid[c].rank(method="average")
                r = x.corr(y, method="pearson")
                if pd.notna(r):
                    corr[c] = float(r)
        st.write(corr)


with tab6:
    st.subheader("6. Frete: peso/volume impactam no valor do frete?")
    required = ["valor_frete", "peso_produto_g", "comprimento_produto_cm", "altura_produto_cm", "largura_produto_cm"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        st.error(f"Colunas necessarias ausentes: {', '.join(miss)}")
    else:
        base = df[required].copy()
        base["valor_frete"] = pd.to_numeric(base["valor_frete"], errors="coerce")
        base["peso_produto_g"] = pd.to_numeric(base["peso_produto_g"], errors="coerce")
        for c in ["comprimento_produto_cm", "altura_produto_cm", "largura_produto_cm"]:
            base[c] = pd.to_numeric(base[c], errors="coerce")
        base["volume_cm3"] = base["comprimento_produto_cm"] * base["altura_produto_cm"] * base["largura_produto_cm"]

        r_peso = spearman_sem_scipy(base["peso_produto_g"], base["valor_frete"])
        r_volume = spearman_sem_scipy(base["volume_cm3"], base["valor_frete"])
        st.write({"spearman_peso_frete": r_peso, "spearman_volume_frete": r_volume})

        def quartis(var_col: str) -> pd.DataFrame:
            tmp = base[[var_col, "valor_frete"]].dropna().copy()
            if len(tmp) < 50:
                return pd.DataFrame()
            tmp["q"] = pd.qcut(tmp[var_col], q=4, labels=False, duplicates="drop")
            if tmp["q"].nunique(dropna=True) < 2:
                return pd.DataFrame()
            med = tmp.groupby("q")["valor_frete"].median().reset_index().rename(columns={"q": "quartil", "valor_frete": "frete_mediano"})
            med["quartil"] = med["quartil"].astype(int) + 1
            med["frete_mediano"] = med["frete_mediano"].round(2)
            return med

        colA, colB = st.columns(2)
        with colA:
            st.write("Frete mediano por quartil de peso")
            st.dataframe(quartis("peso_produto_g"))
        with colB:
            st.write("Frete mediano por quartil de volume")
            st.dataframe(quartis("volume_cm3"))


with tab7:
    st.subheader("7. Geografia: concentracao de clientes e vendedores")
    required = ["estado_cliente", "cidade_cliente", "cliente_lat", "cliente_lng", "estado_vendedor", "cidade_vendedor", "vendedor_lat", "vendedor_lng"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        st.error(f"Colunas necessarias ausentes: {', '.join(miss)}")
    else:
        def top_counts(series: pd.Series, name: str, top_n: int = 10) -> pd.DataFrame:
            s = series.dropna().astype("string").str.strip().replace("", pd.NA).dropna()
            return s.value_counts().head(top_n).rename_axis(name).reset_index(name="quantidade")

        colA, colB = st.columns(2)
        with colA:
            st.write("Clientes - Top estados/cidades")
            st.dataframe(top_counts(df["estado_cliente"], "estado"))
            st.dataframe(top_counts(df["cidade_cliente"], "cidade"))
        with colB:
            st.write("Vendedores - Top estados/cidades")
            st.dataframe(top_counts(df["estado_vendedor"], "estado"))
            st.dataframe(top_counts(df["cidade_vendedor"], "cidade"))

        st.write("Mapa (amostra) de clientes e vendedores")
        sample_n = st.slider("Amostra para mapa (por grupo)", 500, 5000, 1500, step=500)
        c = pd.DataFrame(
            {
                "lat": pd.to_numeric(df["cliente_lat"], errors="coerce"),
                "lon": pd.to_numeric(df["cliente_lng"], errors="coerce"),
            }
        ).dropna()
        v = pd.DataFrame(
            {
                "lat": pd.to_numeric(df["vendedor_lat"], errors="coerce"),
                "lon": pd.to_numeric(df["vendedor_lng"], errors="coerce"),
            }
        ).dropna()
        st.caption("Obs: `st.map` plota pontos; para heatmap/cluster precisaria lib extra.")
        colC, colD = st.columns(2)
        with colC:
            st.write("Clientes")
            st.map(c.sample(min(sample_n, len(c)), random_state=42))
        with colD:
            st.write("Vendedores")
            st.map(v.sample(min(sample_n, len(v)), random_state=42))


with tab8:
    st.subheader("8. Atrasos x Estados")
    required = ["data_entrega_cliente", "data_estimada_entrega", "estado_cliente", "estado_vendedor"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        st.error(f"Colunas necessarias ausentes: {', '.join(miss)}")
    else:
        entrega = pd.to_datetime(df["data_entrega_cliente"], errors="coerce")
        estimada = pd.to_datetime(df["data_estimada_entrega"], errors="coerce")
        ec = df["estado_cliente"].astype("string").str.strip().replace("", pd.NA)
        ev = df["estado_vendedor"].astype("string").str.strip().replace("", pd.NA)
        base = pd.DataFrame({"entrega": entrega, "estimada": estimada, "estado_cliente": ec, "estado_vendedor": ev}).dropna()
        base["dias_atraso"] = (base["entrega"] - base["estimada"]).dt.days
        base["atrasado"] = base["dias_atraso"] > 0
        base["estados_diferentes"] = base["estado_vendedor"] != base["estado_cliente"]

        def montar(sub: pd.DataFrame) -> dict:
            total = int(len(sub))
            dif = int(sub["estados_diferentes"].sum())
            return {
                "total": total,
                "estados_diferentes": dif,
                "estados_diferentes_pct": round((dif / total * 100) if total else 0.0, 2),
            }

        r1 = montar(base[base["atrasado"]])
        r2 = montar(base[~base["atrasado"]])
        st.dataframe(
            pd.DataFrame(
                [
                    {"situacao": "atrasado", **r1},
                    {"situacao": "antes_ou_no_prazo", **r2},
                ]
            )
        )

        atrasos_rotas = base[base["atrasado"] & base["estados_diferentes"]]
        rotas = (
            atrasos_rotas.groupby(["estado_vendedor", "estado_cliente"])
            .size()
            .reset_index(name="quantidade")
            .sort_values("quantidade", ascending=False)
            .head(10)
        )
        st.write("Top 10 rotas UF->UF com mais atrasos")
        st.dataframe(rotas)


with tab9:
    st.subheader("9. Recompra: padroes (estimado quando nao ha IDs)")
    customer_key, customer_key_src = build_customer_key(df)
    order_key, order_key_src = build_order_key(df, customer_key)
    orders, _ = to_order_level(df, order_key, customer_key)
    st.caption(f"Chave cliente: {customer_key_src}; chave pedido: {order_key_src}.")

    pedidos_por_cliente = orders.groupby("customer_key")["order_key"].nunique().rename("qtd_pedidos")
    recompra_keys = pedidos_por_cliente[pedidos_por_cliente >= 2].index
    orders["grupo_cliente"] = orders["customer_key"].isin(recompra_keys).map({True: "recompra", False: "primeira_compra"})

    st.write(
        {
            "clientes_total_estimado": int(pedidos_por_cliente.shape[0]),
            "clientes_recompra_estimado": int(len(recompra_keys)),
        }
    )

    colA, colB = st.columns(2)
    with colA:
        if "estado_cliente" in orders.columns:
            st.write("Estados (recompra)")
            st.dataframe(orders.loc[orders["grupo_cliente"] == "recompra", "estado_cliente"].value_counts().head(10).reset_index().rename(columns={"index": "estado", "estado_cliente": "quantidade"}))
        if "tipo_pagamento" in orders.columns:
            st.write("Tipo pagamento (recompra)")
            st.dataframe(orders.loc[orders["grupo_cliente"] == "recompra", "tipo_pagamento"].value_counts().head(10).reset_index().rename(columns={"index": "tipo_pagamento", "tipo_pagamento": "quantidade"}))
    with colB:
        if "estado_cliente" in orders.columns:
            st.write("Estados (primeira compra)")
            st.dataframe(orders.loc[orders["grupo_cliente"] == "primeira_compra", "estado_cliente"].value_counts().head(10).reset_index().rename(columns={"index": "estado", "estado_cliente": "quantidade"}))
        if "tipo_pagamento" in orders.columns:
            st.write("Tipo pagamento (primeira compra)")
            st.dataframe(orders.loc[orders["grupo_cliente"] == "primeira_compra", "tipo_pagamento"].value_counts().head(10).reset_index().rename(columns={"index": "tipo_pagamento", "tipo_pagamento": "quantidade"}))

    if "max_parcelas" in orders.columns:
        parcelas = pd.to_numeric(orders["max_parcelas"], errors="coerce")
        parc = orders.assign(parcelas=parcelas).groupby("grupo_cliente")["parcelas"].agg(["count", "mean", "median"]).round(2).reset_index()
        parc = parc.rename(columns={"count": "qtd_pedidos_com_parcelas", "mean": "parcelas_media", "median": "parcelas_mediana"})
        st.write("Parcelas (por pedido)")
        st.dataframe(parc)

    if "__antes_ou_no_prazo__" in orders.columns:
        prazo = orders.groupby("grupo_cliente")["__antes_ou_no_prazo__"].agg(total="count", antes_ou_no_prazo="sum").reset_index()
        prazo["antes_ou_no_prazo_pct"] = (prazo["antes_ou_no_prazo"] / prazo["total"] * 100).round(2)
        st.write("Entrega antes/no prazo (por pedido)")
        st.dataframe(prazo)

    if "__nota__" in orders.columns:
        notas = orders.groupby("grupo_cliente")["__nota__"].agg(["count", "mean", "median"]).round(2).reset_index()
        notas = notas.rename(columns={"count": "qtd_pedidos_com_nota", "mean": "media_nota", "median": "mediana_nota"})
        st.write("Satisfacao (nota por pedido)")
        st.dataframe(notas)

    st.info("Se a chave for `proxy`, recompra e uma estimativa (sem `customer_id`/`order_id`).")
