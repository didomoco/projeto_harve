import os

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    raise SystemExit(
        f"Dependencia ausente: {missing}. Instale com: pip install -r requirements.txt"
    ) from exc


RENAME_MAP = {
    "order_status": "status_pedido",
    "order_purchase_timestamp": "data_compra_pedido",
    "order_approved_at": "data_aprovacao_pedido",
    "order_delivered_carrier_date": "data_envio_transportadora",
    "order_delivered_customer_date": "data_entrega_cliente",
    "order_estimated_delivery_date": "data_estimada_entrega",
    "customer_zip_code_prefix": "cep_cliente_prefixo",
    "customer_city": "cidade_cliente",
    "customer_state": "estado_cliente",
    "order_item_id": "id_item_pedido",
    "shipping_limit_date": "data_limite_envio",
    "price": "preco",
    "freight_value": "valor_frete",
    "product_category_name": "categoria_produto",
    "product_name_lenght": "tamanho_nome_produto",
    "product_description_lenght": "tamanho_descricao_produto",
    "product_photos_qty": "qtd_fotos_produto",
    "product_weight_g": "peso_produto_g",
    "product_length_cm": "comprimento_produto_cm",
    "product_height_cm": "altura_produto_cm",
    "product_width_cm": "largura_produto_cm",
    "seller_zip_code_prefix": "cep_vendedor_prefixo",
    "seller_city": "cidade_vendedor",
    "seller_state": "estado_vendedor",
    "total_payment": "pagamento_total",
    "max_installments": "max_parcelas",
    "payment_type": "tipo_pagamento",
    "review_score": "nota_avaliacao",
    "review_comment": "comentario_avaliacao",
    "customer_lat": "cliente_lat",
    "customer_lng": "cliente_lng",
    "seller_lat": "vendedor_lat",
    "seller_lng": "vendedor_lng",
}

STATUS_MAP = {
    "approved": "aprovado",
    "canceled": "cancelado",
    "created": "criado",
    "delivered": "entregue",
    "invoiced": "faturado",
    "processing": "processando",
    "shipped": "enviado",
    "unavailable": "indisponivel",
}

PAYMENT_TYPE_MAP = {
    "boleto": "boleto",
    "credit_card": "cartao_credito",
    "debit_card": "cartao_debito",
    "not_defined": "nao_definido",
    "voucher": "voucher",
}

def load_dataframe(path: str = "dados_banco.xlsx") -> "pd.DataFrame":
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    df = pd.read_excel(path)
    df = df.rename(columns=RENAME_MAP)
    if "status_pedido" in df.columns:
        df["status_pedido"] = df["status_pedido"].map(STATUS_MAP).fillna(
            df["status_pedido"]
        )
    if "tipo_pagamento" in df.columns:
        df["tipo_pagamento"] = df["tipo_pagamento"].map(PAYMENT_TYPE_MAP).fillna(
            df["tipo_pagamento"]
        )
    return df


def main() -> None:
    df = load_dataframe()
    print(f"Linhas: {len(df)} | Colunas: {len(df.columns)}")
    print(df.head())


if __name__ == "__main__":
    main()
