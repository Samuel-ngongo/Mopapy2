import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Aviator1 - IA Adaptativa", layout="centered")
st.title("Aviator1 - Super IA com Detecção de Mudança de Padrão")

# Histórico
if "valores" not in st.session_state:
    st.session_state.valores = []

# Entrada de dados
novo = st.text_input("Insira um valor (ex: 2.31):")
if st.button("Adicionar") and novo:
    try:
        st.session_state.valores.append(float(novo))
        st.success("Valor adicionado.")
    except:
        st.error("Formato inválido.")

# Função de previsão com regressão e confiança
def prever_valor(dados):
    if len(dados) < 5:
        return 1.50, 30  # valor mínimo com baixa confiança

    X = np.arange(len(dados)).reshape(-1, 1)
    y = np.array(dados)
    modelo = LinearRegression()
    modelo.fit(X, y)
    previsao = modelo.predict(np.array([[len(dados)]])).item()

    desvio = np.std(dados[-10:]) if len(dados) >= 10 else np.std(dados)
    confianca = max(10, 100 - desvio * 100)

    return round(previsao, 2), round(confianca, 1)

# Função para detectar mudança de padrão
def detectar_mudanca(dados):
    if len(dados) < 15:
        return False

    ultimos = np.array(dados[-5:])
    anteriores = np.array(dados[-10:-5])

    media_diff = abs(np.mean(ultimos) - np.mean(anteriores))
    desvio_diff = abs(np.std(ultimos) - np.std(anteriores))

    if media_diff > 1.0 or desvio_diff > 1.2:
        return True
    return False

# Análise
if st.session_state.valores:
    st.subheader("Histórico de valores")
    st.write([f"{v:.2f}x" for v in st.session_state.valores[-30:]])

    st.subheader("Análise e Previsão Inteligente")
    estimativa, confianca = prever_valor(st.session_state.valores)
    st.info(f"Próxima estimativa: {estimativa:.2f}x")
    st.info(f"Confiança: {confianca:.1f}%")

    if confianca >= 75:
        st.success("Alta confiança nas próximas rodadas.")
    elif confianca >= 50:
        st.warning("Confiança moderada. Observe antes de agir.")
    else:
        st.error("Confiança baixa. Possível mudança de padrão.")

    if detectar_mudanca(st.session_state.valores):
        st.warning("Mudança de padrão detectada! IA se ajustando...")

# Limpar
if st.button("Limpar dados"):
    st.session_state.valores = []
    st.success("Histórico limpo.")
