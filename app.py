import streamlit as st
import pandas as pd
import numpy as np
from predicao import fazer_previsao

st.title('Classificando lírios')
st.markdown('Modelo para fins de estudo para classificar flores em \
     (setosa, versicolor, virginica) baseado na sépala/pétala \
    e comprimento/largura.')

st.header("Características da planta")
coluna_1, coluna_2 = st.columns(2)

with coluna_1:
    st.text("Características da sépala")
    comprimento_sepala = st.slider('Comprimento da sépala (cm)', 1.0, 8.0, 0.5)
    largura_sepala = st.slider('Largura da sépala (cm)', 2.0, 4.4, 0.5)

with coluna_2:
    st.text("Características da pétala")
    comprimento_petala = st.slider('Comprimento da pétala (cm)', 1.0, 7.0, 0.5)
    largura_petala = st.slider('Largura da pétala (cm)', 0.1, 2.5, 0.5)

if st.button("Fazer predição do tipo de lírio"):
    resultado = fazer_previsao(
        np.array([[comprimento_sepala,
                   largura_sepala,
                   comprimento_petala,
                   largura_petala]]))

    st.text("Provavelmente, a espécie desse lírio é " +
            resultado[0] +
            ".")

    st.text("O modelo apresenta uma probabilidade predita de " +
            resultado[1] +
            ".")

st.markdown(
    '`Criado por:` [brunomontezano](https://brunomontezano.github.io/) | \
    `Adaptado de:` [santiviquez](https://github.com/santiviquez/iris-streamlit)')