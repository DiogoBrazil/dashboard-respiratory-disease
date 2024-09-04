import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

# Carregar o modelo de machine learning
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/weights-att-03-09-2024.keras')
    return model

model = load_model()

# Função para preprocessar a imagem
def preprocess_image(image):
    # Converter a imagem PIL para um array numpy
    image = np.array(image)
    
    # Verificar se a imagem tem 3 canais (RGB). Se não, converte para RGB.
    if len(image.shape) == 2:  # Imagem em grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:  # Imagem com 4 canais (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Redimensionar a imagem para o tamanho que o modelo espera
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalizar
    image = image.reshape(-1, 256, 256, 3) 
    return image


# Função para fazer a previsão
def predict(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions

# Mapeamento das classes
classes = {0: 'Covid-19', 1: 'Normal', 2: 'Pneumonia viral', 3: 'Pneumonia bacteriana'}

# Interface do Streamlit
st.title("Detecção de doenças respiratórias")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem de Raio-X que contenha a região pulmonar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Exibir a imagem carregada
    st.image(image, caption='Imagem carregada', use_column_width=True)

    message_placeholder = st.empty()
    try:
        with st.spinner('Fazendo previsão...'):
            time.sleep(2)
            predictions = predict(image, model)            
            st.write("Resultados das predições:")
            fig, ax = plt.subplots(figsize=(6, 4))  # Ajustar o tamanho da caixa que envolve o gráfico
            bars = ax.bar(list(classes.values()), predictions[0] * 100, color='skyblue')
            ax.set_ylabel('Probabilidade (%)')
            ax.set_title('Previsão de Doenças')

            # Adicionar os valores em cima de cada barra e ajustar o espaço para não ultrapassar
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

            # Ajustar o espaçamento dos rótulos das classes e evitar sobreposição
            plt.xticks(rotation=15, ha='right')  # Rotaciona os rótulos das classes para evitar sobreposição
            plt.ylim(0, max(predictions[0]) * 100 + 10)  # Adiciona um espaço extra no topo para as porcentagens
            
            st.pyplot(fig)
            message_placeholder.success("Previsão realizada com sucesso!")
            time.sleep(2)
            message_placeholder.empty()
    except Exception as e:
        message_placeholder.error("Erro ao fazer a previsão")
        time.sleep(2)
        message_placeholder.empty()
