import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# Judul aplikasi
st.title('Gambar Angka dan Prediksi')

# Mengatur sidebar
st.sidebar.header("Pengaturan")

stroke_color = "#000000"  # Hitam
bg_color = "#FFFFFF"  # Putih
realtime_update = st.sidebar.checkbox("Update realtime", True)

# Membuat canvas untuk menggambar
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # warna fill canvas
    stroke_width=15,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=450,
    width=450,
    drawing_mode="freedraw",
    key="canvas",
)


# Load model
@st.cache_data
def load_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

model = load_model()

# Jika ada gambar di canvas, proses dan prediksi
if canvas_result.image_data is not None:
    # Mengubah gambar ke format grayscale dan ukuran 28x28 (sesuai dengan input model MNIST)
    image = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)

    # Inversi warna gambar jika latar belakang hitam
    image = np.invert(image)

    # Normalisasi gambar
    image = image / 255.0

    # Membuat prediksi
    prediction = model.predict(image.reshape(1, 28, 28, 1))
    predicted_digit = np.argmax(prediction)

    st.write(f"<p style='font-size: 35px; font-weight: bold;'>Prediksi angka: {predicted_digit}</p>", unsafe_allow_html=True)

