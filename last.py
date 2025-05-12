import numpy as np
import streamlit as st
from PIL import Image
import datetime
import pandas as pd
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="MNIST Classifier"
)

# Chargement du modèle et des données
if 'model_loaded' not in st.session_state:
    @st.cache_resource
    def load_model_and_data():
        from tensorflow.keras.models import load_model
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test / 255.0
        model = load_model('best_mnist_model.keras')
        accuracy = model.evaluate(x_test, to_categorical(y_test, 10), verbose=0)[1]
        return model, x_test, y_test, accuracy

    st.session_state.model, st.session_state.x_test, st.session_state.y_test, st.session_state.model_accuracy = load_model_and_data()
    st.session_state.model_loaded = True

# CSS personnalisé
st.markdown("""
<style>
    .sidebar-menu {
        list-style: none;
        padding: 0;
        margin: 0 0 20px 0;
    }
    .menu-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 8px;
        color: var(--text-color);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s;
        cursor: pointer;
        font-size: 18px !important;  /* Augmentation de la taille de police */
    }
    .menu-item:hover {
        background-color: rgba(78, 115, 223, 0.1);
    }
    .menu-item.active {
        background-color: #4e73df;
        color: white !important;
    }
    .big-metric {
        font-size: 3rem !important;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
        border-left: 5px solid #4e73df;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Augmentation de la taille de police pour les boutons du menu */
    .stButton>button {
        font-size: 18px !important;
        padding: 12px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'draw'

# Sidebar
with st.sidebar:
    st.title("🔍 MNIST Classifier")

    if st.button("🎨 Dessin Main Levée", 
                type="primary" if st.session_state.current_page == 'draw' else "secondary"):
        st.session_state.current_page = 'draw'
        st.rerun()

    if st.button("🔢 Test MNIST Original", 
                type="primary" if st.session_state.current_page == 'test' else "secondary"):
        st.session_state.current_page = 'test'
        st.rerun()

    st.markdown("---")
    st.markdown("**Performances du modèle :**")
    st.code(f"Précision: {st.session_state.model_accuracy:.2%}")
    st.markdown("---")
    st.markdown(f"🔄 Dernière mise à jour: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
# Page Dessin Main Levée
if st.session_state.current_page == 'draw':
    st.title("🎨 Reconnaissance de chiffres")

    if 'canvas_module' not in st.session_state:
        @st.cache_resource
        def load_canvas():
            from streamlit_drawable_canvas import st_canvas
            return st_canvas
        st.session_state.canvas_module = load_canvas()

    col1, col2 = st.columns([1, 2])

    with col1:
        canvas = st.session_state.canvas_module(
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("Prédire le chiffre"):
            if canvas.image_data is not None and np.any(canvas.image_data < 255):
                with st.spinner("Analyse en cours..."):
                    img = Image.fromarray(canvas.image_data.astype('uint8'))
                    img = img.convert('L').resize((28, 28))
                    img_array = 1 - (np.array(img) / 255.0)
                    img_array = img_array.reshape(1, 28, 28, 1)

                    pred = st.session_state.model.predict(img_array, verbose=0)
                    predicted = np.argmax(pred)
                    confidence = np.max(pred)

                    with col2:
                        st.markdown(f"""
                        <div class="big-metric {'correct' if confidence > 0.9 else 'incorrect'}">
                            <div>Prédiction</div>
                            <div style="font-weight:bold">{predicted}</div>
                            <div>Confiance: {confidence:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        chart_data = pd.DataFrame(pred[0], columns=["Confiance"], index=[str(i) for i in range(10)])
                        st.bar_chart(chart_data)
            else:
                st.warning("Veuillez dessiner un chiffre avant de lancer la prédiction.")

# Page Test MNIST
elif st.session_state.current_page == 'test':
    st.title("🔢 Test avec données MNIST")

    # Dans la section test MNIST :
    if 'reload_count' not in st.session_state:
        st.session_state.reload_count = 0

    @st.cache_data
    def get_random_images(dummy_key):
        indices = np.random.choice(len(st.session_state.x_test), 10, replace=False)
        return st.session_state.x_test[indices], st.session_state.y_test[indices], indices

    if st.button("Charger de nouveaux exemples", key="reload_mnist") or 'test_images' not in st.session_state:
        st.session_state.reload_count += 1  # Incrémente le compteur
        st.session_state.test_images, st.session_state.test_labels, _ = get_random_images(st.session_state.reload_count)
    cols = st.columns(5)
    for i in range(10):
        with cols[i % 5]:
            img = st.session_state.test_images[i]
            true_label = st.session_state.test_labels[i]

            st.image(img, caption=f"Vérité: {true_label}", width=100)

            pred = st.session_state.model.predict(img.reshape(1, 28, 28, 1), verbose=0)
            predicted = np.argmax(pred)
            confidence = np.max(pred)

            color = "green" if predicted == true_label else "red"
            st.markdown(f"<span style='color:{color}'>Prédit: {predicted}</span>", unsafe_allow_html=True)
            st.progress(float(confidence), text=f"{confidence:.1%}")
