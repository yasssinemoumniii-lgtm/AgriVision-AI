import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
from PIL import Image

# --------------------------------------------------------------------------
# CONFIGURATION DE L'INTERFACE VISUELLE
# --------------------------------------------------------------------------
st.set_page_config(page_title="AgriVision AI", layout="wide", page_icon="🥔")

st.title("🥔 AgriVision AI — Diagnostic & Système Multi-Agents")
st.write("Projet d'Ingénierie en Intelligence Artificielle et Science des Données")
st.markdown("---")

# ⚠️ METS TA VRAIE CLÉ GEMINI ICI (Générée sur Google AI Studio)
CLE_GEMINI = "AIzaSy..." 

# Noms des classes correspondant à ton dossier data
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model("modele_pomme_de_terre_transfer_learning.keras")

try:
    model = charger_modele()
except Exception as e:
    st.error(f"❌ Impossible de charger le fichier du modèle : {e}")

# --------------------------------------------------------------------------
# FONCTION GRAD-CAM ADAPTÉE POUR LES MODÈLES SÉQUENTIELS
# --------------------------------------------------------------------------
def generate_mobilenet_gradcam(img_array, model, last_conv_layer_name="conv2d_2"):
    conv_layer = model.get_layer(last_conv_layer_name)
    
    # Création d'un sous-modèle qui s'arrête à la couche de convolution conv2d_2
    sub_model_conv = tf.keras.models.Model(inputs=model.inputs, outputs=conv_layer.output)
    
    with tf.GradientTape() as tape:
        # Passage dans la première partie (jusqu'à conv2d_2)
        conv_outputs = sub_model_conv(img_array)
        tape.watch(conv_outputs)
        
        # Passage manuel dans le reste des couches du modèle Séquentiel
        x = conv_outputs
        start_index = model.layers.index(conv_layer) + 1
        for layer in model.layers[start_index:]:
            x = layer(x)
            
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Calcul des gradients par rapport à la sortie de conv2d_2
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Moyenne des gradients (Pooled Gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Application des poids aux cartes de caractéristiques
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalisation de la carte thermique entre 0 et 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# --------------------------------------------------------------------------
# CONNEXION SÉCURISÉE À L'API GEMINI VIA HTTP (AVEC DEBUG CLEAR)
# --------------------------------------------------------------------------
def appeler_gemini_api(prompt_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json()
        
        # Si l'API renvoie un message d'erreur (Clé invalide, quota, etc.)
        if 'error' in response_json:
            return f"❌ Erreur API Gemini ({response_json['error'].get('status')}) : {response_json['error'].get('message')}"
            
        # Si tout est OK, on extrait le texte
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"⚠️ Réponse inattendue de l'API : {response_json}"
            
    except Exception as e:
        return f"⚠️ Erreur de connexion avec l'agent : {e}"

# --------------------------------------------------------------------------
# INTERFACE UTILISATEUR PRINCIPALE (SIDEBAR ET CORPS)
# --------------------------------------------------------------------------
st.sidebar.header("📁 Importation de l'image")
uploaded_file = st.sidebar.file_uploader("Télécharger une photo de feuille...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture et redimensionnement de l'image (256x256)
    image_brute = Image.open(uploaded_file).convert('RGB')
    image_resize = image_brute.resize((256, 256))
    img_array = tf.keras.utils.img_to_array(image_resize)
    img_tensor = np.expand_dims(img_array, axis=0)

    # Classification par ton modèle CNN
    with st.spinner("🤖 Classification de l'image par l'IA..."):
        preds = model.predict(img_tensor, verbose=0)
        pred_class = np.argmax(preds[0])
        score_confiance = np.max(tf.nn.softmax(preds[0])) * 100
        nom_maladie = class_names[pred_class]

    # Affichage des résultats du diagnostic
    st.subheader("🎯 Verdict de l'IA Vision")
    if "healthy" in nom_maladie:
        st.success(f"**Feuille Saine** ({score_confiance:.2f}% de confiance)")
    else:
        st.error(f"**Pathologie détectée : {nom_maladie}** ({score_confiance:.2f}% de confiance)")

    # Génération et traitement de l'image Grad-CAM
    with st.spinner("🔬 Calcul des cartes d'explicabilité (Grad-CAM)..."):
        try:
            heatmap = generate_mobilenet_gradcam(img_tensor, model)
            
            img_cv = np.array(image_resize).astype("uint8")
            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_color = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
            
            # Affichage des images côte à côte
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_resize, caption="Feuille Originale", use_container_width=True)
            with col2:
                st.image(superimposed_img, caption="Explicabilité Grad-CAM (Zones Critiques)", use_container_width=True)
        except Exception as grad_error:
            st.error(f"⚠️ Erreur lors de la génération de la carte Grad-CAM : {grad_error}")

    st.markdown("---")
    st.subheader("🤖 Orchestration de l'Équipe Multi-Agents")
    
    if st.button("🚀 Activer les 3 Agents de Décision"):
        if CLE_GEMINI == "AIzaSy...":
            st.warning("⚠️ Remplace d'abord 'AIzaSy...' par ta vraie clé dans le code de app.py !")
        else:
            # AGENT 1 : Analyste XAI
            with st.spinner("🔬 Agent 1 (Analyste XAI) décrypte les features..."):
                prompt_xai = f"Tu es un expert en IA Explicable. Analyse pourquoi le modèle s'est focalisé sur les zones de lésions pour diagnostiquer : {nom_maladie}. Reste concis et technique."
                rapport_xai = appeler_gemini_api(prompt_xai, CLE_GEMINI)
            
            # AGENT 2 : Agent Terrain (Agronome)
            with st.spinner("⚡ Agent 2 (Agronome) formule les remèdes..."):
                prompt_agronome = f"En tant qu'agronome expert, donne les traitements d'urgence immédiats (bio et chimiques) pour : {nom_maladie}. Base-toi sur cette analyse : {rapport_xai}."
                solution_temps_reel = appeler_gemini_api(prompt_agronome, CLE_GEMINI)
            
            # AGENT 3 : Directeur (Planification)
            with st.spinner("📅 Agent 3 (Directeur) planifie le rapport final..."):
                prompt_directeur = f"Compile un rapport de gestion de culture et un calendrier d'action sur 4 semaines basé sur l'XAI: {rapport_xai} et les traitements: {solution_temps_reel}."
                rapport_final = appeler_gemini_api(prompt_directeur, CLE_GEMINI)

            # Séparation ergonomique sous forme d'onglets
            tab1, tab2, tab3 = st.tabs(["🔬 1. Rapport Explicabilité XAI", "⚡ 2. Solutions Terrain (Temps réel)", "📅 3. Plan Stratégique & Recommandations"])
            with tab1:
                st.markdown(rapport_xai)
            with tab2:
                st.markdown(solution_temps_reel)
            with tab3:
                st.markdown(rapport_final)
else:
    st.info("💡 Importe une photo de feuille de pomme de terre dans le menu de gauche pour démarrer la démo.")