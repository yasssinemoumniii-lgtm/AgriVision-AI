# 🥔 AgriVision AI — Diagnostic & Système Multi-Agents

Ce projet d'ingénierie applique le Deep Learning à l'agronomie pour détecter et analyser automatiquement les maladies de la pomme de terre.

##  Fonctionnalités
- **Classification par CNN :** Détection automatique des pathologies (Feuille Saine, Early Blight, Late Blight).
- **IA Explicable (XAI) :** Visualisation en temps réel des zones de lésions sur la feuille via l'algorithme **Grad-CAM**.
- **Système Multi-Agents (GenAI) :** Intégration de l'API Gemini pour orchestrer 3 agents décisionnels virtuels en cascade (Analyste XAI, Agronome de terrain, Directeur d'exploitation) afin de générer des rapports et plans d'action agronomiques.

##  Technologies Utilisées
- **Python** (TensorFlow/Keras, OpenCV, Pandas, NumPy, Requests)
- **Streamlit** (Interface utilisateur web réactive)
- **Google AI Studio** (Modèle Gemini-2.5-Flash pour l'orchestration des agents)
