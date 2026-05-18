import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ─────────────────────────────────────────────
# CONFIG DE LA PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚕️ Heart Disease Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #f0f7f4;
        color: #1a2e25;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #1a3d2b !important;
    }
    .main { background-color: #f0f7f4 !important; }
    section[data-testid="stSidebar"] {
        background-color: #1a3d2b !important;
    }
    .hero-card {
        background: linear-gradient(135deg, #1a3d2b 0%, #27ae60 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(39,174,96,0.25);
        border-left: 6px solid #a8e6c1;
    }
    .metric-card {
        background: white;
        border: 1px solid #c8e6d4;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        margin: 0.3rem;
        box-shadow: 0 2px 12px rgba(39,174,96,0.08);
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a7a45;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #5a8a6e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }
    .result-positive {
        background: linear-gradient(135deg, #c0392b, #96281b);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        box-shadow: 0 8px 32px rgba(192,57,43,0.3);
        border-left: 6px solid #f1948a;
    }
    .result-negative {
        background: linear-gradient(135deg, #1a3d2b, #27ae60);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        box-shadow: 0 8px 32px rgba(39,174,96,0.3);
        border-left: 6px solid #a8e6c1;
    }
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        border-left: 5px solid #27ae60;
        padding: 0.6rem 1rem;
        margin: 1.5rem 0 1rem 0;
        color: #1a3d2b;
        background: linear-gradient(90deg, #e8f5ee, transparent);
        border-radius: 0 8px 8px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a3d2b, #27ae60) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.7rem 2rem !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(39,174,96,0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ET MODÈLES
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = X.copy()
        df['target'] = y.values
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        df = df.fillna(df.median())
        return df
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

@st.cache_resource
def train_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'SVM':                 SVC(probability=True, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred       = model.predict(X_test_s)
        y_pred_proba = model.predict_proba(X_test_s)[:, 1]
        results[name] = {
            'model':      model,
            'Accuracy':   accuracy_score(y_test, y_pred),
            'Précision':  precision_score(y_test, y_pred),
            'Rappel':     recall_score(y_test, y_pred),
            'F1-Score':   f1_score(y_test, y_pred),
            'AUC-ROC':    roc_auc_score(y_test, y_pred_proba),
            'y_test':     y_test,
            'y_pred':     y_pred,
            'y_pred_proba': y_pred_proba
        }

    return models, results, scaler, X_test, y_test, X.columns.tolist()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 1rem 0;'>
    <span style='font-size:3rem'>⚕️</span>
    <h2 style='font-family:Syne,sans-serif; color:white; margin:0;'>Heart AI</h2>
    <p style='color:#e74c3c; font-size:0.85rem; font-weight:600; margin:0.3rem 0 0 0;'>Maré Richard</p>
    <p style='color:#e74c3c; font-size:0.85rem; font-weight:600; margin:0.1rem 0 0 0;'>Tapsoba Asnath</p>
    <p style='color:#666; font-size:0.75rem; margin:0.4rem 0 0 0;'>IFOAD — 2024</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Analyse des Données", "🤖 Modèles ML", "🔮 Prédiction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color:#666; font-size:0.75rem; text-align:center;'>
    Développé par<br>Maré Richard & Tapsoba Asnath<br>IFOAD 2024
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
with st.spinner("⏳ Chargement des données et entraînement des modèles..."):
    df = load_data()

if df is None:
    st.error("Impossible de charger les données. Vérifie ta connexion internet.")
    st.stop()

models, results, scaler, X_test, y_test, feature_names = train_models(df)


# ═══════════════════════════════════════════════
# PAGE 1 : ACCUEIL
# ═══════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown("""
    <div class='hero-card'>
        <h1 style='margin:0; font-size:2.5rem;'>⚕️ Prédiction des Maladies Cardiaques</h1>
        <p style='margin:0.5rem 0 0 0; opacity:0.9; font-size:1.1rem;'>
            Application de Machine Learning — Maré Richard & Tapsoba Asnath
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-label'>Patients</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{df.shape[1]-1}</div>
            <div class='metric-label'>Variables</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>6</div>
            <div class='metric-label'>Algorithmes</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        best = max(results, key=lambda x: results[x]['F1-Score'])
        best_score = results[best]['F1-Score']
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{best_score*100:.0f}%</div>
            <div class='metric-label'>Meilleur F1</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>À propos du projet</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Dataset :** Heart Disease UCI  
        **Objectif :** Prédire si un patient a une maladie cardiaque  
        **Type de problème :** Classification binaire (0 = sain, 1 = malade)

        **Algorithmes utilisés :**
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Decision Tree
        - Random Forest
        - AdaBoost
        """)
    with col2:
        st.markdown("""
        **Métriques d'évaluation :**
        - ✅ Accuracy
        - ✅ Précision
        - ✅ Rappel
        - ✅ F1-Score
        - ✅ AUC-ROC

        **Livrables :**
        - Jupyter Notebook
        - Application Streamlit
        """)

    # Distribution cible
    st.markdown("<div class='section-title'>Distribution des Classes</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#f0f7f4')
    for ax in axes:
        ax.set_facecolor('#f8fbf9')

    counts = df['target'].value_counts()
    axes[0].bar(['Sain (0)', 'Malade (1)'], [counts[0], counts[1]],
                color=['#27ae60', '#e74c3c'], edgecolor='white', linewidth=0.5)
    axes[0].set_title('Répartition des patients', color='white', fontsize=12)
    axes[0].tick_params(colors='white')
    axes[0].spines['bottom'].set_color('#444')
    axes[0].spines['left'].set_color('#444')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 2, str(v), ha='center', color='white', fontweight='bold')

    axes[1].pie([counts[0], counts[1]], labels=['Sain', 'Malade'],
                colors=['#27ae60', '#e74c3c'], autopct='%1.1f%%',
                startangle=90, textprops={'color': 'white'})
    axes[1].set_title('Proportion des classes', color='white', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# PAGE 2 : ANALYSE DES DONNÉES
# ═══════════════════════════════════════════════
elif page == "📊 Analyse des Données":
    st.markdown("# 📊 Analyse Exploratoire des Données")

    tab1, tab2, tab3 = st.tabs(["📋 Aperçu", "📈 Distributions", "🔗 Corrélations"])

    with tab1:
        st.markdown("<div class='section-title'>Aperçu du Dataset</div>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Statistiques descriptives**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col2:
            st.markdown("**Valeurs manquantes**")
            missing = pd.DataFrame({
                'Manquantes': df.isnull().sum(),
                'Pourcentage': (df.isnull().sum()/len(df)*100).round(2)
            })
            st.dataframe(missing, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Distribution des Variables</div>", unsafe_allow_html=True)
        variable = st.selectbox("Choisir une variable :", feature_names)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#f0f7f4')
        for ax in axes:
            ax.set_facecolor('#f8fbf9')
            ax.tick_params(colors='#1a3d2b')
            for spine in ax.spines.values():
                spine.set_color('#444')

        axes[0].hist(df[variable], bins=20, color='#e74c3c', edgecolor='white', alpha=0.8)
        axes[0].axvline(df[variable].mean(), color='yellow', linestyle='--', linewidth=2,
                        label=f'Moyenne: {df[variable].mean():.1f}')
        axes[0].set_title(f'Distribution de {variable}', color='#1a3d2b')
        axes[0].legend(labelcolor='#1a3d2b')

        df[df['target']==0][variable].hist(bins=20, alpha=0.7, color='#27ae60',
                                            label='Sain', ax=axes[1])
        df[df['target']==1][variable].hist(bins=20, alpha=0.7, color='#e74c3c',
                                            label='Malade', ax=axes[1])
        axes[1].set_title(f'{variable} par Diagnostic', color='#1a3d2b')
        axes[1].legend(labelcolor='#1a3d2b')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Stats par groupe
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Statistiques — Patients sains**")
            st.dataframe(df[df['target']==0][variable].describe().round(2))
        with col2:
            st.markdown("**Statistiques — Patients malades**")
            st.dataframe(df[df['target']==1][variable].describe().round(2))

    with tab3:
        st.markdown("<div class='section-title'>Matrice de Corrélation</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor('#f0f7f4')
        ax.set_facecolor('#f8fbf9')
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=mask, linewidths=0.5, ax=ax,
                    annot_kws={'color': 'white', 'size': 8})
        ax.tick_params(colors='#1a3d2b')
        ax.set_title('Matrice de Corrélation', color='white', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════
# PAGE 3 : MODÈLES ML
# ═══════════════════════════════════════════════
elif page == "🤖 Modèles ML":
    st.markdown("# 🤖 Comparaison des Modèles de Machine Learning")

    # Tableau des métriques
    st.markdown("<div class='section-title'>Tableau Comparatif des Performances</div>", unsafe_allow_html=True)
    metrics_data = {}
    for name, res in results.items():
        metrics_data[name] = {
            'Accuracy':  f"{res['Accuracy']*100:.2f}%",
            'Précision': f"{res['Précision']*100:.2f}%",
            'Rappel':    f"{res['Rappel']*100:.2f}%",
            'F1-Score':  f"{res['F1-Score']*100:.2f}%",
            'AUC-ROC':   f"{res['AUC-ROC']*100:.2f}%"
        }
    metrics_df = pd.DataFrame(metrics_data).T
    st.dataframe(metrics_df, use_container_width=True)

    best_name = max(results, key=lambda x: results[x]['F1-Score'])
    st.success(f"🏆 Meilleur modèle : **{best_name}** avec F1-Score = {results[best_name]['F1-Score']*100:.2f}%")

    # Graphique de comparaison
    st.markdown("<div class='section-title'>Visualisation des Performances</div>", unsafe_allow_html=True)
    metric_choice = st.selectbox("Métrique à afficher :", ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'AUC-ROC'])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#f0f7f4')
    ax.set_facecolor('#f8fbf9')

    names  = list(results.keys())
    values = [results[n][metric_choice] for n in names]
    colors = ['#e74c3c' if n == best_name else '#3498db' for n in names]

    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(metric_choice, color='#1a3d2b')
    ax.set_title(f'Comparaison — {metric_choice}', color='#1a3d2b', fontsize=14)
    ax.tick_params(colors='#1a3d2b', axis='both')
    ax.tick_params(axis='x', rotation=20)
    for spine in ax.spines.values():
        spine.set_color('#c8e6d4')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}%', ha='center', color='white', fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Courbes ROC
    st.markdown("<div class='section-title'>Courbes ROC</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f0f7f4')
    ax.set_facecolor('#f8fbf9')

    roc_colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12','#1abc9c']
    for (name, res), color in zip(results.items(), roc_colors):
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={res['AUC-ROC']:.3f})")

    ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Aléatoire')
    ax.set_xlabel('Taux Faux Positifs', color='#1a3d2b')
    ax.set_ylabel('Taux Vrais Positifs', color='#1a3d2b')
    ax.set_title('Courbes ROC — Tous les modèles', color='white', fontsize=14)
    ax.tick_params(colors='#1a3d2b')
    ax.legend(loc='lower right', labelcolor='#1a3d2b', facecolor='white', edgecolor='#c8e6d4')
    for spine in ax.spines.values():
        spine.set_color('#c8e6d4')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Matrice de confusion
    st.markdown("<div class='section-title'>Matrices de Confusion</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Choisir un modèle :", list(results.keys()))

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#f0f7f4')
    ax.set_facecolor('#f8fbf9')

    cm = confusion_matrix(results[selected_model]['y_test'], results[selected_model]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Sain', 'Malade'], yticklabels=['Sain', 'Malade'],
                linewidths=1, linecolor='white')
    ax.set_xlabel('Prédit', color='#1a3d2b')
    ax.set_ylabel('Réel', color='#1a3d2b')
    ax.set_title(f'Matrice de Confusion — {selected_model}', color='#1a3d2b')
    ax.tick_params(colors='#1a3d2b')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# PAGE 4 : PRÉDICTION
# ═══════════════════════════════════════════════
elif page == "🔮 Prédiction":
    st.markdown("# 🔮 Prédiction pour un Nouveau Patient")
    st.markdown("Remplis les informations du patient ci-dessous :")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Informations générales**")
        age    = st.slider("Âge (années)", 20, 80, 50)
        sex    = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
        cp     = st.selectbox("Type de douleur thoracique (cp)",
                               [0,1,2,3],
                               format_func=lambda x: {0:"Angine typique",1:"Angine atypique",
                                                       2:"Non-anginale",3:"Asymptomatique"}[x])
        fbs    = st.selectbox("Glycémie à jeun > 120 mg/dl",
                               [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        restecg = st.selectbox("Résultat ECG au repos",
                                [0,1,2],
                                format_func=lambda x: {0:"Normal",1:"Anomalie ST-T",2:"Hypertrophie"}[x])

    with col2:
        st.markdown("**🩺 Mesures cliniques**")
        trestbps = st.slider("Pression artérielle au repos (mm Hg)", 80, 200, 130)
        chol     = st.slider("Cholestérol (mg/dl)", 100, 600, 245)
        thalach  = st.slider("Fréquence cardiaque max", 60, 220, 150)
        oldpeak  = st.slider("Dépression ST (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

    with col3:
        st.markdown("**💪 Test à l'effort**")
        exang = st.selectbox("Angine à l'exercice",
                              [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        slope = st.selectbox("Pente du segment ST",
                              [0,1,2],
                              format_func=lambda x: {0:"Ascendante",1:"Plate",2:"Descendante"}[x])
        ca    = st.selectbox("Nombre de vaisseaux (fluoroscopie)", [0,1,2,3])
        thal  = st.selectbox("Thalassémie",
                              [3,6,7],
                              format_func=lambda x: {3:"Normal",6:"Défaut fixé",7:"Défaut réversible"}[x])
        model_choice = st.selectbox("🤖 Algorithme à utiliser", list(models.keys()))

    st.markdown("---")
    if st.button("🔮 Lancer la Prédiction"):
        # Préparer les données du patient
        patient_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                   restecg, thalach, exang, oldpeak, slope, ca, thal]])
        patient_scaled = scaler.transform(patient_data)

        # Prédiction
        model = models[model_choice]
        prediction  = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]

        col1, col2 = st.columns([2, 1])
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class='result-positive'>
                    ⚠️ RISQUE DE MALADIE CARDIAQUE DÉTECTÉ<br>
                    <span style='font-size:1rem; opacity:0.9;'>
                        Probabilité : {probability[1]*100:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
                st.warning("⚠️ Ce résultat est une aide à la décision médicale. Consultez un médecin.")
            else:
                st.markdown(f"""
                <div class='result-negative'>
                    ✅ AUCUNE MALADIE CARDIAQUE DÉTECTÉE<br>
                    <span style='font-size:1rem; opacity:0.9;'>
                        Probabilité d'être sain : {probability[0]*100:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
                st.info("ℹ️ Continuez à maintenir un mode de vie sain !")

        with col2:
            # Graphique probabilité
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#f0f7f4')
            ax.set_facecolor('#f8fbf9')
            bars = ax.bar(['Sain', 'Malade'], [probability[0]*100, probability[1]*100],
                          color=['#27ae60', '#e74c3c'], edgecolor='#1a3d2b')
            ax.set_ylim(0, 110)
            ax.set_ylabel('Probabilité (%)', color='#1a3d2b')
            ax.set_title('Probabilités', color='#1a3d2b')
            ax.tick_params(colors='#1a3d2b')
            for spine in ax.spines.values():
                spine.set_color('#444')
            for bar, val in zip(bars, [probability[0]*100, probability[1]*100]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', color='white', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Récapitulatif patient
        st.markdown("<div class='section-title'>Récapitulatif du Patient</div>", unsafe_allow_html=True)
        recap = pd.DataFrame({
            'Variable': ['Âge','Sexe','Douleur thoracique','Pression artérielle','Cholestérol',
                         'Glycémie à jeun','ECG au repos','Fréquence cardiaque max',
                         'Angine exercice','Dépression ST','Pente ST','Vaisseaux','Thalassémie'],
            'Valeur': [age, 'Homme' if sex==1 else 'Femme',
                       {0:'Angine typique',1:'Angine atypique',2:'Non-anginale',3:'Asymptomatique'}[cp],
                       f'{trestbps} mm Hg', f'{chol} mg/dl',
                       'Oui' if fbs==1 else 'Non',
                       {0:'Normal',1:'Anomalie ST-T',2:'Hypertrophie'}[restecg],
                       thalach, 'Oui' if exang==1 else 'Non',
                       oldpeak,
                       {0:'Ascendante',1:'Plate',2:'Descendante'}[slope],
                       ca,
                       {3:'Normal',6:'Défaut fixé',7:'Défaut réversible'}[thal]]
        })
        st.dataframe(recap, use_container_width=True, hide_index=True)
