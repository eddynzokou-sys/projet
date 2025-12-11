# app.py
# Mini-projet : Analyse exploratoire d'un dataset hospitalier synthétique
# Cours : 8PRO408 - Outils de programmation pour la science des données
# Auteur : (à compléter)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import streamlit as st

# ===============================
# Configuration Streamlit
# ===============================
st.set_page_config(
    page_title="EDA - Dataset hospitalier",
    layout="wide"
)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ===============================
# Fonctions utilitaires
# ===============================

@st.cache_data
def load_data(uploaded_file):
    """Charge le dataset depuis un fichier uploadé."""
    df = pd.read_csv(uploaded_file, low_memory=False)
    return df


def overview(df):
    """Retourne quelques stats générales."""
    missing = df.isna().sum().sort_values(ascending=False)
    duplicates = df.duplicated().sum()
    return {
        "shape": df.shape,
        "missing": missing,
        "duplicates": duplicates
    }


def simple_cleanup(df):
    """Nettoyage simple : conversion dates, normalisation de chaînes, strip, et types."""
    df = df.copy()

    # Normaliser les noms de colonnes : minuscules, underscores
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Colonnes de dates potentielles
    date_cols = [c for c in df.columns if 'date' in c or 'admission' in c or 'discharge' in c]
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            pass

    # Normaliser chaînes
    str_cols = ['gender', 'blood_group', 'admission_type', 'diagnosis',
                'medication', 'insurance', 'hospital', 'doctor']
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})

    # Calculer l'âge si possible
    if 'date_of_birth' in df.columns and 'admission_date' in df.columns:
        try:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
            df['age'] = (df['admission_date'] - df['date_of_birth']).dt.days // 365
        except Exception:
            pass

    # Durée de séjour
    if 'admission_date' in df.columns and 'discharge_date' in df.columns:
        df['length_of_stay'] = (df['discharge_date'] - df['admission_date']).dt.days
        df.loc[df['length_of_stay'] < 0, 'length_of_stay'] = np.nan

    # Montant facturé
    if 'billing_amount' in df.columns:
        df['billing_amount'] = pd.to_numeric(df['billing_amount'], errors='coerce')

    return df


# ===============================
# Fonctions d'analyse + plots
# ===============================

def show_patient_analysis(df):
    st.subheader("Analyse des patients")

    results = {}

    # Distribution des âges
    if 'age' in df.columns:
        results['age_stats'] = df['age'].describe()
        st.markdown("**Distribution des âges**")
        fig, ax = plt.subplots()
        sns.histplot(df['age'].dropna(), bins=30, ax=ax)
        ax.set_xlabel("Âge")
        st.pyplot(fig)

        # Groupes d'âge
        age_bins = [0, 18, 35, 50, 65, 80, 120]
        df['age_group'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=['0-17', '18-34', '35-49', '50-64', '65-79', '80+'],
            right=False
        )
        st.markdown("**Répartition par groupe d'âge**")
        fig, ax = plt.subplots()
        order = df['age_group'].value_counts().index
        sns.countplot(y='age_group', data=df, order=order, ax=ax)
        st.pyplot(fig)

        st.markdown("**Statistiques sur l'âge**")
        st.dataframe(results['age_stats'].to_frame('valeur'))

    # Genre
    if 'gender' in df.columns:
        results['gender_counts'] = df['gender'].value_counts()
        st.markdown("**Répartition par genre**")
        fig, ax = plt.subplots()
        sns.countplot(x='gender', data=df, ax=ax)
        st.pyplot(fig)
        st.dataframe(results['gender_counts'].to_frame('effectif'))

    # Groupe sanguin
    if 'blood_group' in df.columns:
        results['blood_group_counts'] = df['blood_group'].value_counts()
        st.markdown("**Répartition des groupes sanguins**")
        fig, ax = plt.subplots()
        order = df['blood_group'].value_counts().index
        sns.countplot(x='blood_group', data=df, order=order, ax=ax)
        st.pyplot(fig)
        st.dataframe(results['blood_group_counts'].to_frame('effectif'))

    # Résultats de tests
    if 'test_result' in df.columns:
        results['test_result_counts'] = df['test_result'].value_counts()
        st.markdown("**Résultats des tests médicaux**")
        fig, ax = plt.subplots()
        sns.countplot(x='test_result', data=df, ax=ax)
        st.pyplot(fig)
        st.dataframe(results['test_result_counts'].to_frame('effectif'))

    return results


def show_pathology_analysis(df):
    st.subheader("Pathologies et médications")
    results = {}

    # Diagnostics
    if 'diagnosis' in df.columns:
        diag_series = df['diagnosis'].dropna().astype(str).str.replace('\n', ' ').str.split('[;,|]')
        diag_exploded = diag_series.explode().str.strip()
        top_diag = diag_exploded.value_counts().head(30)
        results['top_diagnostics'] = top_diag

        st.markdown("**Top 30 des diagnostics**")
        fig, ax = plt.subplots(figsize=(8, 10))
        sns.barplot(y=top_diag.index, x=top_diag.values, ax=ax)
        ax.set_xlabel("Nombre de patients")
        st.pyplot(fig)

    # Médications
    if 'medication' in df.columns:
        meds = df['medication'].dropna().astype(str).str.split('[;,|]').explode().str.strip()
        top_meds = meds.value_counts().head(30)
        results['top_medications'] = top_meds

        st.markdown("**Médications les plus fréquentes (top 30)**")
        fig, ax = plt.subplots(figsize=(8, 10))
        sns.barplot(y=top_meds.index, x=top_meds.values, ax=ax)
        ax.set_xlabel("Occurrences")
        st.pyplot(fig)

    return results


def show_hospital_analysis(df):
    st.subheader("Analyse hospitalière")
    results = {}

    # Type d'admission
    if 'admission_type' in df.columns:
        results['admission_type_counts'] = df['admission_type'].value_counts()
        st.markdown("**Types d'admission**")
        fig, ax = plt.subplots()
        sns.countplot(x='admission_type', data=df, ax=ax)
        st.pyplot(fig)
        st.dataframe(results['admission_type_counts'].to_frame('effectif'))

    # Durée de séjour
    if 'length_of_stay' in df.columns:
        results['length_of_stay_stats'] = df['length_of_stay'].describe()
        st.markdown("**Durée de séjour (jours)**")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['length_of_stay'].dropna(), ax=ax)
        st.pyplot(fig)
        st.dataframe(results['length_of_stay_stats'].to_frame('valeur'))

    # Répartition par hôpital / médecin / assurance
    for col in ['hospital', 'doctor', 'insurance']:
        if col in df.columns:
            top = df[col].value_counts().head(20)
            results[f'top_{col}'] = top
            st.markdown(f"**Top 20 : {col}**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(y=top.index, x=top.values, ax=ax)
            st.pyplot(fig)

    return results


def show_financial_analysis(df):
    st.subheader("Analyse financière")
    results = {}

    if 'billing_amount' in df.columns:
        results['billing_stats'] = df['billing_amount'].describe()

        st.markdown("**Distribution du montant facturé**")
        fig, ax = plt.subplots()
        sns.histplot(df['billing_amount'].dropna(), bins=50, ax=ax)
        ax.set_xlabel("Billing Amount")
        st.pyplot(fig)

        st.markdown("**Statistiques sur les montants facturés**")
        st.dataframe(results['billing_stats'].to_frame('valeur'))

        # Par type d'admission
        if 'admission_type' in df.columns:
            st.markdown("**Montant facturé par type d'admission**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='admission_type', y='billing_amount', data=df, ax=ax)
            ax.set_yscale('symlog')
            st.pyplot(fig)

        # Par assurance
        if 'insurance' in df.columns:
            st.markdown("**Montant facturé par assurance (top 10)**")
            top_ins = df['insurance'].value_counts().head(10).index
            subset = df[df['insurance'].isin(top_ins)]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='insurance', y='billing_amount', data=subset, ax=ax)
            ax.set_yscale('symlog')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)

    return results


def build_summary_report(summary_stats):
    """Construit un petit rapport texte à afficher dans Streamlit."""
    lines = []
    lines.append("# Mini-projet - Analyse exploratoire (résumé)")
    lines.append(f"**Généré le**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Principaux résultats")

    if 'age_stats' in summary_stats:
        lines.append("\n### Âge")
        lines.append(summary_stats['age_stats'].to_frame().to_markdown())

    if 'gender_counts' in summary_stats:
        lines.append("\n### Genre")
        lines.append(summary_stats['gender_counts'].to_frame().to_markdown())

    if 'length_of_stay_stats' in summary_stats:
        lines.append("\n### Durée de séjour (jours)")
        lines.append(summary_stats['length_of_stay_stats'].to_frame().to_markdown())

    if 'billing_stats' in summary_stats:
        lines.append("\n### Coûts facturés")
        lines.append(summary_stats['billing_stats'].to_frame().to_markdown())

    return "\n\n".join(lines)


# ===============================
# Interface Streamlit
# ===============================

st.title("📊 Analyse exploratoire - Dataset hospitalier synthétique")

st.markdown(
    """
Ce tableau de bord permet de réaliser une **analyse exploratoire** d’un dataset hospitalier
(fichier CSV).  
Téléverse ton fichier, puis navigue dans les onglets pour explorer les différentes dimensions :
patients, pathologies, hospitalisation et coûts.
"""
)

uploaded_file = st.file_uploader("📤 Téléverse le fichier CSV hospitalier", type=["csv"])

if uploaded_file is None:
    st.info("En attente d'un fichier CSV...")
    st.stop()

# Chargement et nettoyage
with st.spinner("Chargement et nettoyage des données..."):
    df_raw = load_data(uploaded_file)
    df = simple_cleanup(df_raw)

# Aperçu général
st.header("👀 Aperçu général du dataset")

overview_stats = overview(df)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Nombre de lignes", overview_stats['shape'][0])
with c2:
    st.metric("Nombre de colonnes", overview_stats['shape'][1])
with c3:
    st.metric("Lignes dupliquées", int(overview_stats['duplicates']))

st.subheader("5 premières lignes")
st.dataframe(df.head())

with st.expander("Valeurs manquantes par colonne (top 20)"):
    st.dataframe(overview_stats['missing'].head(20).to_frame("nb_missing"))

# Onglets d'analyse
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patients",
    "Pathologies",
    "Hospitalisation",
    "Financier",
    "Rapport résumé"
])

summary_stats = {}

with tab1:
    s1 = show_patient_analysis(df)
    summary_stats.update(s1)

with tab2:
    s2 = show_pathology_analysis(df)
    summary_stats.update(s2)

with tab3:
    s3 = show_hospital_analysis(df)
    summary_stats.update(s3)

with tab4:
    s4 = show_financial_analysis(df)
    summary_stats.update(s4)

with tab5:
    st.subheader("📝 Rapport résumé (Markdown)")
    # Compléter avec quelques champs utiles
    if 'age_stats' not in summary_stats and 'age' in df.columns:
        summary_stats['age_stats'] = df['age'].describe()
    if 'gender_counts' not in summary_stats and 'gender' in df.columns:
        summary_stats['gender_counts'] = df['gender'].value_counts()
    if 'length_of_stay_stats' not in summary_stats and 'length_of_stay' in df.columns:
        summary_stats['length_of_stay_stats'] = df['length_of_stay'].describe()
    if 'billing_stats' not in summary_stats and 'billing_amount' in df.columns:
        summary_stats['billing_stats'] = df['billing_amount'].describe()

    report_text = build_summary_report(summary_stats)
    st.markdown(report_text)

    # Option de téléchargement du rapport
    st.download_button(
        label="⬇️ Télécharger le rapport Markdown",
        data=report_text,
        file_name="report_hospital_eda.md",
        mime="text/markdown"
    )
