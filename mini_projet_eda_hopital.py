import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

#=======================
#   FONCTIONS UTILITAIRES
#=======================

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def simple_cleanup(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Dates
    date_cols = [c for c in df.columns if "date" in c]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Text cleanup
    str_cols = ['gender','blood_group','admission_type','diagnosis','medication','insurance','hospital','doctor']
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    # Age
    if "date_of_birth" in df.columns and "admission_date" in df.columns:
        df["age"] = (df["admission_date"] - df["date_of_birth"]).dt.days // 365

    # Length of stay
    if "admission_date" in df.columns and "discharge_date" in df.columns:
        df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days
        df.loc[df["length_of_stay"] < 0, "length_of_stay"] = np.nan

    # Billing
    if "billing_amount" in df.columns:
        df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce")

    return df


#=======================
#     APPLICATION
#=======================

st.title("🏥 Mini-projet : Analyse Exploratoire Hospitalière")
st.write("Analyse interactive du dataset hospitalier (EDA).")

uploaded_file = st.file_uploader("📂 Importer le fichier hospital_data.csv", type=["csv"])

if uploaded_file is not None:

    df = load_data(uploaded_file)
    st.success("Fichier chargé !")

    st.subheader("Aperçu des données brutes")
    st.dataframe(df.head())

    # Nettoyage
    df_clean = simple_cleanup(df)

    st.subheader("Aperçu après nettoyage")
    st.write(df_clean.head())

    st.markdown("---")
    st.subheader("📊 Analyses Exploratoires")

    #=======================
    # Analyse Âges
    #=======================

    if "age" in df_clean.columns:
        st.write("### Distribution des âges")

        fig, ax = plt.subplots()
        sns.histplot(df_clean["age"].dropna(), bins=30, ax=ax)
        st.pyplot(fig)

        st.write(df_clean["age"].describe())

    #=======================
    # Analyse Genre
    #=======================

    if "gender" in df_clean.columns:
        st.write("### Répartition par genre")

        fig, ax = plt.subplots()
        sns.countplot(x="gender", data=df_clean, ax=ax)
        st.pyplot(fig)

        st.write(df_clean["gender"].value_counts())

    #=======================
    # Blood group
    #=======================

    if "blood_group" in df_clean.columns:
        st.write("### Groupes sanguins")

        fig, ax = plt.subplots()
        sns.countplot(x="blood_group", data=df_clean, ax=ax)
        st.pyplot(fig)

        st.write(df_clean["blood_group"].value_counts())

    #=======================
    # Diagnostics
    #=======================

    if "diagnosis" in df_clean.columns:
        st.write("### Diagnostics (Top 30)")

        diag = df_clean["diagnosis"].dropna().astype(str).str.split("[;,|]").explode().str.strip()
        top_diag = diag.value_counts().head(30)

        fig, ax = plt.subplots(figsize=(8, 10))
        sns.barplot(y=top_diag.index, x=top_diag.values, ax=ax)
        st.pyplot(fig)

        st.write(top_diag)

    #=======================
    # Durée séjour
    #=======================

    if "length_of_stay" in df_clean.columns:
        st.write("### Durée de séjour")

        fig, ax = plt.subplots()
        sns.boxplot(x=df_clean["length_of_stay"], ax=ax)
        st.pyplot(fig)

        st.write(df_clean["length_of_stay"].describe())

    #=======================
    # Billing
    #=======================

    if "billing_amount" in df_clean.columns:
        st.write("### Montant facturé")

        fig, ax = plt.subplots()
        sns.histplot(df_clean["billing_amount"].dropna(), bins=50, ax=ax)
        st.pyplot(fig)

        st.write(df_clean["billing_amount"].describe())

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
