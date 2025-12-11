# Mini-projet : Analyse exploratoire d'un dataset hospitalier synthétique
# Cours : 8PRO408 - Outils de programmation pour la science des données
# Auteur : (à compléter)
# Fichier prêt à exécuter. Il produit :
# - un dossier "figures/" contenant toutes les visualisations demandées
# - un rapport résumé "report.md"
# - un archive "submission.zip" prête à déposer

# --- Instructions ---
# Placez le fichier CSV nommé 'hospital_data.csv' dans le même dossier que ce script
# puis exécutez :
#   python Mini-projet_EDA_hopital.py
# ou exécutez les cellules si vous le collez dans un notebook.

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from datetime import datetime

# Configuration graphique (ne spécifie pas de couleurs pour respecter consignes générales)
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- Fonctions utilitaires ---

def load_data(path='hospital_data.csv'):
    """Charge le dataset et renvoie un DataFrame."""
    df = pd.read_csv(path, low_memory=False)
    return df


def overview(df):
    """Affiche un aperçu général et renvoie un dictionnaire de statistiques utiles."""
    print('\n=== APERÇU GÉNÉRAL ===')
    print('Taille :', df.shape)
    print('\nColonnes et types :')
    print(df.dtypes)
    print('\nExemples (5 premières lignes) :')
    display = getattr(pd, 'set_option', None)
    print(df.head())

    missing = df.isna().sum().sort_values(ascending=False)
    print('\nValeurs manquantes (top 20) :')
    print(missing.head(20))

    duplicates = df.duplicated().sum()
    print('\nDoublons entiers :', duplicates)

    return {'shape': df.shape, 'missing': missing, 'duplicates': duplicates}


def simple_cleanup(df):
    """Nettoyage simple : conversion dates, normalisation de chaînes, strip, et types."""
    df = df.copy()

    # Normaliser les noms de colonnes : minuscules, underscores
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Exemple de colonnes typiques qu'on peut rencontrer dans le dataset synthétique
    # On essaye de convertir si présentes
    date_cols = [c for c in df.columns if 'date' in c or 'admission' in c or 'discharge' in c]
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            pass

    # Normaliser chaînes pour quelques colonnes habituelles
    str_cols = ['gender', 'blood_group', 'admission_type', 'diagnosis', 'medication', 'insurance', 'hospital', 'doctor']
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})

    # Calculer l'âge si date_of_birth et admission_date existent
    if 'date_of_birth' in df.columns and 'admission_date' in df.columns:
        try:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
            df['age'] = (df['admission_date'] - df['date_of_birth']).dt.days // 365
        except Exception:
            pass

    # Durée de séjour
    if 'admission_date' in df.columns and 'discharge_date' in df.columns:
        df['length_of_stay'] = (df['discharge_date'] - df['admission_date']).dt.days
        # valeurs négatives -> NaN (si erreurs)
        df.loc[df['length_of_stay'] < 0, 'length_of_stay'] = np.nan

    # Billing amount en float
    if 'billing_amount' in df.columns:
        df['billing_amount'] = pd.to_numeric(df['billing_amount'], errors='coerce')

    return df


def analyze_patients(df, out_dir='figures'):
    """Analyse patients : répartition âge, genre, groupe sanguin, résultats tests."""
    results = {}

    if 'age' in df.columns:
        plt.figure()
        sns.histplot(df['age'].dropna(), bins=30)
        plt.title('Distribution des âges')
        plt.xlabel('Âge')
        plt.savefig(f'{out_dir}/hist_age.png')
        plt.close()
        results['age_stats'] = df['age'].describe()

        # âge par groupe
        age_bins = [0, 18, 35, 50, 65, 80, 120]
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=['0-17','18-34','35-49','50-64','65-79','80+'], right=False)
        plt.figure()
        sns.countplot(y='age_group', data=df, order=df['age_group'].value_counts().index)
        plt.title('Répartition par groupe d\'âge')
        plt.savefig(f'{out_dir}/count_age_group.png')
        plt.close()

    if 'gender' in df.columns:
        plt.figure()
        sns.countplot(x='gender', data=df)
        plt.title('Répartition par genre')
        plt.savefig(f'{out_dir}/count_gender.png')
        plt.close()
        results['gender_counts'] = df['gender'].value_counts()

    if 'blood_group' in df.columns:
        plt.figure()
        sns.countplot(x='blood_group', data=df, order=df['blood_group'].value_counts().index)
        plt.title('Répartition groupe sanguin')
        plt.savefig(f'{out_dir}/count_blood_group.png')
        plt.close()
        results['blood_group_counts'] = df['blood_group'].value_counts()

    # Résultats de tests (Normal, Abnormal, Inconclusive)
    if 'test_result' in df.columns:
        plt.figure()
        sns.countplot(x='test_result', data=df)
        plt.title('Résultats des tests médicaux')
        plt.savefig(f'{out_dir}/count_test_result.png')
        plt.close()
        results['test_result_counts'] = df['test_result'].value_counts()

    return results


def analyze_pathologies(df, out_dir='figures'):
    """Analyse des conditions médicales (diagnoses)."""
    results = {}

    if 'diagnosis' in df.columns:
        # On suppose que diagnosis peut contenir plusieurs conditions séparées par ; ou ,
        diag_series = df['diagnosis'].dropna().astype(str).str.replace('\n',' ').str.split('[;,|]')
        # explode
        diag_exploded = diag_series.explode().str.strip()
        top_diag = diag_exploded.value_counts().head(30)
        plt.figure(figsize=(8,10))
        sns.barplot(y=top_diag.index, x=top_diag.values)
        plt.title('Top 30 des diagnostics')
        plt.xlabel('Nombre de patients')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/top_diagnostics.png')
        plt.close()
        results['top_diagnostics'] = top_diag

    if 'medication' in df.columns:
        meds = df['medication'].dropna().astype(str).str.split('[;,|]').explode().str.strip()
        top_meds = meds.value_counts().head(30)
        plt.figure(figsize=(8,10))
        sns.barplot(y=top_meds.index, x=top_meds.values)
        plt.title('Médications les plus fréquentes (top 30)')
        plt.xlabel('Occurrences')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/top_medications.png')
        plt.close()
        results['top_medications'] = top_meds

    return results


def analyze_hospital(df, out_dir='figures'):
    """Analyse hospitalière : admission type, durée, répartition par hôpital, médecin, assurance."""
    results = {}

    if 'admission_type' in df.columns:
        plt.figure()
        sns.countplot(x='admission_type', data=df)
        plt.title('Types d\'admission')
        plt.savefig(f'{out_dir}/count_admission_type.png')
        plt.close()
        results['admission_type_counts'] = df['admission_type'].value_counts()

    if 'length_of_stay' in df.columns:
        plt.figure()
        sns.boxplot(x=df['length_of_stay'].dropna())
        plt.title('Boxplot durée de séjour (jours)')
        plt.savefig(f'{out_dir}/box_length_of_stay.png')
        plt.close()
        results['length_of_stay_stats'] = df['length_of_stay'].describe()

    # Répartition par hôpital/ médecin / assurance
    for col in ['hospital', 'doctor', 'insurance']:
        if col in df.columns:
            top = df[col].value_counts().head(20)
            plt.figure(figsize=(8,6))
            sns.barplot(y=top.index, x=top.values)
            plt.title(f'Top 20: {col}')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/top_{col}.png')
            plt.close()
            results[f'top_{col}'] = top

    return results


def analyze_financial(df, out_dir='figures'):
    """Analyse des coûts facturés."""
    results = {}
    if 'billing_amount' in df.columns:
        plt.figure()
        sns.histplot(df['billing_amount'].dropna(), bins=50)
        plt.title('Distribution du montant facturé')
        plt.xlabel('Billing Amount')
        plt.savefig(f'{out_dir}/hist_billing_amount.png')
        plt.close()
        results['billing_stats'] = df['billing_amount'].describe()

        # Comparaison selon type d'admission
        if 'admission_type' in df.columns:
            plt.figure(figsize=(8,6))
            sns.boxplot(x='admission_type', y='billing_amount', data=df)
            plt.title('Billing amount par type d\'admission')
            plt.yscale('symlog')  # gestion des valeurs extrêmes
            plt.tight_layout()
            plt.savefig(f'{out_dir}/box_billing_by_admission_type.png')
            plt.close()

        # Comparaison selon insurance
        if 'insurance' in df.columns:
            top_ins = df['insurance'].value_counts().head(10).index
            subset = df[df['insurance'].isin(top_ins)]
            plt.figure(figsize=(10,6))
            sns.boxplot(x='insurance', y='billing_amount', data=subset)
            plt.title('Billing amount par assurance (top 10)')
            plt.yscale('symlog')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/box_billing_by_insurance.png')
            plt.close()

    return results


def save_report_and_zip(summary_stats, out_dir='figures', report_name='report.md', zip_name='submission.zip'):
    """Génère un court rapport markdown et crée le zip de soumission"""

    # Créer report minimal
    lines = []
    lines.append('# Mini-projet - Analyse exploratoire (résumé)')
    lines.append(f'**Généré le**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('\n## Principaux résultats')

    # Inclure quelques stats s'existant
    if 'age_stats' in summary_stats:
        lines.append('### Âge')
        lines.append(str(summary_stats['age_stats'].to_frame().to_markdown()))

    if 'gender_counts' in summary_stats:
        lines.append('\n### Genre')
        lines.append(summary_stats['gender_counts'].to_frame().to_markdown())

    if 'length_of_stay_stats' in summary_stats:
        lines.append('\n### Durée de séjour (jours)')
        lines.append(str(summary_stats['length_of_stay_stats'].to_frame().to_markdown()))

    if 'billing_stats' in summary_stats:
        lines.append('\n### Coûts facturés')
        lines.append(str(summary_stats['billing_stats'].to_frame().to_markdown()))

    # Sauvegarde
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(report_name, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(lines))

    # Créer zip: inclure report + figures
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(report_name)
        for root, _, files in os.walk(out_dir):
            for file in files:
                z.write(os.path.join(root, file))

    print(f'Rapport sauvegardé: {report_name}')
    print(f'Archive prête: {zip_name}')


# -------------------------
# Script principal
# -------------------------

def main(csv_path='hospital_data.csv'):
    out_dir = 'figures'
    Path(out_dir).mkdir(exist_ok=True)

    print('Chargement du dataset...')
    df = load_data(csv_path)

    overview_stats = overview(df)
    df_clean = simple_cleanup(df)

    print('\nLancement des analyses (peut prendre quelques secondes selon la taille)...')
    s1 = analyze_patients(df_clean, out_dir=out_dir)
    s2 = analyze_pathologies(df_clean, out_dir=out_dir)
    s3 = analyze_hospital(df_clean, out_dir=out_dir)
    s4 = analyze_financial(df_clean, out_dir=out_dir)

    # Concaténation des résultats utiles
    summary_stats = {**s1, **s2, **s3, **s4}
    # Ajouter quelques entrées d'overview si souhaité
    if 'age_stats' not in summary_stats and 'age' in df_clean.columns:
        summary_stats['age_stats'] = df_clean['age'].describe()
    if 'gender_counts' not in summary_stats and 'gender' in df_clean.columns:
        summary_stats['gender_counts'] = df_clean['gender'].value_counts()
    if 'length_of_stay_stats' not in summary_stats and 'length_of_stay' in df_clean.columns:
        summary_stats['length_of_stay_stats'] = df_clean['length_of_stay'].describe()
    if 'billing_stats' not in summary_stats and 'billing_amount' in df_clean.columns:
        summary_stats['billing_stats'] = df_clean['billing_amount'].describe()

    save_report_and_zip(summary_stats, out_dir=out_dir)


if __name__ == '__main__':
    # Vous pouvez remplacer le chemin si nécessaire
    main(csv_path='hospital_data.csv')
