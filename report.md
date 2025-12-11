# Mini-projet - Analyse exploratoire (synthétique)

**Généré le**: 2025-12-11 03:39:01


## Principaux résultats


### Aperçu général

- Nombre d'enregistrements : 55500

- Période des admissions : 2018-01-01 à 2024-12-30

- Nombre de diagnostics uniques (approx) : 25


### Âge (statistiques)

|       |        age |
|:------|-----------:|
| count | 55500      |
| mean  |    44.9944 |
| std   |    23.0762 |
| min   |     0      |
| 25%   |    27      |
| 50%   |    45      |
| 75%   |    62      |
| max   |    89      |


### Répartition par genre

|        |   gender |
|:-------|---------:|
| Female |    27431 |
| Male   |    26981 |
| Other  |     1088 |


### Durée de séjour (jours)

|       |   length_of_stay |
|:------|-----------------:|
| count |      54374       |
| mean  |          4.01988 |
| std   |          4.52737 |
| min   |          0       |
| 25%   |          1       |
| 50%   |          3       |
| 75%   |          6       |
| max   |         47       |


### Montants facturés (billing_amount)

|       |   billing_amount |
|:------|-----------------:|
| count |         55500    |
| mean  |          6797.84 |
| std   |         12366.5  |
| min   |            31.83 |
| 25%   |          1460.71 |
| 50%   |          3280.31 |
| 75%   |          7439.8  |
| max   |        491086    |


### Top diagnostics (extrait)

|                         |   diagnosis |
|:------------------------|------------:|
| Hypothyroidism          |        2844 |
| Urinary Tract Infection |        2842 |
| Stroke                  |        2840 |
| Anxiety                 |        2813 |
| Kidney Disease          |        2810 |
| Asthma                  |        2807 |
| Arthritis               |        2791 |
| Pneumonia               |        2788 |
| COPD                    |        2784 |
| Appendicitis            |        2783 |


### Top medications (extrait)

|              |   medication |
|:-------------|-------------:|
| Metformin    |         4683 |
| Ibuprofen    |         4645 |
| Lisinopril   |         4642 |
| Furosemide   |         4642 |
| Aspirin      |         4641 |
| Paracetamol  |         4628 |
| Omeprazole   |         4622 |
| Insulin      |         4616 |
| Albuterol    |         4614 |
| Atorvastatin |         4572 |


## Visualisations

Les figures sont incluses dans le dossier `figures/` de l'archive.


Exemples de figures incluses : `hist_age.png`, `count_gender.png`, `top_diagnostics.png`, `hist_billing_amount.png`, `box_billing_by_insurance.png`.
