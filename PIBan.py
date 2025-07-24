import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from datetime import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import os

# Configuration initiale
st.title("🔍 Prédiction du PIB annuel pour 2024")
cet = pytz.timezone('CET')
current_date_time = cet.localize(datetime(2025, 7, 24, 19, 4))  # Updated to 07:04 PM CET
st.write(f"**Date et heure actuelles :** {current_date_time.strftime('%d/%m/%Y %H:%M %Z')}")

# Set random seed
random.seed(42)
np.random.seed(42)

# Initialize error log
error_log = []

# Normalize string function
def normalize_name(name):
    if pd.isna(name) or not isinstance(name, str):
        error_log.append(f"Valeur non textuelle ou NaN : {name}. Remplacement par 'inconnu'.")
        return "inconnu"
    original_name = name
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8').strip()
    name = re.sub(r"['’´]+", "'", name)
    name = re.sub(r'\s+', ' ', name).lower()
    name = name.replace("d'autre produits", "d'autres produits")
    name = name.replace("de lhabillement", "de l'habillement")
    name = name.replace("crise sociale", "Crise sociale")
    if name.startswith("impots nets de subventions") or name.startswith("impôts nets de subventions"):
        name = "impots nets de subventions sur les produits"
        error_log.append(f"Normalisé '{original_name}' en 'impots nets de subventions sur les produits'.")
    error_log.append(f"Normalisation : '{original_name}' -> '{name}'")
    return name

# Load and preprocess data
@st.cache_data
def load_and_preprocess(uploaded_file=None):
    try:
        if uploaded_file:
            # Reset file pointer and read raw content for debugging
            uploaded_file.seek(0)
            raw_content = uploaded_file.read().decode('utf-8')
            if not raw_content.strip():
                error_log.append("Le fichier uploadé est vide.")
                st.error("Le fichier uploadé est vide. Veuillez vérifier le fichier.")
                raise ValueError("Fichier vide.")
            error_log.append(f"Contenu brut du fichier uploadé : {raw_content[:200]}...")  # Log first 200 chars
            
            # Try reading with different separators
            for sep in [';', ',', '\t']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8')
                    if not df.empty and 'année' in df.columns:
                        error_log.append(f"Fichier chargé avec séparateur '{sep}'.")
                        break
                except Exception as e:
                    error_log.append(f"Échec de lecture avec séparateur '{sep}': {str(e)}")
            else:
                error_log.append("Impossible de lire le fichier avec les séparateurs testés (; , \\t).")
                st.error("Impossible de lire le fichier CSV. Vérifiez le format et le séparateur.")
                raise ValueError("Format CSV invalide ou séparateur incorrect.")
        else:
            # Fallback to default file
            default_file = "VA-2015-2023P.csv"  # Restore default file
            if not os.path.exists(default_file):
                error_log.append(f"Fichier '{default_file}' introuvable.")
                st.error(f"Fichier '{default_file}' introuvable. Vérifiez le chemin du fichier.")
                raise FileNotFoundError(f"Fichier '{default_file}' introuvable.")
            df = pd.read_csv(default_file, sep=';', encoding='utf-8')
            error_log.append(f"Fichier chargé comme CSV avec séparateur ';'.")

        # Validate DataFrame
        if df.empty or len(df.columns) == 0:
            error_log.append("Le fichier CSV ne contient aucune colonne valide.")
            st.error("Le fichier CSV ne contient aucune colonne valide. Vérifiez le contenu du fichier.")
            raise ValueError("Aucune colonne dans le fichier CSV.")
        if 'année' not in df.columns:
            error_log.append(f"Colonne 'année' absente. Colonnes trouvées : {df.columns.tolist()}")
            st.error(f"La colonne 'année' est requise. Colonnes trouvées : {df.columns.tolist()}")
            raise ValueError("Colonne 'année' manquante.")

        df = df.rename(columns={'année': 'Secteur'})
        error_log.append(f"Secteurs bruts dans le CSV : {df['Secteur'].tolist()}")
        df['Secteur'] = df['Secteur'].apply(normalize_name)
        error_log.append(f"Secteurs après normalisation : {df['Secteur'].tolist()}")

        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        sectors = [
            "agriculture, sylviculture et peche",
            "extraction petrole et gaz naturel",
            "extraction des produits miniers",
            "industries agro-alimentaires",
            "industrie du textile, de l'habillement et du cuir",
            "raffinage du petrole",
            "industries chimiques",
            "industrie d'autres produits mineraux non metalliques",
            "industries mecaniques et electriques",
            "industries diverses",
            "production et distribution de l'electricite et gaz",
            "production et distribution d'eau et gestion des dechets",
            "construction",
            "commerce et reparation",
            "transport et entreposage",
            "hebergement et restauration",
            "information et communication",
            "activites financieres et d'assurances",
            "administration publique et defense",
            "enseignement",
            "sante humaine et action sociale",
            "autres services marchands",
            "autres activites des menages",
            "activites des organisations associatives"
        ]
        macro_keywords = [
            "taux de chomage", "taux d'inflation", "taux d'interet", "dette publique", "pression fiscale",
            "politique monetaire internationale", "tensions geopolitiques regionales", "prix matieres premieres",
            "secheresse et desastre climatique", "pandemies", "Crise sociale",
            "impots nets de subventions sur les produits"
        ]
        macro_rates = ["taux de chomage", "taux d'inflation", "taux d'interet", "dette publique", "pression fiscale"]
        events = [
            "politique monetaire internationale", "tensions geopolitiques regionales", "prix matieres premieres",
            "secheresse et desastre climatique", "pandemies", "Crise sociale"
        ]

        if 'f' in df['Secteur'].values:
            error_log.append("Ligne 'f' détectée dans les secteurs. Elle sera exclue.")
            df = df[df['Secteur'] != 'f']

        if not df['Secteur'].str.contains("produit interieur brut pib", case=False).any():
            st.error("Aucune donnée PIB trouvée. Secteurs disponibles : {}".format(df['Secteur'].tolist()))
            error_log.append("Aucune donnée PIB trouvée dans le fichier.")
            raise ValueError("Données PIB manquantes.")

        impots_key = "impots nets de subventions sur les produits"
        df_macro = df[df['Secteur'].isin(macro_keywords)].copy()
        df_pib = df[df['Secteur'] == "produit interieur brut pib"].copy()
        df_secteurs = df[df['Secteur'].isin(sectors)].copy()
        df_secteurs = df_secteurs[df_secteurs['Secteur'] != impots_key]

        if impots_key not in df_macro['Secteur'].values:
            error_log.append(f"Erreur : '{impots_key}' non trouvé dans df_macro.")
            st.error(f"Erreur : '{impots_key}' non trouvé dans df_macro.")
            raise ValueError(f"'{impots_key}' manquant dans df_macro.")
        if impots_key in df_secteurs['Secteur'].values:
            error_log.append(f"Erreur : '{impots_key}' trouvé dans df_secteurs après exclusion.")
            st.error(f"Erreur : '{impots_key}' trouvé dans df_secteurs après exclusion.")
            raise ValueError(f"'{impots_key}' trouvé dans df_secteurs après exclusion.")

        error_log.append(f"Secteurs dans df_secteurs : {df_secteurs['Secteur'].tolist()}")
        error_log.append(f"Macros dans df_macro : {df_macro['Secteur'].tolist()}")

        if df_pib.empty:
            st.error("Aucune donnée PIB trouvée. Secteurs disponibles : {}".format(df['Secteur'].tolist()))
            error_log.append("Aucune donnée PIB trouvée dans le fichier.")
            raise ValueError("Données PIB manquantes.")

        missing_sectors = [s for s in sectors if s not in df['Secteur'].values]
        missing_macro = [m for m in macro_keywords if m not in df['Secteur'].values]
        if missing_sectors:
            st.warning(f"Secteurs manquants : {missing_sectors}. Utilisation de la moyenne des secteurs disponibles.")
            error_log.append(f"Secteurs manquants : {missing_sectors}")
        if missing_macro:
            st.warning(f"Macros manquants : {missing_macro}. Utilisation de valeurs par défaut (0).")
            error_log.append(f"Macros manquants : {missing_macro}")

        df_macro.set_index("Secteur", inplace=True)
        df_pib.set_index("Secteur", inplace=True)
        df_secteurs.set_index("Secteur", inplace=True)

        df_macro_T = df_macro.transpose()
        df_secteurs_T = df_secteurs.transpose()
        df_pib_T = df_pib.transpose()

        X_df = pd.concat([df_secteurs_T, df_macro_T[macro_rates + events]], axis=1).dropna()
        y_df = df_pib_T.loc[X_df.index]

        error_log.append(f"Colonnes dans X_df après concaténation : {list(X_df.columns)}")

        if y_df.empty:
            st.error("y_df vide après alignement avec X_df. Indices X_df : {}. Indices df_pib_T : {}".format(X_df.index.tolist(), df_pib_T.index.tolist()))
            error_log.append("y_df vide après alignement.")
            raise ValueError("Données PIB vides après prétraitement.")

        key_sectors = [
            "agriculture, sylviculture et peche", "industries mecaniques et electriques",
            "hebergement et restauration", "information et communication",
            "activites financieres et d'assurances"
        ]
        for sector in key_sectors:
            if sector in X_df.columns:
                X_df[f"{sector}_lag1"] = X_df[sector].shift(1).fillna(X_df[sector].mean())
            else:
                X_df[f"{sector}_lag1"] = X_df[sectors].mean(axis=1).shift(1).fillna(X_df[sectors].mean().mean()) if sectors else 0
                error_log.append(f"Feature décalée '{sector}_lag1' ajoutée avec moyenne des secteurs car '{sector}' est absent.")

        for rate in macro_rates:
            if rate in X_df.columns:
                X_df[f"{rate}_lag1"] = X_df[rate].shift(1).fillna(X_df[rate].mean())
            else:
                X_df[f"{rate}_lag1"] = 0
                error_log.append(f"Feature décalée '{rate}_lag1' ajoutée avec valeur 0 car '{rate}' est absent.")

        X_df['gdp_lag1'] = y_df.shift(1).fillna(y_df.mean())

        expected_features = sectors + macro_rates + events + [f"{s}_lag1" for s in key_sectors] + [f"{r}_lag1" for r in macro_rates] + ['gdp_lag1']
        error_log.append(f"Colonnes attendues dans X_df : {expected_features} (nombre: {len(expected_features)})")

        missing_cols = [col for col in expected_features if col not in X_df.columns]
        extra_cols = [col for col in X_df.columns if col not in expected_features]
        if missing_cols:
            existing_cols = [col for col in sectors + macro_rates + events if col in X_df.columns]
            for col in missing_cols:
                if col in sectors and existing_cols:
                    X_df[col] = X_df[existing_cols].mean(axis=1)
                    error_log.append(f"Feature manquante '{col}' ajoutée avec la moyenne des secteurs disponibles.")
                elif col.endswith('_lag1') and col.replace('_lag1', '') in X_df.columns:
                    X_df[col] = X_df[col.replace('_lag1', '')].shift(1).fillna(X_df[col.replace('_lag1', '')].mean())
                    error_log.append(f"Feature manquante '{col}' ajoutée avec décalage.")
                else:
                    X_df[col] = 0
                    error_log.append(f"Feature manquante '{col}' ajoutée avec valeur 0.")
        if extra_cols:
            st.warning(f"Colonnes supplémentaires dans X_df : {extra_cols}")
            error_log.append(f"Colonnes supplémentaires dans X_df : {extra_cols}")
            X_df = X_df.drop(columns=extra_cols, errors='ignore')

        error_log.append(f"Colonnes dans X_df après ajout des features manquantes : {list(X_df.columns)}")
        error_log.append(f"Nombre de colonnes dans X_df : {X_df.shape[1]} (attendu : {len(expected_features)})")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_df = X_df[expected_features]
        scaler_X.fit(X_df)
        error_log.append(f"Scaler_X ajusté sur {scaler_X.n_features_in_} features")
        X = scaler_X.transform(X_df)
        y = scaler_y.fit_transform(y_df.values.reshape(-1, 1)).flatten()
        years = X_df.index.astype(int)

        return X, y, years, X_df, scaler_X, scaler_y, sectors, macro_rates, events, max(years), y_df, expected_features, df

    except Exception as e:
        error_log.append(f"Erreur lors du chargement du fichier : {str(e)}")
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        raise

# File uploader
uploaded_file = st.file_uploader("Upload your updated dataset (CSV, optional)", type=["csv"])
if uploaded_file:
    st.write("### Aperçu du fichier CSV chargé")
    try:
        # Reset file pointer and read for preview
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        st.write(df_preview)
        
        # Allow adding a new row
        if st.button("Ajouter une nouvelle ligne"):
            # Ensure the new year is one more than the maximum year in the data
            year_columns = [col for col in df_preview.columns if col != 'année' and col.isdigit()]
            max_year = max([int(col) for col in year_columns]) if year_columns else 2023
            new_year = max_year + 1
            # Create new row with all columns, initializing new year
            new_row = pd.DataFrame({col: ['produit interieur brut pib' if col == 'année' else 0.0] for col in df_preview.columns})
            if str(new_year) not in df_preview.columns:
                new_row[str(new_year)] = 0.0  # Add new year column
            st.write(f"### Ajouter des données pour l'année {new_year}")
            edited_row = st.data_editor(new_row, num_rows="dynamic")
            
            if st.button("Enregistrer la nouvelle ligne"):
                # Ensure new row has all columns, including new year
                for col in df_preview.columns:
                    if col not in edited_row.columns:
                        edited_row[col] = 0.0
                if str(new_year) not in df_preview.columns:
                    df_preview[str(new_year)] = 0.0  # Add new year column to original data
                df_updated = pd.concat([df_preview, edited_row], ignore_index=True)
                output_file = "updated_VA-2015-2023P.csv"  # Save to updated file
                df_updated.to_csv(output_file, sep=';', index=False, encoding='utf-8')
                st.success(f"Nouvelle ligne enregistrée dans '{output_file}'.")
                # Update uploaded_file to use the new file
                with open(output_file, 'rb') as f:
                    uploaded_file = f
                uploaded_file.seek(0)  # Reset pointer for further processing
    except Exception as e:
        error_log.append(f"Erreur lors de la lecture du fichier uploadé pour l'aperçu : {str(e)}")
        st.error(f"Erreur lors de la lecture du fichier uploadé : {str(e)}")
        st.stop()

# Load data
try:
    X, y, years, X_df, scaler_X, scaler_y, sectors, macro_rates, events, last_year, y_df, expected_features, df = load_and_preprocess(uploaded_file)
except (ValueError, FileNotFoundError, KeyError) as e:
    st.error(str(e))
    st.stop()

st.write(f"**Dernière année disponible dans les données :** {last_year}")
st.write(f"**Nombre de features dans X_df :** {X_df.shape[1]} (attendu : {len(expected_features)})")

# Cache model structure
@st.cache_resource(show_spinner=False)
def get_model_structure(model_type):
    if model_type == "Ridge":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('ridge', Ridge())
        ])
    elif model_type == "ElasticNet":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('elasticnet', ElasticNet())
        ])
    elif model_type == "Huber":
        return Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('huber', HuberRegressor(max_iter=1000))
        ])

# Define models
loo = LeaveOneOut()

ridge_params = {
    'ridge__alpha': np.logspace(-2, 3, 50),
    'feature_selection__k': [5, 10, 15, 20]
}
ridge_cv = RandomizedSearchCV(get_model_structure("Ridge"), ridge_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)

elasticnet_params = {
    'elasticnet__alpha': np.logspace(-2, 3, 50),
    'elasticnet__l1_ratio': np.linspace(0.1, 0.9, 9),
    'feature_selection__k': [5, 10, 15, 20]
}
elasticnet_cv = RandomizedSearchCV(get_model_structure("ElasticNet"), elasticnet_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)

huber_params = {
    'huber__epsilon': np.linspace(1.1, 2.0, 10),
    'huber__alpha': np.logspace(-4, 1, 20),
    'feature_selection__k': [5, 10, 15, 20]
}
huber_cv = RandomizedSearchCV(get_model_structure("Huber"), huber_params, cv=loo, scoring='neg_mean_absolute_error', n_iter=20, random_state=42)

# Evaluation and interpretation function
def interpret_results(model_name, train_mae, test_mae, train_r2, test_r2):
    rel_error = test_mae / np.mean(scaler_y.inverse_transform(y.reshape(-1, 1)))
    st.markdown("#### 💡 Interprétation")
    st.write(f"**R² sur test :** {test_r2:.4f} — indique la qualité de généralisation.")
    st.write(f"**MAE absolue :** {test_mae:.0f} — pour un PIB moyen ~{np.mean(scaler_y.inverse_transform(y.reshape(-1, 1))):,.0f}, soit une erreur relative d’environ **{rel_error*100:.1f}%**.")
    diff_r2 = train_r2 - test_r2
    if diff_r2 > 0.15:
        st.error("⚠️ Écart important entre R² train et test → possible surapprentissage.")
    else:
        st.success("✅ Pas de signe évident de surapprentissage.")

    st.markdown("#### ✅ Conclusion")
    if test_r2 >= 0.96 and rel_error < 0.03:
        st.write(f"✔️ **{model_name} donne d’excellents résultats.**")
        st.write("- Peut être utilisé comme benchmark.")
        st.write("- Très fiable pour un usage en prévision du PIB.")
    elif test_r2 >= 0.90:
        st.write(f"✔️ **{model_name} est un bon modèle,** mais peut être amélioré.")
    else:
        st.write(f"❌ **{model_name} montre des limites.** Envisage une autre méthode ou un tuning plus poussé.")

def evaluate_model(model_cv, X, y, model_name):
    model_cv.fit(X, y)
    train_pred = model_cv.predict(X)
    train_pred_unscaled = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    y_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    train_mae = mean_absolute_error(y_unscaled, train_pred_unscaled)
    train_r2 = r2_score(y_unscaled, train_pred_unscaled)

    preds_test = []
    for tr, te in loo.split(X):
        best_model = model_cv.best_estimator_
        best_model.fit(X[tr], y[tr])
        preds_test.append(best_model.predict(X[te])[0])

    test_pred_unscaled = scaler_y.inverse_transform(np.array(preds_test).reshape(-1, 1)).flatten()
    test_mae = mean_absolute_error(y_unscaled, test_pred_unscaled)
    test_r2 = r2_score(y_unscaled, test_pred_unscaled)

    st.markdown(f"### 🔍 Résultats pour **{model_name}**")
    st.write(f"Train MAE: {train_mae:.2f}, Test MAE (LeaveOneOut): {test_mae:.2f}")
    st.write(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    st.write(f"Meilleurs hyperparamètres : {model_cv.best_params_}")

    interpret_results(model_name, train_mae, test_mae, train_r2, test_r2)
    return test_mae, test_r2, model_cv

# Run models
st.header("📊 Diagnostic et interprétation des modèles")
results = []
models = {}
test_maes = {}

# Check if models need to be trained
train_models = True
if "last_input" in st.session_state and st.session_state.last_input == uploaded_file:
    # If no new file is uploaded, check if models are already trained
    if "trained_models" in st.session_state and "test_maes" in st.session_state:
        models = st.session_state.trained_models
        test_maes = st.session_state.test_maes
        results = st.session_state.results
        train_models = False
        st.write("Utilisation des résultats des modèles déjà entraînés.")

# Train models if needed
if train_models:
    for model, name in [(ridge_cv, "Ridge"), (elasticnet_cv, "ElasticNet"), (huber_cv, "Huber")]:
        with st.spinner(f"Training {name}..."):
            mae, r2, trained_model = evaluate_model(model, X, y, name)
            results.append({
                'Modèle': name,
                'CV MAE': mae,
                'Train R²': r2_score(scaler_y.inverse_transform(y.reshape(-1, 1)), scaler_y.inverse_transform(model.predict(X).reshape(-1, 1)))
            })
            models[name] = trained_model
            test_maes[name] = mae
    # Cache results in session state
    st.session_state.trained_models = models
    st.session_state.test_maes = test_maes
    st.session_state.results = results
    st.session_state.last_input = uploaded_file

# Check if test_maes is empty
if not test_maes:
    st.error("Aucun modèle n'a été entraîné. Veuillez vérifier les données d'entrée ou réinitialiser la session.")
    st.stop()

# Select best model
best_model_name = min(test_maes, key=test_maes.get)
best_model = models[best_model_name].best_estimator_
st.markdown(f"### 🏆 Modèle sélectionné : **{best_model_name}**")
st.write(f"Le modèle **{best_model_name}** a été choisi car il a le MAE le plus bas : {test_maes[best_model_name]:.2f}")

# Vérification du modèle sélectionné
st.header("🔎 Vérification du modèle sélectionné")
st.markdown("#### 1. Vérification de l'intégrité des données")
if X_df.isna().any().any():
    error_log.append("Valeurs manquantes détectées dans X_df.")
    st.error("Valeurs manquantes dans les données d'entrée. Remplacement par 0.")
    X_df = X_df.fillna(0)
if y_df.isna().any().any():
    error_log.append("Valeurs manquantes détectées dans y_df.")
    st.warning("Valeurs manquantes dans les données cibles. Remplacement par la moyenne.")
    y_df = y_df.fillna(y_df.mean())
if y_df.empty or y_df.shape[0] == 0:
    error_log.append("y_df est vide ou n'a aucune ligne.")
    st.error("Les données cibles (y_df) sont vides. Arrêt du programme.")
    st.stop()
st.success("Aucune valeur manquante dans les données après prétraitement. y_df shape: {}".format(y_df.shape))

st.markdown("#### 2. Vérification sur un ensemble de test")
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_unscaled = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_mae = mean_absolute_error(y_test_unscaled, y_pred_test_unscaled)
test_r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
st.write(f"MAE sur l'ensemble de test : {test_mae:.2f}")
st.write(f"R² sur l'ensemble de test : {test_r2:.4f}")
if test_mae > 1.5 * test_maes[best_model_name]:
    error_log.append(f"MAE sur l'ensemble de test ({test_mae:.2f}) significativement plus élevé que le MAE CV ({test_maes[best_model_name]:.2f}).")
    st.warning("Performance sur l'ensemble de test moins bonne que prévue.")

st.markdown("#### 3. Analyse des résidus")
residuals = y_test_unscaled - y_pred_test_unscaled
fig_residuals = px.scatter(x=years[train_size:], y=residuals, title="Résidus sur l'ensemble de test",
                           labels={'x': 'Année', 'y': 'Résidus (million TND)'}, color_discrete_sequence=['#FF6B6B'])
fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
st.plotly_chart(fig_residuals)
if np.abs(residuals).mean() > test_maes[best_model_name]:
    error_log.append(f"Les résidus moyens ({np.abs(residuals).mean():.2f}) sont élevés par rapport au MAE CV ({test_maes[best_model_name]:.2f}).")
    st.warning("Les résidus montrent une erreur moyenne élevée, indiquant une possible sous-performance.")

st.markdown("#### 4. Intervalles de prédiction")
n_bootstraps = 100
bootstrap_preds = []
for _ in range(n_bootstraps):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    best_model.fit(X_train[indices], y_train[indices])
    pred = best_model.predict(X_test)
    bootstrap_preds.append(scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten())
bootstrap_preds = np.array(bootstrap_preds)
lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
st.write("Intervalles de prédiction à 95% pour l'ensemble de test :")
for i, (lower, upper, actual) in enumerate(zip(lower_bound, upper_bound, y_test_unscaled)):
    st.write(f"Année {years[train_size+i]}: Prédit = {y_pred_test_unscaled[i]:,.0f}, Intervalle = [{lower:,.0f}, {upper:,.0f}], Réel = {actual:,.0f}")

# Prediction for 2024
if st.button("🔮 Prédire le PIB pour 2024"):
    with st.spinner("Training and predicting..."):
        target_year = last_year + 1
        historical_df = pd.DataFrame({'Année': years, 'PIB': scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()})
        pred_df = pd.DataFrame({'Année': [target_year], 'PIB': [0.0]})
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)

        feature_vector = pd.DataFrame(index=[0], columns=expected_features).fillna(0.0)
        base_year_data = X_df.loc[last_year] if last_year in X_df.index else X_df.iloc[-3:].mean()

        recent_data = X_df[expected_features].tail(3)
        growth_rates = {}
        for col in sectors + macro_rates:
            if col in recent_data.columns:
                year_growth = recent_data[col].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                growth_rates[col] = year_growth.mean() * 100 if not year_growth.empty else 0.0
            else:
                growth_rates[col] = 0.0
                error_log.append(f"Taux de croissance pour '{col}' non calculé (colonne absente). Utilisation de 0.")
        
        for event in events:
            if event in recent_data.columns:
                growth_rates[event] = recent_data[event].mean() if not recent_data[event].empty else 0
            else:
                growth_rates[event] = 0
                error_log.append(f"Valeur pour '{event}' non trouvée. Utilisation de 0.")

        for sector in sectors:
            try:
                if sector not in X_df.columns:
                    error_log.append(f"Erreur pour {sector} ({target_year}): non trouvé dans X_df. Utilisation de 0.")
                    feature_vector[sector] = 0.0
                else:
                    feature_vector[sector] = base_year_data[sector] * (1 + growth_rates[sector] / 100)
            except Exception as e:
                error_log.append(f"Erreur pour {sector} ({target_year}): {str(e)}. Utilisation de 0.")
                feature_vector[sector] = 0.0

        for rate in macro_rates:
            try:
                if rate not in X_df.columns:
                    error_log.append(f"Erreur pour {rate} ({target_year}): non trouvé dans X_df. Utilisation de 0.")
                    feature_vector[rate] = 0.0
                else:
                    feature_vector[rate] = base_year_data[rate] * (1 + growth_rates[rate] / 100)
            except Exception as e:
                error_log.append(f"Erreur pour {rate} ({target_year}): {str(e)}. Utilisation de 0.")
                feature_vector[rate] = 0.0

        for event in events:
            try:
                if event in X_df.columns:
                    feature_vector[event] = growth_rates[event]
                else:
                    error_log.append(f"Erreur pour {event} ({target_year}): non trouvé dans X_df. Utilisation de 0.")
                    feature_vector[event] = 0
            except Exception as e:
                error_log.append(f"Erreur pour {event} ({target_year}): {str(e)}. Utilisation de 0.")
                feature_vector[event] = 0

        for col in expected_features:
            if col not in sectors + macro_rates + events:
                if col.endswith('_lag1'):
                    base_col = col.replace('_lag1', '')
                    if base_col in feature_vector.columns:
                        feature_vector[col] = base_year_data.get(base_col, X_df[base_col].mean() if base_col in X_df.columns else 0.0)
                    else:
                        feature_vector[col] = base_year_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Feature décalée '{col}' pour {target_year} définie à 0 (données manquantes).")
                else:
                    feature_vector[col] = base_year_data.get(col, X_df[col].mean() if col in X_df.columns else 0.0)
                    if feature_vector[col].iloc[0] == 0.0:
                        error_log.append(f"Feature '{col}' pour {target_year} définie à 0 (données manquantes).")

        if feature_vector.isna().any().any():
            error_log.append(f"Valeurs NaN pour {target_year} : {feature_vector.columns[feature_vector.isna().any()].tolist()}. Remplacement par 0.")
            feature_vector = feature_vector.fillna(0.0)

        feature_vector = feature_vector[expected_features]
        error_log.append(f"Feature vector pour {target_year} : {list(feature_vector.columns)} (nombre: {len(feature_vector.columns)})")
        X_new = scaler_X.transform(feature_vector)

        predicted_gdp = float(scaler_y.inverse_transform(best_model.predict(X_new).reshape(-1, 1))[0])
        combined_df.loc[combined_df['Année'] == target_year, 'PIB'] = predicted_gdp

        st.markdown("### 📈 Résultat de la prédiction")
        st.write(f"**Modèle utilisé :** {best_model_name}")
        st.write(f"**PIB prédit pour {target_year} :** {predicted_gdp:,.0f} million TND")

        fig = px.line(combined_df, x='Année', y='PIB', title=f'PIB historique et prédiction pour {target_year} (basée sur {best_model_name})',
                      markers=True, color_discrete_sequence=['#45B7D1'])
        fig.add_scatter(x=[target_year], y=[predicted_gdp], mode='markers', marker=dict(size=10, color='red'), name=f'Prédiction {target_year}')
        st.plotly_chart(fig)

        st.markdown("### 🧠 Explication des prédictions avec SHAP")
        st.write(f"Les graphiques suivants expliquent comment chaque feature contribue à la prédiction du PIB pour {target_year}.")

        best_model.fit(X, y)
        feature_vector_for_shap = X_new
        error_log.append(f"Shape de feature_vector_for_shap : {feature_vector_for_shap.shape}")
        background_data = scaler_X.transform(X_df[expected_features])
        error_log.append(f"Shape de background_data : {background_data.shape}")

        try:
            if best_model_name in ["Ridge", "ElasticNet"]:
                explainer = shap.LinearExplainer(
                    best_model,
                    background_data,
                    feature_names=expected_features
                )
            else:  # Huber
                explainer = shap.KernelExplainer(
                    best_model.predict,
                    background_data,
                    feature_names=expected_features
                )

            shap_values = explainer.shap_values(feature_vector_for_shap)
            error_log.append(f"Shape de shap_values : {np.array(shap_values).shape}")

            st.markdown("#### 📊 Importance globale des features (Summary Plot)")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, feature_vector_for_shap, feature_names=expected_features, show=False)
            st.pyplot(plt)
            plt.close()

            st.markdown("#### 📉 Dependence Plot pour gdp_lag1")
            plt.figure(figsize=(10, 6))
            shap.dependence_plot("gdp_lag1", shap_values, feature_vector_for_shap, feature_names=expected_features, show=False)
            st.pyplot(plt)
            plt.close()

            st.markdown(f"#### 📈 Contribution des features pour {target_year} (Force Plot)")
            plt.figure(figsize=(10, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                feature_vector_for_shap[0],
                feature_names=expected_features,
                matplotlib=True,
                show=False
            )
            st.pyplot(plt)
            plt.close()

            st.markdown(f"#### 📊 Importance des features pour {target_year}")
            plt.figure(figsize=(10, 6))
            shap.bar_plot(shap_values[0], feature_names=expected_features, max_display=10, show=False)
            st.pyplot(plt)
            plt.close()

        except Exception as e:
            error_log.append(f"Erreur lors du calcul SHAP : {str(e)}")
            st.error(f"Impossible de générer les explications SHAP : {str(e)}. Veuillez vérifier les données.")

        st.markdown("#### 📈 PIB historique vs prédictions")
        y_pred_historical = best_model.predict(X)
        y_pred_historical_unscaled = scaler_y.inverse_transform(y_pred_historical.reshape(-1, 1)).flatten()
        y_historical_unscaled = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
        historical_df = pd.DataFrame({
            'Année': years,
            'PIB Réel': y_historical_unscaled,
            'PIB Prédit': y_pred_historical_unscaled
        })
        pred_df = pd.DataFrame({
            'Année': [target_year],
            'PIB Réel': [np.nan],
            'PIB Prédit': [predicted_gdp]
        })
        combined_df = pd.concat([historical_df, pred_df], ignore_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df['Année'], y=combined_df['PIB Réel'], mode='lines+markers', name='PIB Réel', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=combined_df['Année'], y=combined_df['PIB Prédit'], mode='lines+markers', name='PIB Prédit', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'PIB Historique vs Prédictions (incl. {target_year})', xaxis_title='Année', yaxis_title='PIB (million TND)')
        st.plotly_chart(fig)

        st.info(f"🧪 Prédiction basée sur le modèle {best_model_name} avec le MAE le plus bas, utilisant les tendances historiques extrapolées à partir des 3 dernières années.")

        show_errors = st.checkbox("Afficher le journal", value=True)
        if show_errors and error_log:
            st.markdown("### Journal informatif")
            for error in error_log:
                st.write(error)
