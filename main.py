from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, MODEL_PATH, FEATURES_PATH
from src.data_loader import load_csv
from src.features import classify_sales, get_model_features
from src.evaluation import evaluate_model
from src.modeling import train_optimized_random_forest

from sklearn.model_selection import train_test_split
import joblib

def main():
    
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe el dataset limpio en {PROCESSED_DATA_PATH}. "
            "Ejecuta primero: python scripts/make_dataset.py"
        )
        
    # Cargamos datos
    df = load_csv(PROCESSED_DATA_PATH)

    # Creamos las categorías de venta
    df["sales_category"] = df["bought_in_last_month"].apply(classify_sales)
    df_model = df.dropna(subset=["sales_category"]).copy()

    # Cargamos features
    features = get_model_features()

    X = df_model[features].fillna(0)
    y = df_model["sales_category"]

    # Separamos en grupos de test y train
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    # Utilizamos los mejores parámetros encontrados en el notebook 04
    best_params = {
        'n_estimators': 500,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'max_depth': 40,
        'class_weight': 'balanced_subsample'
    }

    # Entrenamos el random forest optimizado
    final_model = train_optimized_random_forest(
        X_train,
        y_train,
        best_params=best_params,
        random_state=RANDOM_STATE
    )

    # Evaluamos el modelo
    results = evaluate_model(final_model, X_test, y_test)

    print("Accuracy:", results["accuracy"])
    print("Macro F1:", results["macro_f1"])
    print("Weighted F1:", results["weighted_f1"])
    print(results["classification_report"])

    # Guardamos el modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Features guardadas en: {FEATURES_PATH}")

    
if __name__=="__main__":
    main()