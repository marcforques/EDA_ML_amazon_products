# Amazon Products Sales — EDA y Machine Learning

Proyecto completo de ciencia de datos sobre **42.675 productos de Amazon**: comprensión del dataset,
limpieza, análisis exploratorio y un modelo de clasificación que predice el **volumen de ventas** de un
producto a partir de sus características (precio, descuentos, reseñas, envío…).

**Stack:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Joblib · Jupyter

## Resultado principal

| Modelo | Accuracy | F1 macro | F1 ponderado |
|---|---|---|---|
| Baseline (`DummyClassifier`, clase mayoritaria) | 46,7 % | 0,159 | 0,298 |
| Gradient Boosting (con pesos por clase) | 83,9 % | 0,789 | 0,851 |
| Random Forest | 92,1 % | 0,858 | 0,922 |
| **Random Forest optimizado** (`RandomizedSearchCV`) | **92,4 %** | **0,867** | **0,924** |

*Conjunto de test: 6.433 productos (20 %, partición estratificada).*

Por clase, el modelo final acierta muy bien en `low`, `medium` y `very_high` (F1 de 0,95, 0,92 y 0,97).
La clase **`high` es la más difícil** (F1 0,63): solo es el 4,4 % de los datos y se confunde con `medium`.

## Pregunta de negocio

> ¿Qué características de un producto de Amazon están relacionadas con un mayor volumen de compras recientes?

La variable objetivo `sales_category` se construye a partir de `bought_in_last_month`:

| Clase | Compras en el último mes | Productos |
|---|---|---|
| `low` | ≤ 100 | 14.768 |
| `medium` | 101 – 1.000 | 15.032 |
| `high` | 1.001 – 10.000 | 1.430 |
| `very_high` | > 10.000 | 934 |

Los 10.511 productos sin dato de ventas se excluyen del modelado, que queda con **32.164 productos**.
Las clases están muy **desbalanceadas**, por eso se compara siempre con un baseline y se usan
F1 macro y pesos por clase (`class_weight="balanced_subsample"`).

## Datos

El CSV original no se incluye en el repositorio (está en `.gitignore`, igual que el modelo entrenado).
<!-- TODO: añadir aquí el enlace a la fuente original del dataset -->

1. Descarga el dataset de productos de Amazon (16 columnas) y guárdalo como
   `data/raw/amazon_products_sales.csv`.
2. Genera el dataset limpio y entrena el modelo con los comandos de [Cómo ejecutarlo](#cómo-ejecutarlo).

## Fases del proyecto

### 1. Comprensión del dataset — `01_dataset_understanding.ipynb`
Estructura, tipos, nulos y duplicados. Muchas columnas numéricas o booleanas llegan como texto
(por ejemplo, `"100+ bought in past month"`), así que hay que transformarlas.

### 2. Limpieza — `02_data_cleaning.ipynb` y `scripts/make_dataset.py`
- Conversión a número o booleano de `rating`, `number_of_reviews`, `bought_in_last_month`, precios,
  `is_sponsored`, `is_couponed`, `is_best_seller` y `delivery_details`.
- Variables nuevas: `discount_amount`, `discount_percentage`, `coupon_discount_value`,
  `coupon_discount_type`, `has_free_delivery`, `has_fastest_delivery`, `title_length` y `title_word_count`.

### 3. Análisis exploratorio — `03_eda.ipynb`
- `bought_in_last_month` tiene una distribución muy asimétrica: la mayoría de productos venden poco.
- Los productos que más venden suelen tener **más reseñas, mejor valoración y precios más bajos**.
- Que el producto esté **patrocinado** (`is_sponsored`) aparece relacionado con las ventas.

### 4. Modelado — `04_model_training.ipynb`
Clasificación multiclase con 17 variables (precio, descuentos, cupón, valoración, reseñas,
patrocinado, best seller, envío, longitud del título…). Se comparan un baseline, Random Forest y
Gradient Boosting, se analiza la importancia de las variables y se optimiza el Random Forest con
`RandomizedSearchCV` (validación cruzada de 3 particiones).

Mejores hiperparámetros: `n_estimators=500`, `max_depth=40`, `min_samples_split=5`,
`min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced_subsample"`.

## Estructura

```text
EDA_ML_amazon_products/
├── data/
│   ├── raw/                  # amazon_products_sales.csv (no incluido)
│   └── processed/            # amazon_products_clean.csv (se genera)
├── models/                   # optimized_random_forest.pkl y features.pkl (se generan)
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_model_training.ipynb
├── scripts/make_dataset.py   # raw → processed
├── src/
│   ├── config.py             # rutas y semilla
│   ├── data_loader.py
│   ├── cleaning.py
│   ├── features.py           # clases de ventas y lista de variables
│   ├── modeling.py
│   └── evaluation.py
├── main.py                   # entrena y guarda el modelo final
└── requirements.txt
```

## Cómo ejecutarlo

```bash
python -m venv .venv
```

Activa el entorno: en Windows `.venv\Scripts\activate`; en Mac/Linux `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
python scripts/make_dataset.py   # limpia data/raw → data/processed
python main.py                   # entrena el Random Forest final, muestra métricas y lo guarda en models/
```

## Limitaciones y siguientes pasos

- La clase `high` sigue siendo la más débil: probar umbrales por clase, sobremuestreo (SMOTE) o
  agrupar `high` y `very_high`.
- `number_of_reviews` refleja ventas acumuladas, así que parte de la señal es "el producto ya vende";
  conviene medir cuánto rinde el modelo sin esa variable.
- Probar modelos de boosting más modernos (LightGBM, XGBoost) y explicar las predicciones con SHAP.
