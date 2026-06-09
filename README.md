# Amazon Products Sales - EDA & Machine Learning

Proyecto completo de Data Science aplicado a un dataset de productos de Amazon.
El objetivo principal es realizar un análisis exploratorio de datos, limpiar y transformar variables, y entrenar modelos de Machine Learning capaces de predecir la categoría de ventas de un producto.

## Objetivo del proyecto

El proyecto busca responder a la siguiente pregunta:

> ¿Qué características de un producto de Amazon están relacionadas con un mayor volumen de compras recientes?

Para ello, se trabaja con variables como:

- Rating del producto
- Número de reseñas
- Precio actual
- Precio listado
- Descuentos
- Cupones
- Producto patrocinado
- Best seller
- Información de entrega
- Longitud del título

La variable objetivo utilizada para el modelado es `sales_category`, creada a partir de `bought_in_last_month`.

## Estructura del proyecto

```text
EDA_ML_amazon_products/
│
├── data/
│   ├── raw/
│   │   └── amazon_products_sales.csv
│   └── processed/
│       └── amazon_products_clean.csv
│
├── models/
│   ├── optimized_random_forest.pkl
│   └── features.pkl
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_model_training.ipynb
│
├── scripts/
│   └── make_dataset.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── cleaning.py
│   ├── features.py
│   ├── modeling.py
│   └── evaluation.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Fases del proyecto

### 1. Comprensión inicial del dataset

En el primer notebook se analiza la estructura general del dataset:

- Número de filas y columnas
- Tipos de datos
- Valores nulos
- Duplicados
- Variables disponibles

Se detecta que muchas columnas aparecen inicialmente como texto, aunque contienen información numérica o booleana.

### 2. Limpieza de datos

En la fase de limpieza se transforman columnas como:

- `rating`
- `number_of_reviews`
- `bought_in_last_month`
- `current_discounted_price`
- `listed_price`
- `price_on_variant`
- `is_sponsored`
- `is_couponed`
- `is_best_seller`
- `delivery_details`

También se generan nuevas variables como:

- `discount_amount`
- `discount_percentage`
- `coupon_discount_value`
- `coupon_discount_type`
- `has_free_delivery`
- `has_fastest_delivery`
- `title_length`
- `title_word_count`

La limpieza puede ejecutarse automáticamente con:

python scripts/make_dataset.py

### 3. Análisis exploratorio de datos

Durante el EDA se analizan las relaciones entre ventas recientes y variables como rating, reseñas, precio, descuentos y características comerciales.

Principales conclusiones:

- La variable `bought_in_last_month` presenta una distribución muy asimétrica.
- La mayoría de productos tienen ventas bajas o medias.
- Los productos con mayores ventas suelen tener más reseñas, mejor rating y precios más bajos.
- La variable `is_sponsored` muestra una relación fuerte con las ventas.
- Las clases de ventas están desbalanceadas, especialmente entre `low`, `medium`, `high` y `very_high`.

### 4. Modelado de Machine Learning

El problema se plantea como una tarea de clasificación multiclase.

La variable objetivo es:

```text
sales_category
```

Con las siguientes clases:

```text
low
medium
high
very_high
```

Modelos entrenados:

- Baseline con `DummyClassifier`
- Random Forest
- Gradient Boosting
- Random Forest optimizado con `RandomizedSearchCV`

### 5. Resultados

El modelo baseline obtiene una accuracy aproximada del 46,7 %, prediciendo siempre la clase mayoritaria.

El mejor modelo es el Random Forest optimizado, con resultados aproximados de:

```text
Accuracy: 92,38 %
Macro F1: 0.867
Weighted F1: 0.924
```

El modelo clasifica muy bien las clases `low`, `medium` y `very_high`, aunque la clase `high` sigue siendo la más difícil de separar.

### 6. Modelo final

El modelo final se guarda en:

```text
models/optimized_random_forest.pkl
```

Y la lista de variables utilizadas se guarda en:

```text
models/features.pkl
```

Para entrenar el modelo final desde el dataset procesado:

python main.py


## Ejecución del proyecto

### 1. Crear y activar entorno virtual

python -m venv .venv
.venv\Scripts\activate

### 2. Instalar dependencias

pip install -r requirements.txt

### 3. Generar dataset limpio

python scripts/make_dataset.py

### 4. Entrenar modelo final

python main.py

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Conclusión

Este proyecto demuestra un flujo completo, desde la comprensión y limpieza de datos hasta el entrenamiento, evaluación y guardado de modelos de Machine Learning.

El análisis muestra que variables como el número de reseñas, el rating, el precio actual y si un producto está patrocinado contienen información relevante para predecir la categoría de ventas de productos.

