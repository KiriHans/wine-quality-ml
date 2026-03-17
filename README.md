# Clasificación de Calidad de Vinos usando Árboles de Decisión y Naive Bayes

## Descripción del Proyecto

Este proyecto analiza las propiedades fisicoquímicas de los vinos para predecir si un vino es de alta calidad o de calidad estándar utilizando modelos de clasificación de aprendizaje automático.

El análisis sigue un flujo completo de trabajo en machine learning, que incluye:

* Análisis Exploratorio de Datos (EDA)
* Verificación de la calidad de los datos
* Análisis de variables y estudio de correlaciones
* Transformación binaria de la variable objetivo
* Entrenamiento de modelos
* Ajuste de hiperparámetros
* Comparación de modelos
* Análisis de errores
* Evaluación mediante ROC y AUC

Se evalúan dos modelos principales:

* Árbol de Decisión
* Gaussian Naive Bayes

El objetivo es determinar qué modelo es más adecuado para un sistema de control de calidad del vino en una bodega.

## Conjunto de Datos

El conjunto de datos utilizado en este proyecto proviene del [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

El dataset contiene mediciones fisicoquímicas de muestras de vino tinto provenientes del norte de Portugal.

### Características del dataset

* 6497 observaciones
* 11 variables predictoras
* 1 variable objetivo (`quality`)

Las variables predictoras incluyen:

* fixed_acidity
* volatile_acidity
* citric_acid
* residual_sugar
* chlorides
* free_sulfur_dioxide
* total_sulfur_dioxide
* density
* pH
* sulphates
* alcohol

La puntuación original de calidad varía entre 0 y 10.

Para este proyecto, el problema se convierte en un problema de clasificación binaria:

| Clase | Significado          |
| ----- | -------------------- |
| 0     | Calidad Estándar     |
| 1     | Alta Calidad         |

Los vinos de alta calidad se definen como:

```

quality >= 7

```

## Carga de Datos

El dataset se obtiene de forma programática utilizando el paquete de Python `ucimlrepo`, garantizando la reproducibilidad.

## Análisis Exploratorio de Datos

El análisis exploratorio se centra en evaluar la calidad de los datos y sus propiedades estadísticas.

Se examinaron los siguientes aspectos:

### Estructura de los datos

* Todas las variables predictoras son numéricas
* No se encontraron valores faltantes
* La variable objetivo (`quality`) es categórica con valores enteros

### Completitud

El dataset no contiene valores faltantes, por lo que no se requiere imputación.

### Consistencia

La variable `quality` contiene valores dentro del rango esperado definido en la documentación del dataset.

### Análisis de distribución

Se utilizaron histogramas y gráficos de densidad para estudiar la distribución de cada variable.

Varias variables presentan asimetría positiva, incluyendo:

* volatile_acidity
* residual_sugar
* total_sulfur_dioxide

## Desbalance de Clases

Después de transformar la variable objetivo en una clasificación binaria, el dataset muestra desbalance de clases:

* 80% Calidad Estándar
* 20% Alta Calidad

Este desbalance hace que la exactitud (accuracy) sea una métrica insuficiente para la evaluación, por lo que el análisis se centra en:

* Precision
* Recall
* F1-score
* ROC-AUC

## Análisis de Correlación

Se utilizó una matriz de correlación para identificar relaciones entre variables.

Las variables más correlacionadas con la calidad del vino incluyen:

* Alcohol
* Density
* Volatile acidity
* Chlorides

El contenido de alcohol mostró la relación positiva más fuerte con la calidad del vino.

## Preprocesamiento de Datos

La etapa de preprocesamiento incluye:

* Transformación binaria de la variable objetivo
* División en conjuntos de entrenamiento y prueba
* Escalado de características para Naive Bayes
* Manejo del desbalance de clases utilizando:

```

class_weight = 'balanced'

```

para los modelos de Árbol de Decisión.


## Modelos de Machine Learning

### Clasificador Árbol de Decisión

Los Árboles de Decisión son modelos interpretables (caja blanca) que clasifican muestras utilizando reglas de decisión jerárquicas.

Se exploraron tres configuraciones:

#### 1. Árbol de Decisión Base

Un árbol poco profundo con:

```

max_depth = 3

```

para comprender las reglas de decisión más importantes.

#### 2. Árbol de Decisión con Pre-Poda

Los hiperparámetros fueron optimizados analizando el rendimiento para diferentes valores de:

* `max_depth`
* `min_samples_leaf`

La mejor configuración encontrada fue:

```

max_depth = 5
min_samples_leaf = 60

```

#### 3. Árbol de Decisión con Post-Poda

Se aplicó poda por complejidad de costo utilizando el parámetro:

```

ccp_alpha = 0.00238

```

Esto simplificó el árbol manteniendo un buen rendimiento predictivo.

### Gaussian Naive Bayes

Naive Bayes es un clasificador probabilístico basado en el Teorema de Bayes.

Debido a que las variables predictoras son continuas, se utilizó la variante Gaussian Naive Bayes.

Este modelo asume que:

$P(x_j \mid C = c_k) \sim \mathcal{N}(\mu_{jk}, \sigma^2_{jk})$


donde cada variable sigue una distribución normal dentro de cada clase.

Aunque es computacionalmente eficiente, el modelo asume independencia condicional entre las variables.

## Evaluación de Modelos

Debido al desbalance de clases, el rendimiento se evaluó utilizando:

* Precision
* Recall
* F1-score
* Matriz de confusión
* Curva ROC
* AUC

## Comparación de Rendimiento

### Árbol de Decisión (Pre-Poda)

Precision: 0.398
Recall: 0.796
F1-score: 0.531

### Árbol de Decisión (Post-Poda)

Precision: 0.402
Recall: 0.804
F1-score: 0.536

### Gaussian Naive Bayes

Precision: 0.423
Recall: 0.643
F1-score: 0.510

Los Árboles de Decisión lograron un mayor recall, lo que significa que detectan más vinos de alta calidad.


## Curva ROC y AUC

La curva ROC evalúa el equilibrio entre la tasa de verdaderos positivos y la tasa de falsos positivos.

Resultados de AUC:

| Modelo                       | AUC       |
| ---------------------------- | --------- |
| Árbol de Decisión (Pre-Poda) | 0.751     |
| Árbol de Decisión (Post-Poda)| 0.756     |
| Gaussian Naive Bayes         | 0.714     |

Los Árboles de Decisión muestran una mejor capacidad de discriminación entre clases.

## Análisis de Errores

Se consideraron dos tipos de errores de clasificación:

**Falso Positivo**: Vino estándar clasificado como de alta calidad.

**Falso Negativo**: Vino de alta calidad clasificado como estándar.

En el contexto de una bodega, los falsos positivos son más costosos, ya que pueden resultar en vender vinos de menor calidad como premium, lo que podría dañar la reputación de la marca.

## Recomendación Final

Basado en el análisis previo, el modelo recomendado es el **Árbol de Decisión con Post-Poda**.
Esto se debe a que tiene el mayor F1-score, el mejor rendimiento en AUC, posee reglas de decisión transparentes y mantiene un rendimiento robusto a pesar de las correlaciones entre las variables.


## Estructura del Repositorio

```

wine-quality-ml/
│
├── caso_de_estudio.ipynb
│
├── images/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
└── requirements.txt

```


## Instalación de Dependencias y Ejecución del Proyecto

Para ejecutar el notebook de este proyecto es necesario instalar las librerías utilizadas para el análisis de datos, visualización y modelos de aprendizaje automático.

### Opción 1: Usando `pip` y un entorno virtual (recomendado)

Crear un entorno virtual ayuda a evitar conflictos entre dependencias.

En **Windows**:

```bash
python -m venv wine-quality-ml
wine-quality-ml\Scripts\activate
```

En **Linux / macOS**:

```bash
python3 -m venv wine-quality-ml
source wine-quality-ml/bin/activate
````
Una vez activado el entorno virtual, instalar las librerías necesarias:

```bash
pip install -r requirements.txt
```

### Opción 2: Usando `conda`

Si utilizas Anaconda o Miniconda, puedes crear un entorno dedicado para el proyecto:

```bash
conda create -n wine-quality-ml python=3.14
conda activate wine-quality-ml
```

Instalar las librerías principales:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Ejecutar el Notebook

Después de instalar las dependencias, iniciar Jupyter Notebook:

```bash
jupyter notebook
```

Luego abrir el archivo:

```
analisis_de_calidad_del_vino.ipynb
```

y ejecutar las celdas en orden para reproducir todo el análisis.

