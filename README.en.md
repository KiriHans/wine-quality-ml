# Wine Quality Classification using Decision Trees and Naive Bayes

## Project Overview

This project analyzes physicochemical properties of wines to predict whether a wine is high quality or standard quality using machine learning classification models.

The analysis follows a complete machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data quality verification
* Feature analysis and correlation study
* Binary target transformation
* Model training
* Hyperparameter tuning
* Model comparison
* Error analysis
* ROC and AUC evaluation

Two main models are evaluated:

* Decision Tree Classifier
* Gaussian Naive Bayes

The objective is to determine which model is most suitable for a wine quality control system in a winery.

## Dataset

The dataset used in this project comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

The dataset contains physicochemical measurements of red wine samples from northern Portugal.

### Dataset characteristics

* 6497 observations
* 11 predictive variables
* 1 target variable (`quality`)

Predictor variables include:

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

The original quality score ranges from 0 to 10.

For this project, the problem is converted into a binary classification problem:

| Class | Meaning          |
| ----- | ---------------- |
| 0     | Standard Quality |
| 1     | High Quality     |

High-quality wines are defined as:

```
quality >= 7
```

## Data Loading

The dataset is retrieved programmatically using the Python package `ucimlrepo`, ensuring reproducibility.

## Exploratory Data Analysis

The exploratory analysis focuses on evaluating data quality and statistical properties.

The following aspects were examined:

### Data structure

* All predictor variables are numeric
* No missing values were found
* The target variable (`quality`) is categorical with integer values

### Completeness

The dataset contains no missing values, so no imputation is required.

### Consistency

The `quality` variable contains values within the expected range defined in the dataset documentation.

### Distribution analysis

Histograms and density plots were used to study the distribution of each feature.

Several variables exhibit positive skewness, including:

* volatile_acidity
* residual_sugar
* total_sulfur_dioxide

## Class Imbalance

After transforming the target variable into binary classification, the dataset shows class imbalance:

* 80% Standard Quality
* 20% High Quality

This imbalance makes accuracy an insufficient evaluation metric, so the analysis focuses on:

* Precision
* Recall
* F1-score
* ROC-AUC

## Correlation Analysis

A correlation matrix was used to identify relationships between variables.

Variables most correlated with wine quality include:

* Alcohol
* Density
* Volatile acidity
* Chlorides

Alcohol content showed the strongest positive relationship with wine quality.

## Data Preprocessing

The preprocessing stage includes:

* Binary transformation of the target variable
* Train-test split
* Feature scaling for Naive Bayes
* Handling class imbalance using:

```
class_weight = 'balanced'
```

for the Decision Tree models.


## Machine Learning Models

### Decision Tree Classifier

Decision Trees are interpretable (white-box) models that classify samples using hierarchical decision rules.

Three configurations were explored:

#### 1. Baseline Decision Tree

A shallow tree with:

```
max_depth = 3
```

to understand the most important decision rules.

#### 2. Pre-Pruned Decision Tree

Hyperparameters were optimized by analyzing performance across different values of:

* `max_depth`
* `min_samples_leaf`

The best configuration found were:

```
max_depth = 5
min_samples_leaf = 60
```

#### 3. Post-Pruned Decision Tree

Cost-complexity pruning was applied using the parameter:

```
ccp_alpha = 0.00238
```

This simplified the tree while maintaining strong predictive performance.

### Gaussian Naive Bayes

Naive Bayes is a probabilistic classifier based on the Bayes Theorem.

Because the predictors are continuous variables, the Gaussian Naive Bayes variant was used.

This model assumes that:

$P(x_j \mid C = c_k) \sim \mathcal{N}(\mu_{jk}, \sigma^2_{jk})$


where each feature follows a normal distribution within each class.

Although computationally efficient, the model assumes conditional independence between features.

## Model Evaluation

Due to class imbalance, performance was evaluated using:

* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve
* AUC

## Performance Comparison

### Decision Tree (Pre-Pruning)

Precision: 0.398
Recall: 0.796
F1-score: 0.531

### Decision Tree (Post-Pruning)

Precision: 0.402
Recall: 0.804
F1-score: 0.536

### Gaussian Naive Bayes

Precision: 0.423
Recall: 0.643
F1-score: 0.510

Decision Trees achieved higher recall, meaning they detect more high-quality wines.


## ROC Curve and AUC

The ROC curve evaluates the trade-off between true positive rate and false positive rate.

AUC results:

| Model                        | AUC       |
| ---------------------------- | --------- |
| Decision Tree (Pre-Pruning)  | 0.751     |
| Decision Tree (Post-Pruning) | 0.756     |
| Gaussian Naive Bayes         | 0.714     |

Decision Trees show better class discrimination capability.

## Error Analysis

Two classification errors were considered:

**False Positive**: Standard wine classified as high quality.

**False Negative**: High-quality wine classified as standard.

In a winery context, false positives are more costly, since they may result in selling lower-quality wine as premium, potentially harming the brand’s reputation.

## Final Recommendation

Based on the previous analysis, the recommended model chosen was the **Decision Tree with Post-Pruning**.
This is because it has the highest F1-score, best AUC performance, it has a transparent decision rules and it has robust performance despite the features correlations.


## Repository Structure

```
wine-quality-ml/
│
├── analisis_de_calidad_del_vino.ipynb
│
├── images/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
└── requirements.txt
```

## Dependency Installation and Project Execution

To run the notebook for this project, it is necessary to install the libraries used for data analysis, visualization, and machine learning models.

### Option 1: Using `pip` and a virtual environment (recommended)

Creating a virtual environment helps avoid dependency conflicts.

On **Windows**:

```bash
python -m venv wine-quality-ml
wine-quality-ml\Scripts\activate
````

On **Linux / macOS**:

```bash
python3 -m venv wine-quality-ml
source wine-quality-ml/bin/activate
```

Once the virtual environment is activated, install the required libraries:

```bash
pip install -r requirements.txt
```

### Option 2: Using `conda`

If you use Anaconda or Miniconda, you can create a dedicated environment for the project:

```bash
conda create -n wine-quality-ml python=3.14
conda activate wine-quality-ml
```

Install the main libraries:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook

After installing the dependencies, start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the file:

```
analisis_de_calidad_del_vino.ipynb
```

and run the cells in order to reproduce the full analysis.
