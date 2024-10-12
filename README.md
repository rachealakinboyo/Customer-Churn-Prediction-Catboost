
# Telco Customer Churn Prediction

## Project Overview

In this project, I aim to predict customer churn using a Telco customer dataset. I explore the dataset, preprocess it, engineer features, and build a machine learning model using **CatBoost** to predict customer churn. The notebook walks through the entire process of loading the data, performing exploratory data analysis (EDA), preprocessing, feature engineering, training a model, and evaluating its performance.

The goal is to identify customers likely to churn so that proactive measures can be taken to retain them.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Data Loading and Exploration](#data-loading-and-exploration)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Results](#results)
9. [Model Deployment](#model-deployment)
10. [License](#license)

## Installation

To run the notebook, follow these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/telco-churn-prediction.git
    ```
   
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   
3. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open the notebook `churn_prediction.ipynb` and run the cells step by step.

## Project Structure

The repository contains the following files:

- `churn_prediction.ipynb`: The main notebook containing the code for data loading, preprocessing, model training, and evaluation.
- `requirements.txt`: The list of Python packages required to run the project.
- `README.md`: Project documentation.

## Data Loading and Exploration

The dataset used is the Telco Customer Churn dataset, which contains information about telecom customers, including their demographics, account information, and churn status. In the notebook, I load the dataset using **pandas** and conduct initial exploratory analysis.

```python
import pandas as pd

# Load the dataset
file_path = '/path-to-dataset/Telco_customer_churn.xlsx'
df = pd.read_excel(file_path)

# View the first few rows of the data
df.head()
```

## Data Preprocessing

Data preprocessing steps include:

- Dropping unnecessary columns
- Handling missing values
- Converting categorical values to numerical (Yes/No → 1/0)
- Scaling numerical features

This is done within the notebook itself.

## Feature Engineering

In this section of the notebook, new features are created, and categorical variables are one-hot encoded. Features such as **log-transformed charges** and **interaction terms** are also added.

```python
df['Log Total Charges'] = np.log1p(df['Total Charges'])
df = pd.get_dummies(df, columns=['Contract', 'Payment Method', 'Internet Service'], drop_first=True)
```

## Exploratory Data Analysis (EDA)

I visualize the distribution of churn across different categories and plot numerical distributions using **seaborn** and **matplotlib**.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Visualizing churn distribution
sns.countplot(x='Churn Label', data=df)
plt.show()
```

## Model Training and Evaluation

I trained a **CatBoost** classifier to predict customer churn. I handled class imbalance using **SMOTE** and evaluated the model using accuracy, precision, recall, and AUC-ROC.

```python
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Train the model
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.005, depth=6)
catboost_model.fit(X_train_smote, y_train_smote, eval_set=(X_test, y_test))
```

## Results

The model's performance is evaluated using the following metrics:

- **Accuracy**: 73%
- **Precision**: 49.53%
- **Recall**: 83.96%
- **AUC-ROC**: 0.85

### Why Recall is Low:
The precision is lower due to the business goal, which prioritizes capturing as many potential churners as possible (high recall). A slightly lower precision is acceptable because it's more important to identify churners even at the cost of false positives. By capturing potential churners, the business can focus on retention strategies for those customers who are likely to leave, helping to mitigate potential revenue losses.

## Model Deployment

The trained model is saved using **joblib** for future deployment.

```python
import joblib

# Save the model
joblib.dump(catboost_model, 'customer_churn_model.cbm')
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

