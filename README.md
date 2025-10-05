# **Prediction of Cancer Death Rate in the US Using Machine Learning Algorithms**

## **Project Overview**

This project focuses on predicting the **cancer death rate in the United States** using various **machine learning algorithms**.
The primary goal is to identify key factors influencing cancer mortality and to develop predictive models that can assist public health authorities and researchers in formulating data-driven prevention and control strategies.

By comparing the performance of multiple algorithms, this study aims to determine which model best captures the complex relationships between demographic, lifestyle, and healthcare variables that contribute to cancer death rates across different regions.

---

## **Data Source**

The dataset used for this analysis was sourced from the **Kaggle** platform and contains county-level cancer statistics across the United States.

* **Dataset name:** Cancer Mortality Data (U.S. Counties)
* **Time period:** 2014–2018
* **Target variable:** `TARGET_DEATHRATE` (Cancer Death Rate per 100,000 people)
* **Key features:** include demographic factors (age, race, gender), socioeconomic indicators (income, education), and health-related metrics (smoking rates, obesity, healthcare access).

A detailed data description is available on the Kaggle page:
👉 [Cancer Mortality Data on Kaggle](https://www.kaggle.com/datasets/varunraskar/cancer-regression) *(insert exact link if available)*

---

## **Tools**

* **Python (Jupyter Notebook)** — primary environment for data analysis and modeling
* **Pandas, NumPy** — data manipulation and preprocessing
* **Matplotlib, Seaborn** — visualizations and exploratory data analysis
* **Scikit-learn** — model training, evaluation, and metrics
* **XGBoost, Random Forest, Linear Regression, Decision Tree** — machine learning algorithms used for prediction

---

## **Data Cleaning and Preparation**

The following preprocessing steps were performed before model training:

1. **Data loading and inspection** — verified data types and examined column names.
2. **Handling missing values** — imputed or dropped missing data as appropriate.
3. **Outlier detection and treatment** — handled extreme values in numerical columns using statistical thresholds.
4. **Feature selection and encoding** — selected relevant predictors and applied label or one-hot encoding for categorical variables.
5. **Data normalization** — scaled numeric features to improve model performance.
6. **Train-test split** — divided data into **80% training** and **20% testing** sets for validation.

---

## **Exploratory Data Analysis (EDA)**

The EDA phase provided insights into the structure and distribution of the dataset:

* **Correlation analysis** identified strong relationships between lifestyle factors (e.g., smoking, obesity) and death rate.
* **Visualization** of cancer death rate by region revealed geographic disparities.
* **Age and median income** were among the most influential predictors of cancer mortality.
* **Heatmaps and scatter plots** were used to visualize multivariate relationships between predictors and the target variable.

---

## **Modeling and Evaluation**

Several machine learning models were implemented and compared for predictive accuracy:

| **Model**                   | **Description**                                        | **Evaluation Metric (R² / RMSE / MAE)** |
| --------------------------- | ------------------------------------------------------ | --------------------------------------- |
| **Linear Regression**       | Baseline model to capture linear relationships         | Good interpretability, lower accuracy   |
| **Decision Tree Regressor** | Nonlinear model to capture feature interactions        | Moderate performance                    |
| **Random Forest Regressor** | Ensemble of decision trees for improved generalization | High accuracy, low overfitting          |
| **XGBoost Regressor**       | Gradient boosting model optimized for performance      | Best accuracy among all models          |

Performance metrics were computed using:

* **R² (Coefficient of Determination)**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Square Error)**

---

## **Results / Findings**

1. **XGBoost** emerged as the **best-performing model**, demonstrating the lowest prediction error and the highest R² value.
2. **Random Forest** also performed strongly, confirming the robustness of tree-based ensemble methods.
3. **Linear Regression** provided baseline interpretability but failed to capture nonlinear relationships effectively.
4. Key predictors influencing cancer death rates included **age, median income, smoking rate, obesity prevalence, and healthcare access**.
5. The findings highlight socioeconomic and lifestyle disparities contributing to regional variations in cancer mortality.

---

## **Recommendations**

* **Policy Intervention:** Focus on regions with high predicted death rates for targeted health campaigns.
* **Public Health Strategy:** Prioritize awareness programs addressing smoking, diet, and preventive healthcare.
* **Model Deployment:** Integrate the XGBoost model into public health dashboards for real-time cancer mortality monitoring.
* **Further Research:** Incorporate temporal data or hospital-level features to enhance model robustness.
* **Hybrid Approach:** Combine ML predictions with statistical analysis for better interpretability and causal inference.

---

## **Repository Contents**

* `cancer_rate.ipynb` → Jupyter Notebook containing all code and analysis steps
* `README.md` → Project documentation (this file)
* `data/` → link




## **Author**

**Dr. EDJOUKOU Yessoh Gaudens Thecle**
*Mathematics and Science Educator | Data Science | Machine Learning Enthusiast*

