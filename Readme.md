# 🌊 EcoPredict: Predictive Environmental Analytics

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Next.js](https://img.shields.io/badge/Next.js-React-black)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview
The quantification of aquatic pollution (such as microplastics and chemical runoff) typically relies on labor-intensive and expensive manual sampling. **EcoPredict** is a machine learning pipeline and web platform designed to provide a cost-effective, indirect method to forecast water contamination metrics using readily available oceanographic proxy data.

This project was developed to serve as the foundation for a comparative research analysis on the efficacy of various regression algorithms in environmental forecasting.

## 🚀 Key Features
* **Big Data Processing:** Handles and cleans large-scale oceanographic data (CalCOFI dataset with over 800,000 records).
* **Comparative Machine Learning:** Evaluates Multiple Linear Regression, Ridge, Random Forest, and Gradient Boosting algorithms.
* **Automated Metrics:** Automatically computes R², RMSE, and MAE, generating publication-ready charts.
* **API-Ready:** Exports the top-performing model as a serialized `.pkl` artifact, ready to be consumed by a FastAPI/Next.js full-stack dashboard.

## 🛠️ Technology Stack
* **Machine Learning:** Python, Scikit-Learn, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Joblib
* **Frontend Dashboard (Integration Phase):** Next.js, TypeScript, Tailwind CSS
* **Backend API (Integration Phase):** FastAPI / Flask

## 📊 Dataset Context
This model utilizes the **CalCOFI (California Cooperative Oceanic Fisheries Investigations)** dataset. To avoid overfitting commonly seen in small microplastic datasets, this project leverages massive physical oceanographic metrics (Depth, Salinity, Dissolved Oxygen) to predict water temperature variants and proxy contamination zones. 

## ⚙️ Installation & Usage

### 1. Clone the repository
~~~bash
git clone https://github.com/log1-codes/EcoPredict-Machine-Learning-for-Aquatic-Microplastic-Forecasting.git
~~~

### 2. Change the directory 
~~~bash
cd EcoPredict-Machine-Learning-for-Aquatic-Microplastic-Forecasting/ML
~~~

### 3. Get the dataset
Download the **bottle.csv** file from the [CalCOFI Kaggle Dataset](https://www.kaggle.com/datasets) and place it inside the `ML` directory.

### 4. Create a Virtual Environment
~~~bash
python3 -m venv .venv 
~~~

### 5. Activate the Virtual Environment 
~~~bash 
source .venv/bin/activate
~~~
*(Note: If you are on Windows, use `.venv\Scripts\activate`)*

### 6. Install Dependencies
~~~bash 
pip install pandas numpy scikit-learn matplotlib seaborn joblib
~~~

### 7. Run the model 
~~~bash 
python model.py 
~~~

### 8. Outputs
Upon successful execution, the script will generate:
* `model_comparison_metrics.png`: A bar chart comparing the R² and RMSE of all trained models.
* `scaler.pkl`: The saved StandardScaler for future API inputs.
* `best_ecopredict_model.pkl`: The highest-performing regression model saved for production inference.

## 📝 Research Paper Abstract 
*(Draft context for academic submission)*
This project accompanies a comparative analysis titled: *"Predictive Modeling of Aquatic Contamination: Evaluating Machine Learning Approaches for Cost-Effective Environmental Monitoring."* It proposes a software-driven methodology for preliminary environmental monitoring, comparing ensemble methods against traditional linear models to establish the lowest error rate for environmental forecasting.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
