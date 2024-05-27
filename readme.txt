# Loan Prediction Project

This repository contains the code and data for a machine learning project aimed at predicting loan eligibility based on various financial and demographic factors. The project utilizes different machine learning models to analyze and predict the outcome.

## Project Structure

- **app.py**: The main application file that runs the prediction model.
- **Decision_Tree_Model.pkl**: Serialized Decision Tree model for loan prediction.
- **Loan Prediction.csv**: Dataset containing loan application data.
- **model.csv**: Additional model-related data.
- **train.csv**: Training dataset used for model training.
- **XGboost.ipynb**: Jupyter notebook for exploring and training the XGBoost model.
- **data/**:
  - **Data collection/**: Scripts and notebooks for data collection.
  - **Data pre processing/**: Scripts and notebooks for data preprocessing.
  - **pycache/**: Cache files (usually included in `.gitignore`).
- **static/**: Static files for the web application (CSS, JS, images).
- **templates/**: HTML templates for the web application.

## Installation

To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/loan-prediction-project.git
cd loan-prediction-project
pip install -r requirements.txt
