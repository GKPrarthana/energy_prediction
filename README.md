# Appliance Energy Consumption Prediction

This project is a comprehensive response to the "Multivariate Time-Series Prediction Using Deep Learning" assessment. The primary objective is to predict appliance energy consumption using a time series dataset from a low energy house. The project covers the entire machine learning lifecycle, from data exploration and feature engineering to model development, evaluation, and deployment as a simple web application.

## Key Features
- **In-Depth EDA:** Detailed exploratory data analysis to uncover trends, seasonality, and correlations.
- **Robust Feature Engineering:** Creation of time-based, lagged, and rolling-window features to enhance model performance.
- **Baseline & Deep Learning Models:** Development and comparison of Linear Regression, Random Forest, and LSTM models.
- **Modular Code Structure:** The entire project is refactored from a notebook into a clean, modular, and reproducible Python pipeline.
- **Interactive Web UI:** A simple web application built with Flask that allows for real-time predictions using the best-performing model.


## Project Structure
The repository is organized into a standard structure for professional data science projects:
```bash
├── data/
│   ├── raw/energydata_complete.csv
│   └── processed/
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── EDA.ipynb
├── reports/
│   └── report.pdf
├── src/
│   └── appliance_energy_predictor/
│       ├── components/
│       ├── pipeline/
│       └── init.py
├── templates/
│   └── index.html
├── app.py
├── main.py
├── setup.py
└── requirements.txt
```

---

## Setup and Installation

Follow these steps to set up the project environment locally.

**1. Clone the Repository:**
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]
```
**2. Create and Activate a Virtual Environment (Recommended):**


For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
Install all required packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

**4. Install the Project as a Local Package:**
This step is crucial for the modular imports to work correct    ly.

```bash
pip install -e .
```

## Usage
There are two main ways to run this project: executing the full training pipeline or running the interactive web application.

**1. Running the Full Training Pipeline**
This script will perform all steps: data preprocessing, feature engineering, model training, evaluation, and saving the final models (random_forest_model.pkl and scaler.pkl) to the /models directory.

From the project root, run:

```bash
python main.py
```

**2. Running the Prediction Web App**
This will launch a local web server with a user interface to make real-time predictions. Make sure you have already run the training pipeline at least once to generate the model files.

From the project root, run:
```bash
python app.py
```
Then, open your web browser and navigate to http://127.0.0.1:5000.

## Technologies Used
- **Language**: Python 3.10

- **Data Manipulation**: Pandas, NumPy

- **Machine Learning**: Scikit-learn

- **Deep Learning**: TensorFlow, Keras

- **Web Framework**: Flask

- **Development**: Jupyter Notebook, Visual Studio Code
