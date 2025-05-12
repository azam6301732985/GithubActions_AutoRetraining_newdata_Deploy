# 🚀 Regression CI/CD Project: Auto-Retraining & Local Deployment

### 📌 Project Objective:
Build and deploy a regression model trained on initial data, then automatically retrain with new data using CI/CD GitHub Actions. Log and compare accuracy (MSE) between versions.


## 🔧 Setup (Virtual Environment)
```bash
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
python train_model.py
python app.py

note: screenshots has been added in the folder of venv_results
