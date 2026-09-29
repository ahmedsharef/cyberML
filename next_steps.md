# Next Steps — Elevating This to a Strong Major Project

## Priority order: do these in sequence before your submission date.

---

## Phase 1 — Make It Demo-Ready (1–2 days)

### 1. Fix the PATH so you can run from any terminal
Add Python and Scripts to your Windows PATH permanently:
- Open **System Properties → Advanced → Environment Variables**
- Under **User variables**, edit `Path` and add:
  - `C:\Users\moham\AppData\Local\Python\pythoncore-3.14-64`
  - `C:\Users\moham\AppData\Local\Python\pythoncore-3.14-64\Scripts`
- After this, `python`, `pip`, and `streamlit` will work from any terminal

### 2. Create a one-click launch script
Create `run_dashboard.bat` in the project root:
```bat
@echo off
set PYTHONPATH=%~dp0
python -m streamlit run dashboard/app.py
pause
```
Double-clicking this file launches the dashboard instantly — good for live demos.

### 3. Run the live monitoring feed and generate real alert data
- Open the dashboard at http://localhost:8501
- Go to **📡 Live Monitoring** → click **▶ Start Live Feed**
- Let it run for 2–3 minutes to populate the database with real detections
- This makes every other page (Overview, Attack Distribution, Alerts) show real live data
- Take screenshots of each page for your report

### 4. Inject sample alerts for the demo
- Go to **⚡ Alert & Response**
- Use the **Inject Simulated Threat** button to create DDoS, Ransomware, and SQL Injection alerts
- This demonstrates the full alerting pipeline end-to-end

### 5. Take and store screenshots
Create a `screenshots/` folder and capture:
- Overview page with populated KPIs
- Live monitoring feed in action
- Attack distribution donut chart
- Model performance gauge (XGBoost at 99.66%)
- Alerts table with real data
- Paste these into README.md (there is a Screenshots section ready for them)

---

## Phase 2 — Strengthen the ML (2–3 days)

### 6. Add cross-validation to the evaluation
Currently models are evaluated on a single 80/20 split. For a major project,
add k-fold cross-validation scores to the comparison report.

In `src/evaluate.py`, add a function:
```python
from sklearn.model_selection import cross_val_score

def cross_validate_model(clf, X, y, cv=5):
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    return {"cv_mean": scores.mean(), "cv_std": scores.std()}
```

### 7. Add UNSW-NB15 support and compare datasets
- Download UNSW-NB15 (see `data/download_instructions.md`)
- Change `DATASET=unsw_nb15` in `.env`
- Re-run `python -m src.train_models`
- Compare results between the two datasets in your report
- This demonstrates dataset generalisability — a key academic contribution

### 8. Add feature importance visualisation to the dashboard
In `dashboard/pages/6_Model_Performance.py`, add a bar chart of the top 30
selected features ranked by RandomForest importance. This makes the feature
selection step visually explainable during your viva.

### 9. Hyperparameter tuning for XGBoost (best model)
Run a small grid search to find better hyperparameters:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}
```
Even a small improvement from 99.66% → 99.80% strengthens the report.

### 10. Explainability with SHAP
Add SHAP (SHapley Additive exPlanations) to explain individual predictions:
```bash
pip install shap
```
Generate a SHAP summary plot for XGBoost and include it in the report.
This is a strong talking point in viva: "I can explain *why* the model
flagged a specific packet as a DDoS attack."

---

## Phase 3 — Strengthen the System (3–4 days)

### 11. Real network packet capture (optional but impressive)
Integrate `scapy` or `pyshark` to capture live network traffic and feed
real packets through the prediction pipeline instead of replaying test-set rows:
```bash
pip install scapy
```
This transforms the project from "classification on a dataset" to
"real-time network intrusion detection system" — a significant upgrade.

### 12. Add a REST API endpoint
Wrap `predict()` in a FastAPI endpoint so external tools can call it:
```bash
pip install fastapi uvicorn
```
Create `src/api.py`:
```python
from fastapi import FastAPI
from src.predict import load_pipeline, predict

app = FastAPI()
bundle = load_pipeline()

@app.post("/predict")
def predict_traffic(record: dict):
    return predict(record, bundle)
```
Run with: `uvicorn src.api:app --reload`
This makes the system production-deployable — great for viva.

### 13. Add user authentication to the dashboard
Add a simple login page to the Streamlit dashboard using `streamlit-authenticator`:
```bash
pip install streamlit-authenticator
```
Prevents unauthorized access — relevant for a security-focused project.

### 14. Docker containerisation
Create a `Dockerfile` so the entire system runs in one command:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboard/app.py"]
```
This demonstrates deployment awareness and makes the project portable.

---

## Phase 4 — Report & Viva Preparation (2–3 days)

### 15. Write the Literature Survey chapter
Key papers to cite (search on Google Scholar):
- "NSL-KDD: A New Dataset for IDS Evaluation" — Tavallaee et al., 2009
- "UNSW-NB15: A Comprehensive Dataset for IDS" — Moustafa & Slay, 2015
- "Random Forests" — Breiman, 2001
- "XGBoost: A Scalable Tree Boosting System" — Chen & Guestrin, 2016
- "SMOTE: Synthetic Minority Over-sampling Technique" — Chawla et al., 2002
- Any recent (2020–2024) survey on ML-based intrusion detection

### 16. Write the Results chapter using real numbers
Use the actual metrics from `reports/model_comparison_report.md`:

| Model | Accuracy | Weighted F1 | AUC |
|---|---|---|---|
| XGBoost | 99.62% | 99.66% | 1.0000 |
| Random Forest | 99.62% | 99.49% | 0.9997 |
| Decision Tree | 99.47% | 99.31% | 0.9977 |
| MLP Neural Net | 99.14% | 98.97% | 0.9996 |
| SVM/SGD | 92.62% | 89.83% | 0.9720 |

These are real numbers from 148,517 NSL-KDD records — cite them with confidence.

### 17. Prepare viva answers
Study the **Viva Q&A Guide** section in `README.md`. Key questions to prepare:
- Why two-stage pipeline?
- Why XGBoost performed best?
- Why SMOTE? What alternatives exist?
- How does feature selection work in this system?
- What are the limitations and future scope?
- How would you deploy this in a real network?

### 18. Record a demo video
Record a 3–5 minute screen recording showing:
1. Starting the dashboard
2. Running the live feed (attacks being detected in real time)
3. Showing the model comparison page
4. Injecting a simulated threat and seeing the alert
Upload to YouTube (unlisted) and include the link in your report.

---

## Phase 5 — Publication-Level Polish (optional, for distinction)

### 19. Write a conference paper
The results (99.66% F1, AUC=1.0) are strong enough for a short paper at:
- ICCCNT (IEEE conference, India-focused)
- ICCCS
- Any Springer LNCS workshop on cybersecurity/ML

### 20. Add Grad-CAM / attention visualisation for MLP
Show which features the neural network attends to most — advanced explainability.

### 21. Deploy to cloud (Streamlit Community Cloud — free)
1. Push the repo to GitHub (already done ✅)
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set `dashboard/app.py` as the entry point
5. Get a public URL like `https://ahmedsharef-cyberml.streamlit.app`
This makes your project accessible from anywhere — impressive for viva and report.

---

## Quick Reference — Commands You'll Use Most

```powershell
# Set PYTHONPATH (run once per terminal session until PATH is fixed)
$env:PYTHONPATH = "C:\Users\moham\OneDrive\Desktop\MLcybe\cyberattack-detection"
$py = "C:\Users\moham\AppData\Local\Python\pythoncore-3.14-64\python.exe"

# Run training (needed after any code/config change)
& $py "C:\Users\moham\OneDrive\Desktop\MLcybe\cyberattack-detection\src\train_models.py"

# Launch dashboard
& $py -m streamlit run "C:\Users\moham\OneDrive\Desktop\MLcybe\cyberattack-detection\dashboard\app.py"

# Run tests
& $py -m pytest "C:\Users\moham\OneDrive\Desktop\MLcybe\cyberattack-detection\tests\" -v
```
