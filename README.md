# LoanGuard AI

A Streamlit-based loan risk assessment interface using the project's saved Random Forest classifier and label encoders.

## Run locally

Use a Python version supported by scikit-learn 1.9.0. The saved model was serialized with scikit-learn 1.9.0, so use that version to avoid pickle compatibility warnings or behavior differences.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## What is included

- Overview dashboard with charts calculated from the supplied dataset
- New assessment form using the saved Random Forest model and existing encoders
- Prediction result page with the model's actual predicted class and class probability when supported
- Sliding Recent Applications card stack populated only by assessments created in the current Streamlit session
- Risk analytics based on the supplied dataset
- Model performance metrics calculated against the held-out split specified by the training script

## Data integrity notes

- The supplied CSV is training/reference data, not a record of live loan applications.
- Assessment history is kept in Streamlit session state only; it is not persisted to a database.
- Model probability is classifier confidence, not a guarantee of repayment or final lending approval.
- The model files are serialized Python artifacts. Install a compatible scikit-learn version for the environment used to train them. Do not load `.pkl` files from untrusted sources.
