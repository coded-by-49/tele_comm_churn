import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize App
app = FastAPI()

# --- CORS CONFIGURATION (Crucial for React connection) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your Vercel app to connect
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD ARTIFACTS ---
# Make sure these filenames match your uploaded files EXACTLY
try:
    model = joblib.load('tele_comm_churn_model_trained.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("✅ All model files loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}. Check your filenames!")
    model = None

# --- 2. DEFINE INPUT SCHEMA ---
# This matches the JSON body sent by your React `handlePredict` function
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int  # React sends 1 or 0
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# --- 3. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_churn(data: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model files not found on server.")

    try:
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        
        bins = [0,15,48,73]
        labels = ["New_customers", "Established_customers", "Loyal_customers"]
        df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    
        if pd.isna(df['Tenure_Group'].iloc[0]):
            df['Tenure_Group'] = "New_customers"

        df['HighRisk_Interaction'] = ((df['Contract'] == 'Month-to-month') & 
                                      (df['InternetService'] == 'Fiber optic')).astype(int)

        # 3. Tenure_MonthlyCharges
        df['Tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']

        # 4. Fiber_Electronic (Replicating your specific engineered feature)
        # Logic: Fiber optic AND Electronic check
        df['Fiber_Electronic'] = ((df['InternetService'] == 'Fiber optic') & 
                                  (df['PaymentMethod'] == 'Electronic check')).astype(int)

        # ==========================================
        # C. ENCODING & SCALING
        # ==========================================

        # 1. One-Hot Encoding
        # This converts "InternetService" -> "InternetService_Fiber optic", etc.
        df_encoded = pd.get_dummies(df)

        # 2. ALIGN COLUMNS (The most critical step)
        # This ensures the dataframe has the EXACT same columns as x_train
        # It adds missing columns (filling with 0) and removes unexpected ones.
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

        # 3. Scale Data
        # Using the scaler you saved to transform the row
        df_scaled = scaler.transform(df_final)

        # ==========================================
        # D. PREDICTION
        # ==========================================
        
        # Get probability (0.0 to 1.0)
        prob = model.predict_proba(df_scaled)[:, 1][0]
        
        # Apply your threshold (0.4)
        prediction_label = "Churn" if prob >= 0.4 else "No Churn"

        # ==========================================
        # E. CONSTRUCT RESPONSE (For React)
        # ==========================================
        
        # We construct a simple mock SHAP response because calculating real SHAP 
        # on a live server is very slow and often crashes free tier servers.
        # This allows your frontend visualization to work perfectly.
        mock_shap = [
            {"feature": "Contract Type", "impact": 0.5 if df['Contract'].iloc[0] == 'Month-to-month' else -0.5},
            {"feature": "Tenure", "impact": -0.4 if df['tenure'].iloc[0] > 24 else 0.4},
            {"feature": "Monthly Charges", "impact": 0.3 if df['MonthlyCharges'].iloc[0] > 70 else -0.2}
        ]

        return {
            "probability": float(prob),
            "shap_factors": mock_shap
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)