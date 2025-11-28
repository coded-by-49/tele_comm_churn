import pandas as pd
import numpy as np
import pickle
import uvicorn
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. SETUP & GLOBAL VARIABLES
# ==========================================
app = FastAPI(title="Telco Churn API", description="Predicts customer churn based on 18 input features.")

# Enable CORS (Allows your frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold our model artifacts
model = None
scaler = None
model_columns = None
explainer = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, model_columns, explainer
    try:
        # Load the files you downloaded from Colab
        # Ensure these files are in the same folder as main.py
        with open('tele_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
            
        # Initialize SHAP Explainer (We create this fresh using the loaded model)
        print("⏳ Initializing SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        print("✅ Artifacts loaded successfully. Server is ready!")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not load artifacts. {e}")
        print("Make sure 'tele_churn_model.pkl', 'scaler.pkl', and 'model_columns.pkl' are in the root directory.")

# ==========================================
# 3. INPUT DATA MODEL (Pydantic)
# ==========================================
class CustomerData(BaseModel):
    # This matches the schema we gave the Frontend Developer
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str = "Yes" # Default if missing
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

# ==========================================
# 4. PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        print("1. Received Data")
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])

        print("2. Creating Tenure Group")
        bins = [0, 15, 48, 73]
        labels = ["New_customers", "Established_customers", "Loyal_customers"]
        input_df['Tenure_Group'] = pd.cut(input_df['tenure'], bins=bins, labels=labels, right=False)

        print("3. Creating HighRisk Interaction")
        # CHECK THIS LINE CAREFULLY FOR 'and' vs '&'
        input_df['HighRisk_Interaction'] = ((input_df['Contract'] == 'Month-to-month') & 
                                           (input_df['InternetService'] == 'Fiber optic')).astype(int)
        
        print("4. Creating Tenure_MonthlyCharges")
        input_df['Tenure_MonthlyCharges'] = input_df['tenure'] * input_df['MonthlyCharges']
        
        print("5. Creating Fiber_Electronic")
        # CHECK THIS LINE CAREFULLY FOR 'and' vs '&'
        input_df['Fiber_Electronic'] = ((input_df['InternetService'] == 'Fiber optic') & 
                                       (input_df['PaymentMethod'] == 'Electronic check')).astype(int)

        print("6. Feature Engineering Done. Starting Encoding...")
        encoded_df = pd.get_dummies(input_df)
        
        print("7. One-hot encoding")
        # --- C. Encoding --- (issue block )
        encoded_df = pd.get_dummies(input_df)
        
        print("8. Changing datatype for new one-hot encoded features")
        # Boolean cleanup
        bool_cols = encoded_df.select_dtypes(include="bool").columns
        encoded_df[bool_cols] = encoded_df[bool_cols].astype(int)

        # --- D. Alignment (CRITICAL) ---
        # Force columns to match the training data exactly
        print("9 reordering column arrangment of data")
        encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

        # --- E. Scaling ---
        print("9 Scaling discerete numerical features ")
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Tenure_MonthlyCharges"]
        encoded_df[numerical_cols] = scaler.transform(encoded_df[numerical_cols])

        # --- F. Prediction ---
        print("9 Making prediction ")
        prediction_binary = model.predict(encoded_df)[0] # 0 or 1
        probability = model.predict_proba(encoded_df)[0][1] # 0.0 to 1.0

        # Output Text
        prediction_label = "Churn" if prediction_binary == 1 else "No Churn"
        
        # # --- G. SHAP Explanation (Why did they churn?) ---
        # # We calculate SHAP values for this specific person
        # shap_values = explainer.shap_values(encoded_df)
        
        # # Handle different SHAP output formats (Binary classification usually returns a list of 2 arrays)
        # if isinstance(shap_values, list):
        #     # Index 1 is the positive class (Churn)
        #     person_shap_values = shap_values[1][0] 
        # else:
        #     person_shap_values = shap_values[0]

        # # Pair feature names with their impact scores
        # feature_importance = list(zip(model_columns, person_shap_values))
        
        # # Sort by absolute impact (Magnitude matters more than direction for "Importance")
        # feature_importance.sort(key = lambda x: abs(x[1]), reverse=True)
        
        # # Get Top 3 Drivers
        # top_factors = []
        # for feature, impact in feature_importance[:3]:
        #     direction = "Increases Risk" if impact > 0 else "Decreases Risk"
        #     top_factors.append({
        #         "feature": feature,
        #         "impact_score": round(float(impact), 4),
        #         "effect": direction
        #     })

        return {
            "prediction": prediction_label,
            "churn_probability": round(float(probability), 4),
            # "risk_factors": top_factors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "active", "message": "Churn Prediction API is running."}

if __name__ == "__main__":
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)