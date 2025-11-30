import pandas as pd
import numpy as np
import pickle
import uvicorn
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Telco Churn API", description="Predicts customer churn based on 18 input features.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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
        with open('tele_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
    
        print("Initializing SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        print("____Artifacts loaded successfully. Server is ready!")

    except FileNotFoundError as e:
        print(f"!!! Error: Could not load artifacts. {e}")
        print("Make sure 'tele_churn_model.pkl', 'scaler.pkl', and 'model_columns.pkl' are in the root directory.")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str = "Yes" 
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
    
        input_df['HighRisk_Interaction'] = ((input_df['Contract'] == 'Month-to-month') & 
                                           (input_df['InternetService'] == 'Fiber optic')).astype(int)
        
        print("4. Creating Tenure_MonthlyCharges")
        input_df['Tenure_MonthlyCharges'] = input_df['tenure'] * input_df['MonthlyCharges']
        
        print("5. Creating Fiber_Electronic")

        input_df['Fiber_Electronic'] = ((input_df['InternetService'] == 'Fiber optic') & 
                                       (input_df['PaymentMethod'] == 'Electronic check')).astype(int)

        print("6. Feature Engineering Done. Starting Encoding...")
        encoded_df = pd.get_dummies(input_df)
        
        print("7. One-hot encoding")

        encoded_df = pd.get_dummies(input_df)
        
        print("8. Changing datatype for new one-hot encoded features")
        # Boolean cleanup
        bool_cols = encoded_df.select_dtypes(include="bool").columns
        encoded_df[bool_cols] = encoded_df[bool_cols].astype(int)


        print("9 reordering column arrangment of data")
        encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)


        print("10 Scaling discerete numerical features ")
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Tenure_MonthlyCharges"]
        encoded_df[numerical_cols] = scaler.transform(encoded_df[numerical_cols])

        print("11 Making prediction ")
        prediction_binary = model.predict(encoded_df)[0] # 0 or 1
        probability = model.predict_proba(encoded_df)[0][1] # 0.0 to 1.0


        prediction_label = "Churn" if prediction_binary == 1 else "No Churn"
        
        print(f"this is the prediciton made for this customer {prediction_label}, This is a prediction_binary {probability*100}")

    
        # --- 3. ROBUST SHAP (Fixed for 3D Arrays) ---
        shap_factors = [] 
        
        try:
            print("\n=== SHAP COMPUTATION STARTED ===")
            
            shap_values = explainer.shap_values(encoded_df)
            print(f" -> Raw shap_values type: {type(shap_values)}")

       
            if isinstance(shap_values, list):

                vals = shap_values[1]
            else:
                vals = shap_values

            print(f" -> Shape before slicing: {vals.shape}")


            if len(vals.shape) == 3:
                print(" -> Detected 3D Array (Sample, Feature, Class)")
                # Slice: 
                # [0] = The first (and only) customer
                # [:] = All 34 features
                # [1] = The Positive Class (Churn)
                vals = vals[0, :, 1]
            
            # Handle (1, 34) Shape (If it was already 2D)
            elif len(vals.shape) == 2:
                print(" -> Detected 2D Array (Sample, Feature)")
                vals = vals[0]

            print(f" -> Final Shape (must be (34,)): {vals.shape}")

            # 4. Create Feature Importance List
            feature_importance = list(zip(model_columns, vals))

            # 5. Sort by Magnitude (Absolute value)
            feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)

            # 6. Extract Top 3
            for feature, impact in feature_importance[:3]:
                shap_factors.append({
                    "feature": feature,
                    "impact": float(impact) 
                })

            print("=== SHAP SUCCESS ===\n")

        except Exception as e:
            # Fallback (Log error but don't crash app)
            print(f"⚠️ SHAP Logic Failed: {str(e)}")
            shap_factors = []

        return {
            "prediction": prediction_label,
            "probability": probability,      
            "shap_factors": shap_factors     
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "active", "message": "Churn Prediction API is running."}

if __name__ == "__main__":
    port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)