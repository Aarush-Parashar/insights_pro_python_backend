import os
import io
import json
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import timedelta

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer   
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# Supabase
from supabase import create_client, Client

# --- Configuration ---
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Insights Pro Backend",
    version="1.0.0",
)

# CORS
origins = ["*"]  # Restrict this to your Flutter domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class PreprocessingConfig(BaseModel):
    # e.g., {"num_missing_values": "Mean", "encoding": "Ordinal", ...}
    config: Dict[str, str] 
    target_variable: str

class ModelSelectionResponse(BaseModel):
    model_names: Dict[str, str] # e.g. {"XGBoost": "92%"}
    prediction_input_data: List[Dict[str, Any]]

class ForecastResponse(BaseModel):
    title: str
    time_label: str
    value_label: str
    historical: List[Dict[str, Any]] 
    forecast: List[Dict[str, Any]]

# ... (imports and supabase init) ...

# --- Auth Models ---
class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    user_id: str
    token: str
    is_first_time_user: bool = False
# --- Add these to your existing main.py ---

class FileListResponse(BaseModel):
    files: List[str]

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    columns: List[str]

    

@app.post("/auth/signup", response_model=AuthResponse)
async def signup(request: AuthRequest):
    try:
        # 1. Create user in Supabase
        res = supabase.auth.sign_up({
            "email": request.email, 
            "password": request.password
        })
        
        # 2. Check for errors or missing user
        if not res.user:
             # If "Confirm Email" is ON in Supabase, user is None until clicked.
             # If OFF, user is returned immediately.
             raise HTTPException(status_code=400, detail="Signup failed. check email confirmation settings.")

        # 3. Return success
        return AuthResponse(
            user_id=res.user.id, 
            token=res.session.access_token if res.session else "check_email_for_link",
            is_first_time_user=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": request.email, 
            "password": request.password
        })
        
        if not res.user or not res.session:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return AuthResponse(
            user_id=res.user.id,
            token=res.session.access_token,
            is_first_time_user=False
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Login failed. Check credentials.")


# --- Dependency: Auth & User Context ---
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Verifies the Bearer token with Supabase Auth.
    Returns the user object if valid.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        # Supabase-py uses the token to get the user
        response = supabase.auth.get_user(token)
        if not response.user:
             raise HTTPException(status_code=401, detail="Invalid Token")
        return response.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/data/files", response_model=FileListResponse)
async def list_user_files(user: Any = Depends(get_current_user)):
    """Lists all files in the user's storage folder."""
    try:
        # List files in the user's folder
        res = supabase.storage.from_("datasets").list(path=user.id)
        
        # Extract filenames 
        filenames = [f['name'] for f in res if not f['name'].startswith('.')]
        return {"files": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


# --- Helper: Download Data from Supabase ---
def get_dataframe_from_storage(file_path: str) -> pd.DataFrame:
    try:
        # Download file bytes from Supabase 'datasets' bucket
        response = supabase.storage.from_("datasets").download(file_path)
        return pd.read_csv(io.BytesIO(response))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found in storage: {e}")

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "active", "service": "Insights Pro Backend"}

# 1. File Upload
@app.post("/data/upload")
async def upload_file(file: UploadFile = File(...), user: Any = Depends(get_current_user)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        
        # Verify it's a valid CSV before saving
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except:
            raise HTTPException(status_code=400, detail="Invalid CSV content")

        # Save to Supabase Storage
        file_path = f"{user.id}/{file.filename}"
        
        # 'upsert' allows overwriting if user uploads same file again
        supabase.storage.from_("datasets").upload(
            file_path, 
            contents, 
            file_options={"content-type": "text/csv", "upsert": "true"}
        )
        
        return {
            "file_id": file_path, # We use the storage path as ID
            "filename": file.filename,
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.get("/data/load/{filename}", response_model=UploadResponse)
async def load_existing_file(filename: str, user: Any = Depends(get_current_user)):
    """Loads metadata (columns) for an existing file so the user can switch to it."""
    file_path = f"{user.id}/{filename}"
    try:
        # Reuse the helper to download and read columns
        df = get_dataframe_from_storage(file_path)
        
        return {
            "file_id": file_path,
            "filename": filename,
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File load failed: {str(e)}")

# 2. Predict - Step 1: Preprocessing Questions (Static Config)
@app.get("/predict/questions")
async def get_predict_questions():
    # Matches the PDF workflow
    return [
        {"key": "categorical_check", "question": "Does your dataset contain categorical columns?", "type": "radio", "options": ["Yes", "No"]},
        {"key": "num_missing_values", "question": "Numerical Missing Values Strategy", "type": "dropdown", "options": ["Mean", "Median", "Drop"]},
        {"key": "cat_missing_values", "question": "Categorical Missing Values Strategy", "type": "dropdown", "options": ["Mode", "Previous Value", "Drop"]},
        {"key": "outliers_intensity", "question": "Outliers Treatment Intensity", "type": "dropdown", "options": ["Low", "Medium", "High", "None"]},
        {"key": "encoding", "question": "Categorical Encoding Strategy", "type": "dropdown", "options": ["Ordinal", "One-Hot"]}, 
        {"key": "scaling", "question": "Feature Scaling Strategy", "type": "dropdown", "options": ["Normalization", "Standardization", "None"]},
    ]

# 3. Predict - Step 2: The "AI Selection" Engine (Core Logic)
@app.post("/predict/model_selection/{file_path:path}", response_model=ModelSelectionResponse)
async def ai_model_selection(file_path: str, config: PreprocessingConfig, user: Any = Depends(get_current_user)):
    """
    1. Loads data.
    2. Builds a Scikit-Learn Pipeline based on user config.
    3. Trains 4 models (RF, XGB, SVM, Linear/Log).
    4. Saves the BEST model to Supabase Storage.
    5. Returns accuracy scores.
    """
    
    # A. Load Data
    df = get_dataframe_from_storage(file_path)
    target = config.target_variable
    
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target variable not found in dataset")
    initial_count = len(df)
    df = df.dropna(subset=[target])
    
    if len(df) == 0:
         raise HTTPException(status_code=400, detail="Target variable column is completely empty.")
    # --- FIX END ---

    # B. Separate X and y
    X = df.drop(columns=[target])
    y = df[target]

    # C. Detect Problem Type (Regression or Classification)
    # Heuristic: If target is object or low cardinality numeric, it's classification
    is_classification = False
    if y.dtype == 'object' or (y.nunique() < 20 and pd.api.types.is_integer_dtype(y)):
        is_classification = True
        
    # D. Build Preprocessing Pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    # 1. Numeric Transformer
    num_strategy = "mean" if config.config.get("num_missing_values") == "Mean" else "median"
    scaler = StandardScaler() if config.config.get("scaling") == "Standardization" else MinMaxScaler()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_strategy)),
        ('scaler', scaler)
    ])

    # 2. Categorical Transformer
    cat_strategy = "most_frequent" # "Mode"
    
    # Decide Encoder
    if config.config.get("encoding") == "One-Hot":
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cat_strategy)),
        ('encoder', encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

  
    # E. Train Models
    
    le = None
    if is_classification:
        le = LabelEncoder()
        # Fit on the entire dataset 'y' first
        y = le.fit_transform(y) 
    
    # NOW split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # (Remove the old 'if is_classification' block that was here)

    # Define candidate models
    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC(probability=True)
        }
    else:
        # ... (rest of the regression models remain the same)
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "XGBoost": XGBRegressor(),
            "SVM": SVR()
        }

    results = {}
    best_score = -1
    best_model_name = ""
    best_pipeline = None

    # Train Loop
    for name, model in models.items():
        try:
            # Create full pipeline: Preprocessor -> Model
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            if is_classification:
                score = accuracy_score(y_test, preds)
                results[name] = f"{score*100:.1f}%"
            else:
                score = r2_score(y_test, preds)
                results[name] = f"{score*100:.1f}%"
            
            # Track best model
            if score > best_score:
                best_score = score
                best_model_name = name
                best_pipeline = clf
                
        except Exception as e:
            # If a model fails (like XGBoost strictness), log it and continue
            print(f"Model {name} failed to train: {e}")
            results[name] = "N/A (Data Error)"

    # Check if we have at least one working model
    if best_pipeline is None:
        raise HTTPException(status_code=400, detail="All models failed to train on this data.")


    # F. Save BEST Model to Supabase (for Step 3)
    # We serialize the entire pipeline (including preprocessor)
    model_artifact = {
        "pipeline": best_pipeline,
        "is_classification": is_classification,
        "label_encoder": le,
        "target_name": target,
        "features_info": {
            "numeric": numeric_features.tolist(),
            "categorical": categorical_features.tolist(),
            "unique_cats": {col: X[col].dropna().unique().tolist() for col in categorical_features}
        }
    }
    
    buffer = io.BytesIO()
    joblib.dump(model_artifact, buffer)
    buffer.seek(0)
    
    model_path = f"{user.id}/{file_path.split('/')[-1]}_model.joblib"
    supabase.storage.from_("models").upload(model_path, buffer.getvalue(), file_options={"upsert": "true"})

    # G. Generate Dynamic Input Fields for Frontend
    # Based on the PDF, return fields so user can input data for prediction
    input_fields = []
    
    # Add Dropdowns for Categorical
    for col in categorical_features:
        unique_vals = X[col].dropna().unique().tolist()
        # Limit dropdown size for UI sanity
        input_fields.append({
            "name": col,
            "inputtype": "Dropdown",
            "values": unique_vals[:50], 
            "placeholder": f"Select {col}"
        })
        
    # Add Number Inputs
    for col in numeric_features:
        input_fields.append({
            "name": col,
            "inputtype": "Number",
            "placeholder": f"Enter {col}",
            "min": float(X[col].min()),
            "max": float(X[col].max())
        })

    # Sort results by accuracy (descending)
# --- FIX START: Safe Sorting Helper ---
    def get_score_value(item):
        name, score_str = item
        # If the model failed (N/A), give it a score of -1 so it goes to the bottom
        if "N/A" in score_str:
            return -1.0
        try:
            return float(score_str.strip('%'))
        except:
            return -1.0

    # Sort using the helper
    sorted_results = dict(sorted(results.items(), key=get_score_value, reverse=True))
    # --- FIX END ---
    return ModelSelectionResponse(
        model_names=sorted_results,
        prediction_input_data=input_fields
    )

# 4. Predict - Step 3: Get Result
@app.post("/predict/result/{file_path:path}")
async def get_prediction_result(file_path: str, input_data: Dict[str, Any], user: Any = Depends(get_current_user)):
    """
    Loads the saved model from Storage and runs inference.
    """
    model_path = f"{user.id}/{file_path.split('/')[-1]}_model.joblib"
    
    try:
        # Download Model
        response = supabase.storage.from_("models").download(model_path)
        artifact = joblib.load(io.BytesIO(response))
        
        pipeline = artifact["pipeline"]
        is_class = artifact["is_classification"]
        le = artifact["label_encoder"]
        
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction = pipeline.predict(input_df)
        result_val = prediction[0]
        
        confidence_str = "N/A" # Default for regression

        # Decode label if classification & try to get probability
        if is_class:
            if hasattr(pipeline, "predict_proba"):
                try:
                    # Get probability of the predicted class
                    probs = pipeline.predict_proba(input_df)
                    max_prob = np.max(probs)
                    confidence_str = f"{max_prob*100:.2f}%"
                except:
                    pass
            
            if le:
                result_val = le.inverse_transform([int(result_val)])[0]
            
        return {
            "prediction": str(result_val),
            # FIX: Added the 'confidence' field required by Flutter
            "confidence": confidence_str,
            "model_used": "Best Performer (Auto-Selected)",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
# 5. Visualize (Safer Numeric-Only Version)
@app.post("/visualize/{file_path:path}")
async def visualize_data(file_path: str, config: PreprocessingConfig, user: Any = Depends(get_current_user)):
    df = get_dataframe_from_storage(file_path)
    target = config.target_variable
    
    # Filter for numeric columns only (float/int)
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    # Remove target from X options if present
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    # DECISION LOGIC:
    # 1. If we have other numeric columns, pick the first one as 'X'.
    # 2. If NO other numeric columns exist, use the Row Index (0, 1, 2...) as 'X'.
    if len(numeric_cols) > 0:
        x_col = numeric_cols[0]
        use_index = False
    else:
        x_col = "Row Index"
        use_index = True

    # Limit to 100 points for performance
    sample = df.sample(min(100, len(df))).fillna(0)
    
    data_points = []
    for idx, row in sample.iterrows():
        # Ensure 'y' is a number (handle bad data)
        try:
            y_val = float(row[target])
        except:
            y_val = 0.0

        # Ensure 'x' is a number
        if use_index:
            x_val = float(idx)
        else:
            try:
                x_val = float(row[x_col])
            except:
                x_val = float(idx) # Fallback to index if column has bad data
                
        data_points.append({"x": x_val, "y": y_val})
        
    return {
        "chart_type": "Scatter",
        "title": f"Relationship between {x_col} and {target}",
        "x_label": x_col,
        "y_label": target,
        "data": data_points
    }

# --- Helper: Auto-Detect Date Column ---
def detect_date_column(df: pd.DataFrame) -> str:
    # 1. Check for columns with "date", "time", "year" in the name
    candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
    
    # 2. Try to convert candidates to datetime objects
    for col in candidates:
        try:
            pd.to_datetime(df[col], errors='raise')
            return col
        except:
            continue
            
    # 3. If no obvious name, check ALL object/string columns
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            # If >80% of rows parse as valid dates, assume it is a date column
            if pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.8:
                return col
        except:
            continue
            
    return None

# 6. Forecast (Time-Series Prediction)
@app.post("/forecast/{file_path:path}", response_model=ForecastResponse)
async def get_forecast(file_path: str, config: PreprocessingConfig, user: Any = Depends(get_current_user)):
    """
    1. Detects a Date column.
    2. Aggregates data by Date (Daily/Monthly).
    3. Trains a model (Random Forest) to learn the trend.
    4. Predicts the next 30 periods.
    """
    df = get_dataframe_from_storage(file_path)
    target = config.target_variable
    
    # A. Detect Date Column
    date_col = detect_date_column(df)
    if not date_col:
        raise HTTPException(status_code=400, detail="No date/time column detected in this dataset. Forecast requires a time column.")

    # B. Preprocess Data
    try:
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, target]) # Drop bad dates/values
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Aggregate duplicates (e.g. multiple sales on same day -> sum them)
        # We assume if target is numeric, we sum it. If it's a rate, we might want mean, but sum is safer for "Sales".
        df_grouped = df.groupby(date_col)[target].sum().reset_index()
        
        if len(df_grouped) < 5:
             raise HTTPException(status_code=400, detail="Not enough historical data points to forecast (need at least 5).")

    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Data preparation failed: {e}")

    # C. Feature Engineering for ML (Convert Date -> Numbers)
    # We use "Days since start" as the feature to predict trend
    start_date = df_grouped[date_col].min()
    df_grouped['days_since'] = (df_grouped[date_col] - start_date).dt.days
    
    X = df_grouped[['days_since']]
    y = df_grouped[target]
    
    # D. Train Forecaster
    # We use Random Forest because it handles non-linear trends better than Linear Regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # E. Generate Future Dates (Next 30 Steps)
    last_days_since = df_grouped['days_since'].max()
    # Try to detect frequency (approximate)
    avg_diff = df_grouped['days_since'].diff().mean() 
    step_size = max(1, int(round(avg_diff if not np.isnan(avg_diff) else 1)))
    
    future_days = []
    future_dates = []
    
    last_date = df_grouped[date_col].max()
    
    for i in range(1, 31): # Forecast 30 steps ahead
        next_day_idx = last_days_since + (i * step_size)
        future_days.append([next_day_idx])
        future_dates.append(last_date + timedelta(days=int(i * step_size)))
        
    # Predict
    future_preds = model.predict(future_days)
    
    # F. Format Response
    historical_data = []
    for _, row in df_grouped.iterrows():
        historical_data.append({
            "date": row[date_col].strftime("%Y-%m-%d"),
            "value": float(row[target])
        })
        
    forecast_data = []
    for date_val, pred_val in zip(future_dates, future_preds):
        forecast_data.append({
            "date": date_val.strftime("%Y-%m-%d"),
            "value": float(pred_val)
        })

    return {
        "title": f"30-Period Forecast for {target}",
        "time_label": date_col,
        "value_label": target,
        "historical": historical_data,
        "forecast": forecast_data
    }