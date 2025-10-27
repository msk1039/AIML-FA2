# 🎯 JEE Cutoff Prediction - Updated Project Roadmap

**Last Updated**: October 27, 2025  
**Rank Filter**: 200,000 (optimized for quality)

---

## ✅ COMPLETED PHASES

### Phase 1: Data Loading and Cleaning ✅
**File**: `main.ipynb`  
**Status**: ✅ Complete with 200k filter

**Completed Steps**:
- ✅ Load historical JEE cutoff data (2018-2024)
- ✅ Data exploration and quality checks
- ✅ Remove invalid ranks (MAX_VALID_RANK = 200,000)
- ✅ Focus on Round 7 (final closing ranks)
- ✅ Save `cutoffs_cleaned.csv`

**Output Files**: `cutoffs_cleaned.csv`

---

### Phase 2: Feature Engineering ✅
**File**: `phase2_feature_engineering.ipynb`  
**Status**: ✅ Complete (needs re-run with 200k data)

**Completed Steps**:
- ✅ Lag features (previous 1-3 years cutoffs)
- ✅ Statistical features (mean, std, min, max over years)
- ✅ Categorical encoding (seat type, quota, branch)
- ✅ Aggregate features (institute/branch averages)
- ✅ Time-based features (year trends)
- ✅ Total: 21 engineered features

**Output Files**: `cutoffs_features.csv`, `cutoffs_model_ready.csv`, `feature_names.csv`

---

### Phase 3: Model Building and Training ✅
**File**: `phase3_model_building.ipynb`  
**Status**: ✅ Complete (needs re-run with 200k data)

**Completed Steps**:
- ✅ Time-series train/test split (2018-2023 train, 2024 test)
- ✅ Baseline Linear Regression model
- ✅ Initial XGBoost model
- ✅ Hyperparameter tuning (RandomizedSearchCV with TimeSeriesSplit)
- ✅ Model comparison and selection
- ✅ Feature importance analysis
- ✅ Error analysis by category
- ✅ Model saving (`xgboost_cutoff_model.pkl`)

**What's Implemented from Original Roadmap**:
- ✅ Phase 3.1: Missing value handling
- ✅ Phase 3.2: Time-series train/test split  
- ✅ Phase 3.3: Feature selection (21 features)
- ✅ Phase 4.1: Baseline Linear Regression
- ✅ Phase 4.2: XGBoost training
- ✅ Phase 4.3: Hyperparameter tuning (RandomizedSearchCV)
- ✅ Phase 4.4: Time-series cross-validation (3-fold)
- ✅ Phase 5.1: Metrics (MAE, RMSE, R², MAPE)
- ✅ Phase 5.2: Error analysis by branch/institute
- ✅ Phase 5.3: Feature importance analysis
- ✅ Phase 5.4: Visualizations (actual vs predicted, residuals, feature importance)

**Output Files**: `xgboost_cutoff_model.pkl`, `model_performance.json`, `feature_importance.csv`

---

## 🚀 REMAINING PHASES (To Implement)

### Phase 4: 2025 Predictions with Confidence Intervals ⏳
**File**: `phase4_predictions_2025.ipynb` (NEW - To Create)  
**Status**: ❌ Not started

**Objectives**:
1. Load trained model (`xgboost_cutoff_model.pkl`)
2. Load 2024 data as base for predictions
3. Generate 2025 cutoff predictions
4. **Add confidence intervals** (not yet implemented)
5. Export predictions to CSV

**What to Implement**:

#### 4.1 Load Model and 2024 Data
```python
# Load trained model
with open('xgboost_cutoff_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load 2024 data (most recent year) as input
df_2024 = df[df['year'] == 2024].copy()
```

#### 4.2 Prepare Features for 2025 Prediction
```python
# Use 2024 as lag features for 2025
# Update year = 2025
# Recalculate rolling statistics with 2024 included
```

#### 4.3 Generate Point Predictions
```python
predictions_2025 = model.predict(X_2025)
```

#### 4.4 Add Confidence Intervals ⭐ NEW
**Method 1: Quantile Regression (Recommended)**
```python
# Train two additional models:
# - Lower bound model (10th percentile)
# - Upper bound model (90th percentile)

from xgboost import XGBRegressor

# Lower bound (conservative)
model_lower = XGBRegressor(objective='reg:quantileerror', 
                           quantile_alpha=0.1)
model_lower.fit(X_train, y_train)
lower_bound = model_lower.predict(X_2025)

# Upper bound (optimistic)
model_upper = XGBRegressor(objective='reg:quantileerror', 
                           quantile_alpha=0.9)
model_upper.fit(X_train, y_train)
upper_bound = model_upper.predict(X_2025)
```

**Method 2: Prediction Intervals from Residuals**
```python
# Calculate prediction errors on test set
errors = y_test - y_test_pred
std_error = np.std(errors)

# 90% confidence interval
lower_bound = predictions_2025 - 1.645 * std_error
upper_bound = predictions_2025 + 1.645 * std_error
```

#### 4.5 Export Predictions
```python
results = pd.DataFrame({
    'institute': institutes_2025,
    'branch': branches_2025,
    'seat_type': seat_types_2025,
    'quota': quotas_2025,
    'predicted_cutoff_2025': predictions_2025,
    'lower_bound': lower_bound,
    'upper_bound': upper_bound
})

results.to_csv('predictions_2025.csv', index=False)
```

**Expected Output Files**: 
- `predictions_2025.csv` - Main predictions with confidence intervals
- `predictions_2025_summary.csv` - Summary by institute/branch

---

### Phase 5: Interactive Prediction Interface ⏳
**File**: `phase5_interactive_predictor.ipynb` (NEW - To Create)  
**Status**: ❌ Not started  
**Priority**: Optional (for user-friendly application)

**Objectives**:
1. Create function to predict cutoff for any seat combination
2. Build student recommendation system
3. Interactive widgets for easy querying

**What to Implement**:

#### 5.1 Cutoff Prediction Function
```python
def predict_cutoff(institute, branch, seat_type, quota, year=2025):
    """
    Predict cutoff for a specific seat combination.
    
    Parameters:
    -----------
    institute : str
        Institute name (e.g., "IIT Delhi")
    branch : str
        Branch code (e.g., "CSE", "ME")
    seat_type : str
        Seat type (e.g., "OPEN", "SC")
    quota : str
        Quota (e.g., "AI", "HS")
    year : int
        Prediction year (default: 2025)
    
    Returns:
    --------
    dict : Predicted cutoff with confidence intervals
    """
    # 1. Load model
    # 2. Prepare features for this seat
    # 3. Make prediction
    # 4. Return with confidence interval
    
    return {
        'predicted_cutoff': prediction,
        'lower_bound': lower,
        'upper_bound': upper,
        'confidence': '90%'
    }
```

#### 5.2 Student Recommendation System
```python
def get_eligible_seats(student_rank, preferences={}):
    """
    Get list of seats a student can likely get.
    
    Parameters:
    -----------
    student_rank : int
        Student's JEE Main rank
    preferences : dict
        Optional filters:
        - branches: ['CSE', 'ECE', 'ME']
        - institutes: ['IIT Delhi', 'NIT Trichy']
        - seat_type: 'OPEN'
        - quota: 'AI'
    
    Returns:
    --------
    DataFrame : Eligible seats sorted by preference
    """
    # 1. Load all 2025 predictions
    # 2. Filter by student rank (rank <= upper_bound)
    # 3. Apply preference filters
    # 4. Sort by cutoff (best options first)
    
    return eligible_seats_df
```

#### 5.3 Interactive Widgets (Optional)
```python
import ipywidgets as widgets
from IPython.display import display

# Dropdown for institute selection
institute_dropdown = widgets.Dropdown(
    options=institutes_list,
    description='Institute:'
)

# Dropdown for branch
branch_dropdown = widgets.Dropdown(
    options=branches_list,
    description='Branch:'
)

# Button to predict
predict_button = widgets.Button(description='Predict Cutoff')

def on_predict_click(b):
    result = predict_cutoff(
        institute=institute_dropdown.value,
        branch=branch_dropdown.value,
        seat_type=seat_type_dropdown.value,
        quota=quota_dropdown.value
    )
    print(f"\n🎯 Predicted 2025 Cutoff: {result['predicted_cutoff']:,.0f}")
    print(f"📊 Confidence Interval: {result['lower_bound']:,.0f} - {result['upper_bound']:,.0f}")

predict_button.on_click(on_predict_click)

# Display widgets
display(institute_dropdown, branch_dropdown, predict_button)
```

**Expected Output Files**: 
- `student_recommendations.csv` - Personalized seat recommendations
- Interactive notebook cells for queries

---

### Phase 6: Model Retraining on Full Dataset ⏳
**File**: `phase6_full_retrain.ipynb` (NEW - To Create)  
**Status**: ❌ Not started  
**Priority**: Medium (for production deployment)

**Objectives**:
1. Retrain model on **ALL years** (2018-2024) - no test set
2. Use best hyperparameters found in Phase 3
3. Save as production model

**What to Implement**:

#### 6.1 Load Full Dataset
```python
# Load ALL data (no train/test split)
df_full = pd.read_csv('cutoffs_model_ready.csv')
X_full = df_full[feature_columns]
y_full = df_full['cutoff']
```

#### 6.2 Train on All Data
```python
# Use best parameters from Phase 3
best_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    # ... (from model_performance.json)
}

# Train on full dataset
model_production = xgb.XGBRegressor(**best_params)
model_production.fit(X_full, y_full)

# Save
with open('xgboost_cutoff_model_production.pkl', 'wb') as f:
    pickle.dump(model_production, f)
```

**Rationale**: 
- Phase 3 model trained on 2018-2023 (held out 2024 for testing)
- For actual 2025 predictions, use ALL data including 2024
- Gives model access to most recent trends

**Expected Output Files**: 
- `xgboost_cutoff_model_production.pkl` - Model trained on all data
- `production_model_info.json` - Metadata

---

### Phase 7: Web Application (Optional) ⏳
**Technology**: Flask/Streamlit  
**Status**: ❌ Not started  
**Priority**: Low (for public deployment)

**Objectives**:
1. Create web interface for predictions
2. Deploy model as REST API
3. User-friendly UI for students

**Implementation Options**:

**Option A: Streamlit App** (Faster, simpler)
```python
import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('xgboost_cutoff_model_production.pkl', 'rb'))

st.title('🎓 JEE Cutoff Predictor 2025')

# Input fields
institute = st.selectbox('Select Institute', institutes_list)
branch = st.selectbox('Select Branch', branches_list)
seat_type = st.selectbox('Seat Type', ['OPEN', 'OBC-NCL', 'SC', 'ST'])
quota = st.selectbox('Quota', ['AI', 'HS'])

if st.button('Predict Cutoff'):
    prediction = predict_cutoff(institute, branch, seat_type, quota)
    st.success(f"Predicted 2025 Cutoff: {prediction:,.0f}")
```

**Option B: Flask REST API** (More flexible)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'cutoff': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📋 IMMEDIATE ACTION ITEMS

### Step 1: Re-run Phases 1-3 with 200k Filter ⚡ URGENT
Since you've updated to MAX_VALID_RANK = 200,000:

```bash
✅ Already done: Updated main.ipynb with 200k filter

📝 TODO:
1. Run main.ipynb (Phase 1) - Generate cutoffs_cleaned.csv
2. Run phase2_feature_engineering.ipynb - Generate cutoffs_model_ready.csv
3. Run phase3_model_building.ipynb - Train final model

Expected Time: ~15-20 minutes total
```

**Expected Results with 200k filter**:
- Test MAE: **~1,800-2,200 ranks** (huge improvement!)
- Test R²: **~0.93-0.95** 
- Test MAPE: **~40-60%** (vs current 115%)
- Best predictions: **< 0.2% error** for top IITs
- Worst predictions: **< 80% error** (vs current 132%)

### Step 2: Create Phase 4 (2025 Predictions) 🎯 NEXT
After Phase 3 completes with good results:

```bash
📝 Create: phase4_predictions_2025.ipynb
- Implement confidence intervals (quantile regression)
- Generate 2025 cutoff predictions
- Export to CSV for analysis
```

### Step 3: Optional Enhancements
- **Phase 5**: Interactive predictor (if building tool for students)
- **Phase 6**: Production model retraining (if deploying)
- **Phase 7**: Web app (if making it public)

---

## 🎯 RECOMMENDED NEXT STEPS

### Option A: Quick Finish (Recommended for Course Project)
1. ✅ Re-run Phases 1-3 with 200k data (~20 min)
2. ✅ Create Phase 4 basic predictions (~30 min)
3. ✅ Generate 2025 predictions CSV
4. ✅ Submit with documentation

**Total Time**: ~1 hour  
**Deliverables**: Trained model + 2025 predictions + analysis

### Option B: Complete Implementation (Recommended for Portfolio)
1. ✅ Re-run Phases 1-3 with 200k data
2. ✅ Phase 4: Full predictions with confidence intervals
3. ✅ Phase 5: Interactive predictor notebook
4. ✅ Phase 6: Production model
5. ⚪ Phase 7: Web app (optional)

**Total Time**: ~3-4 hours  
**Deliverables**: Production-ready system with UI

---

## 📊 WHAT'S ALREADY IMPLEMENTED

From your original roadmap, these are **✅ COMPLETE**:

| Original Phase | Status | Implemented In |
|---------------|--------|----------------|
| Phase 3.1: Handle Missing Values | ✅ Done | phase3_model_building.ipynb Step 5 |
| Phase 3.2: Train/Test Split | ✅ Done | phase3_model_building.ipynb Step 6 |
| Phase 3.3: Feature Selection | ✅ Done | phase2_feature_engineering.ipynb (21 features) |
| Phase 4.1: Baseline Model | ✅ Done | phase3_model_building.ipynb Step 7 |
| Phase 4.2: XGBoost Training | ✅ Done | phase3_model_building.ipynb Step 8 |
| Phase 4.3: Hyperparameter Tuning | ✅ Done | phase3_model_building.ipynb Step 9 |
| Phase 4.4: Cross-Validation | ✅ Done | phase3_model_building.ipynb Step 9 |
| Phase 5.1: Metrics | ✅ Done | phase3_model_building.ipynb Steps 7-10 |
| Phase 5.2: Error Analysis | ✅ Done | phase3_model_building.ipynb Step 13 |
| Phase 5.3: Feature Importance | ✅ Done | phase3_model_building.ipynb Step 11 |
| Phase 5.4: Visualizations | ✅ Done | phase3_model_building.ipynb Steps 11-12 |
| **Phase 6: Final Model & Save** | ✅ Done | phase3_model_building.ipynb Step 15 |

## ❌ NOT YET IMPLEMENTED

| Original Phase | Status | Where to Implement |
|---------------|--------|-------------------|
| Phase 6.1: Retrain on Full Data | ❌ Missing | Create phase6_full_retrain.ipynb |
| Phase 6.2: Predict 2025 | ❌ Missing | Create phase4_predictions_2025.ipynb |
| Phase 6.3: Confidence Intervals | ❌ Missing | Add to phase4_predictions_2025.ipynb |
| Interactive Interface | ❌ Missing | Create phase5_interactive_predictor.ipynb |

---

## 🏆 PROJECT COMPLETION STATUS

**Overall Progress**: 75% Complete ✅✅✅⚪

- ✅ Phase 1: Data Cleaning (100%)
- ✅ Phase 2: Feature Engineering (100%)
- ✅ Phase 3: Model Building & Training (100%)
- ⏳ Phase 4: 2025 Predictions (0% - needs creation)
- ⏳ Phase 5: Interactive Interface (0% - optional)
- ⏳ Phase 6: Production Retrain (0% - optional)
- ⏳ Phase 7: Web App (0% - optional)

---

## 📁 FILE STRUCTURE

```
/Users/meow/Downloads/jee-cutoffs/
├── 📊 Data Files
│   ├── josaa_cutoffs_pivoted_by_rounds.csv (raw data)
│   ├── cutoffs_cleaned.csv (Phase 1 output)
│   ├── cutoffs_features.csv (Phase 2 output)
│   ├── cutoffs_model_ready.csv (Phase 2 final)
│   └── feature_names.csv (feature metadata)
│
├── 🤖 Model Files
│   ├── xgboost_cutoff_model.pkl (trained model)
│   ├── model_performance.json (metrics)
│   └── feature_importance.csv (feature scores)
│
├── 📓 Phase Notebooks (Completed)
│   ├── main.ipynb (Phase 1) ✅
│   ├── phase2_feature_engineering.ipynb ✅
│   └── phase3_model_building.ipynb ✅
│
├── 📓 Phase Notebooks (To Create)
│   ├── phase4_predictions_2025.ipynb ❌
│   ├── phase5_interactive_predictor.ipynb ❌ (optional)
│   ├── phase6_full_retrain.ipynb ❌ (optional)
│   └── phase7_web_app.py ❌ (optional)
│
└── 📝 Documentation
    ├── PROJECT_ROADMAP_UPDATED.md (this file)
    ├── DATA_CLEANING_SUMMARY.md
    ├── IMPROVEMENT_OPTIONS.md
    └── HYPERPARAMETER_ADJUSTMENT.md
```

---

## 🚀 LET'S START!

**Your immediate next step**: Re-run Phases 1-3 with the 200k filter to get the best possible model, then we'll create Phase 4 for 2025 predictions! 🎯
