# AI Copilot Instructions - Cancer Diagnosis Predictor

## Project Overview
This is a **Streamlit-based web application** that predicts cancer risk (malignant vs. benign) using a k-Nearest Neighbors (kNN) ML model trained on patient health data. It's a single-file application focused on providing an interactive UI for medical predictions.

## Architecture & Key Components

### Core Application Structure
- **`app.py`** - The entire application (Streamlit single-page app)
  - Input form with 8 patient features (age, gender, smoking, genetic risk, BMI, physical activity, alcohol intake, family history)
  - Pre-trained kNN model and StandardScaler loaded from pickled artifacts
  - Binary classification output with probability distribution

### Data Flow
1. User enters patient details via 2-column form layout
2. Input is scaled using `scaler.pkl` (fitted on training data)
3. kNN model predicts class (0=benign, 1=malignant)
4. Probabilities displayed as percentages for both classes

### Model Artifacts
- **`cancer_knn_model.pkl`** - Trained kNN classifier (must exist; loaded on app startup)
- **`scaler.pkl`** - StandardScaler artifact for feature normalization
- These are loaded once at startup via `@st.cache_resource` for performance

### Training Data
- **`cancer_data.csv`** - 9 columns: 8 features + 1 target (`diagnosis`)
  - Features: age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history
  - Target: diagnosis (0/1 binary)
  - Note: Input form uses 8 features; dataset has `cancer_history` but UI uses `family_history` (same concept)

## Development Patterns & Conventions

### Streamlit-Specific Patterns
- **Layout**: Two-column form using `st.columns(2)` for organized input presentation
- **State Management**: Streamlit handles session state automatically; no manual state needed
- **Performance**: Use `@st.cache_resource` decorator for loading expensive artifacts (models, scalers)
- **Output Format**: Prediction results shown as conditional messages (error for malignant, success for benign) with percentages

### Input Validation
- Features use appropriate input types: `st.number_input()` for continuous, `st.selectbox()` for binary choices
- Input ranges hardcoded (e.g., age 1-120, BMI 10-60) - update if model retraining changes distributions
- All inputs are float/int compatible; scaling happens before prediction

### Common Modifications
When updating the app, consider:
1. **Adding/removing features**: Update form inputs AND ensure model retraining matches
2. **Changing input ranges**: Affects user experience and model assumptions about data
3. **Retraining the model**: Always update `cancer_knn_model.pkl` and potentially `scaler.pkl`
4. **UI styling**: Use Streamlit's built-in components; avoid custom CSS complexity

## Running & Testing

### Prerequisites
- Python 3.7+ (required by Streamlit)
- Dependencies: streamlit, joblib, numpy, pandas, scikit-learn (implied by model format)

### Start the Application
```bash
streamlit run app.py
```
Default runs on `http://localhost:8501`

### Development Workflow
1. Install dependencies (create `requirements.txt` if missing with versions pinned)
2. Ensure model artifacts exist in project root
3. Run `streamlit run app.py`
4. Streamlit auto-reloads on `.py` changes
5. For model retraining: regenerate `.pkl` files, restart app

## Critical Files & Their Purpose

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application; only production code file |
| `cancer_knn_model.pkl` | Serialized kNN model; **must exist at startup** |
| `scaler.pkl` | StandardScaler for feature normalization |
| `cancer_data.csv` | Training/reference dataset; not loaded by app at runtime |

## Integration Points & Dependencies

- **Streamlit**: Provides UI framework, session handling, caching
- **joblib**: Model serialization/deserialization
- **scikit-learn** (implicit): kNN model and StandardScaler come from sklearn
- **numpy/pandas**: Data manipulation in model training (not used in deployed app)

## Potential Issues & Gotchas

1. **Missing Model Artifacts**: App crashes on startup if `.pkl` files absent. Always commit these to git.
2. **Feature Mismatch**: If model trained on different features than input form expects, predictions will be wrong.
3. **Scaler Mismatch**: Using a scaler trained on different data distribution will degrade predictions.
4. **Input Ranges**: Extreme values outside training distribution may produce unreliable predictions (models don't extrapolate well).

## AI Agent Focus Areas

- **For improvements**: Preserve the simple, single-file structure. Avoid over-engineering.
- **For debugging**: Check feature order consistency between input form and model training.
- **For extensions**: Add model comparison (different k values) or feature importance explanations if expanding.
- **For data updates**: Remember to refit the scaler when retraining with new data distributions.
