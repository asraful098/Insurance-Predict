from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# import importlib
from sklearn.metrics import mean_squared_error

from cross_validation import cross_validaton_model
from data_loading import data_structure, load_data
from data_preprocessing import preprocess_data
from hyperparameter_tuning import tune_gradinet_boosting, tune_xgboost_reg
from model_perfomace_evaluation import evaluation_model
from preprocess_pipeline import create_preprocessor
from primary_model import get_primary_model, quick_comprace
from train_model import train_model

output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

data_path = Path("../data/insurance.csv")
df = load_data(data_path)
# data_s = data_structure(df)

target = "charges"
X_train, X_test, y_train, y_test = preprocess_data(df, target)
# print(X_train.shape)

preprocessor = create_preprocessor(X_train)

models = get_primary_model(preprocessor)
basline_models_reslut = quick_comprace(
    models=models, xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test
)

baseline_df = pd.DataFrame(basline_models_reslut)
baseline_df.to_csv(output_dir / "baseline_model_comparison.csv", index=False)

print(baseline_df.to_string(index=False))

trained_models = {}
for name, model in models.items():
    trained_models[name] = train_model(model, X_train, y_train)


cv_result = []
for name, model in models.items():
    result = cross_validaton_model(model, X_train, y_train)
    result["Model"] = name
    cv_result.append(result)

# print('cv result', cv_result)

cv_df = pd.DataFrame(cv_result)
cv_df = cv_df[
    [
        "Model",
        "CV RMSE Mean",
        "CV RMSE Std",
        "CV MAE Mean",
        "CV MAE Std",
        "CV R2 Mean",
        "CV R2 Std",
    ]
]
cv_df.to_csv(output_dir / "Cross_validation_result.csv", index=False)

search_gbr = tune_gradinet_boosting(preprocessor, X_train, y_train)

search_xgb = tune_xgboost_reg(preprocessor, X_train, y_train)

tuning_results = []

for name, search in [
    ("Gradient Boosting", search_gbr),
    ("XGBoost", search_xgb),
]:
    tuning_results.append(
        {
            "Model": name,
            "Best CV RMSE": -search.best_score_,
            "Best Parameters": str(search.best_params_),
        }
    )

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    output_dir / "hyperparameter_tuning_results.csv",
    index=False,
)

baseline_cv = cv_df.set_index("Model")["CV RMSE Mean"]

candidate_scores = {name: score for name, score in baseline_cv.items()}

candidate_scores["Tuned Gradient Boosting"] = -search_gbr.best_score_
candidate_scores["Tuned XGBoost"] = -search_xgb.best_score_

best_model_name = min(
    candidate_scores,
    key=candidate_scores.get,
)
# print(tuned_gbr_rmse)
# print(tuned_xgb_rmse)

if best_model_name == "Tuned Gradient Boosting":
    best_model = search_gbr.best_estimator_

elif best_model_name == "Tuned XGBoost":
    best_model = search_xgb.best_estimator_

else:
    # Train the selected baseline model on the complete training set.
    best_model = train_model(
        models[best_model_name],
        X_train,
        y_train,
    )

# print('Best model name')
# print(best_model_name)

metrics = evaluation_model(best_model, X_test, y_test, output_dir)

joblib.dump(best_model, output_dir / "best_insurance_model.joblib")
