import joblib
import pandas as pd
import numpy as np
from pathlib import Path
# import importlib

from sklearn.metrics import mean_squared_error
from data_loading import load_data, data_structure
from data_preprocessing import preprocess_data
from preprocess_pipeline import create_preprocessor
from primary_model import get_primary_model, quick_comprace
from train_model import train_model
from cross_validation import cross_validaton_model
from hyperparameter_tuning import tune_gradinet_boosting, tune_xgboost_reg
from model_perfomace_evaluation import evaluation_model

output_dir = Path('../outputs')
output_dir.mkdir(exist_ok=True)

data_path = Path("../data/insurance.csv")
df = load_data(data_path)
# data_s = data_structure(df)

target = 'charges'
X_train, X_test, y_train, y_test = preprocess_data(df,target)
# print(X_train.shape)

preprocessor = create_preprocessor(X_train)

models = get_primary_model(preprocessor)
basline_models_reslut = quick_comprace(
    models=models,
    xtrain=X_train,
    ytrain=y_train,
    xtest=X_test,
    ytest=y_test
)

baseline_df = pd.DataFrame(basline_models_reslut)
baseline_df.to_csv(output_dir/'baseline_model_comparison.csv',index=False)

print(baseline_df.to_string(index=False))

trained_models = {}
for name, model in models.items():
    trained_models[name] = train_model(model, X_train, y_train)


cv_result = []
for name, model in trained_models.items():
    result = cross_validaton_model(model,X_train,y_train)
    result['Model'] = name
    cv_result.append(result)

# print('cv result', cv_result)

cv_df = pd.DataFrame(cv_result)
cv_df = cv_df[['Model', "CV RMSE", "CV MAE", "CV R2"]]
cv_df.to_csv(output_dir/'Cross_validation_result.csv',index=False)

search_gbr = tune_gradinet_boosting(
    preprocessor,
    X_train,
    y_train
)

search_xgb = tune_xgboost_reg(
    preprocessor,
    X_train,
    y_train
)

base_rmse = baseline_df.set_index('Model')['RMSE']
best_baseline_name = base_rmse.idxmin()
best_baseline_rmse = base_rmse.min()

tuned_gbr_rmse = -search_gbr.best_score_
tuned_xgb_rmse = -search_xgb.best_score_

# print(tuned_gbr_rmse)
# print(tuned_xgb_rmse)

if tuned_gbr_rmse <= best_baseline_rmse and tuned_gbr_rmse <= tuned_xgb_rmse:
    best_model = search_gbr.best_estimator_
    best_model_name = 'Tuned GradintBoostingRegressor'
elif tuned_xgb_rmse <= best_baseline_rmse:
    best_model = search_xgb.best_estimator_
    best_model_name = 'Tuned XgboostRegressor'
else:
    best_model = trained_models[best_baseline_name]
    best_model_name = best_baseline_name

# print('Best model name')
# print(best_model_name)

metrics = evaluation_model(
    best_model,
    X_test,
    y_test,
    output_dir
)

joblib.dump(best_model, output_dir/"best_insurance_model.joblib")