from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np


def get_primary_model(preprocessor):
    models = {
        'Liner Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),

        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ))
        ]),

        'Gradient Boosting' : Pipeline([
            ('preprocessor', preprocessor),
            ('model',GradientBoostingRegressor(
                random_state=42
            ))
        ]),

        'XGboost' : Pipeline([
            ('preprocessor',preprocessor),
            ('model',XGBRegressor(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror'
            ))
        ])
    }

    return models

def quick_comprace(models, xtrain, ytrain, xtest, ytest):
    results = []

    for name, pipeline in models.items():
        pipeline.fit(xtrain,ytrain)
        pred = pipeline.predict(xtest)

        results.append({
            'Model' : name,
            'MAE' : mean_absolute_error(ytest,pred),
            'RMSE' : np.sqrt(mean_squared_error(ytest,pred)),
            "R2 Score" : r2_score(ytest,pred)
        })

    return sorted(results, key=lambda X: X["RMSE"])

if __name__ == "__main__":
    print("primary model successfuly runing")