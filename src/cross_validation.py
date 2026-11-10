import numpy as np
from sklearn.model_selection import KFold, cross_validate

def cross_validaton_model(model,xtrain,ytrain):
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    score = cross_validate(
        estimator=model,
        X=xtrain,
        y=ytrain,
        cv=cv,
        scoring={
            'MAE' : "neg_mean_absolute_error",
            'RMSE' : "neg_root_mean_squared_error",
            'R2' : 'r2'
        },
        n_jobs=-1
    )

    result = {
        'CV MAE' : -score['test_MAE'].mean(),
        'CV RMSE' : - score['test_RMSE'].mean(),
        'CV R2' : score['test_R2'].mean()
    }

    return result

if __name__ == "__main__":
    print('Cross validation successfully created')