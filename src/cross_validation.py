import numpy as np
from sklearn.model_selection import KFold, cross_validate


def cross_validaton_model(model, xtrain, ytrain):
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
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        n_jobs=-1,
    )

    result = {
        "CV MAE Mean": -score["test_MAE"].mean(),
        "CV MAE Std": score["test_MAE"].std(),
        "CV RMSE Mean": -score["test_RMSE"].mean(),
        "CV RMSE Std": score["test_RMSE"].std(),
        "CV R2 Mean": score["test_R2"].mean(),
        "CV R2 Std": score["test_R2"].std(),
    }

    return result


if __name__ == "__main__":
    print("Cross validation successfully created")
