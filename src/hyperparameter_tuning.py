from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

def tune_gradinet_boosting(preprocessor, xtrain, ytrain):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(
                random_state=42
            ))
        ]
    )

    param_dist = {
        "model__n_estimators" : randint(100, 600),
        "model__learning_rate" : uniform(0.01, 0.2),
        "model__max_depth" : randint(2,8),
        "model__min_samples_split" : randint(2,10),
        "model__min_samples_leaf" : randint(1,5),
        "model__subsample" : uniform(0.1, 0.9)
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(xtrain,ytrain)

    return search

def tune_xgboost_reg(processor, X_train, y_train):
    pipe = Pipeline(
        steps=[
            ('preprcessor', processor),
            ('model', XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    param_dict = {
        'model__n_estimators' : randint(100,600),
        'model__learning_rate' : uniform(0.01, 0.2),
        'model__max_depth' : randint(2,10),
        'model__min_child_weight' : randint(1, 10),
        'model__subsample' : uniform(0.7, 0.3),
        'model__colsample_bytree' : uniform(0.7, 0.3)
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dict,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train,y_train)

    return search



if __name__ == "__main__":
    from pathlib import Path
    path = Path("../data/insurance.csv")
    from data_loading import load_data
    k = load_data(path)
    from data_preprocessing import preprocess_data
    x_tr, x_ts, y_tr, y_ts = preprocess_data(k,target_colmun='charges')
    from preprocess_pipeline import create_preprocessor
    tune_gradinet_boosting(create_preprocessor(x_tr), x_tr, y_tr)
    print('Hyper parameter tuning created gradinet bosting work perfectly')
    tune_xgboost_reg(create_preprocessor(x_tr), x_tr, y_tr)
    print('Hyper parameter tuning created parfectly xgboost work perfectly')