from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def create_preprocessor(X):
    numeric_feature = X.select_dtypes(
        include='number'
    ).columns.tolist()

    categorical_feature = X.select_dtypes(
        include = ['object', 'category']
    ).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(
                handle_unknown='ignore',
            ))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipe, numeric_feature),
            ('categorical', categorical_pipe, categorical_feature)
        ]
    )
    return preprocessor

if __name__ == "__main__":
    print("Preprocessing module ready")
