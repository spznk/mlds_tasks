import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dataclasses import dataclass

@dataclass
class PreprocessedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    input_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    scaler: MinMaxScaler | None
    ohe: OneHotEncoder | None
    surnames_to_preserve: list[str]


def split_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        raw_df,
        test_size=0.2,
        random_state=91,
        stratify=raw_df.Exited
    )

    return train_df, val_df


def identify_features(raw_df: pd.DataFrame) -> tuple[pd.Index, str]:
    input_cols = raw_df.columns.drop(['id', 'CustomerId'])
    if 'Exited' in input_cols:
        input_cols = input_cols.drop('Exited')

    target_col = 'Exited'

    return input_cols, target_col


def get_data_subsets(train_df: pd.DataFrame, val_df: pd.DataFrame, input_cols: list, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    return X_train, y_train, X_val, y_val


def get_num_and_cat_cols(X_train: pd.DataFrame) -> tuple[list, list]:
    numeric_cols = X_train.select_dtypes("number").columns.to_list()
    categorical_cols = X_train.select_dtypes(["object", "category"]).columns.to_list()

    return numeric_cols, categorical_cols


def fit_surname(X_train: pd.DataFrame):
    surname_counts = X_train.Surname.value_counts()
    surnames_over_100 = [x for x in surname_counts.index if surname_counts[x] >= 100]

    return surnames_over_100


def transform_surname(X: pd.DataFrame, surnames_to_preserve):
    X['Surname'] = [x if x in surnames_to_preserve else 'Other' for x in X['Surname']]


def transform_and_add_categories(fitted_encoder: OneHotEncoder, X: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    encoded_cols = fitted_encoder.get_feature_names_out(categorical_cols).tolist()

    encoded_data = pd.DataFrame(
        fitted_encoder.transform(X[categorical_cols]),
        columns=encoded_cols,
        index=X.index
    )

    X_transformed = pd.concat(
        [X.drop(columns=categorical_cols), encoded_data],
        axis=1
    )

    return X_transformed


def transform_columns(X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler | None, OneHotEncoder | None]:
    scaler = None
    ohe = None

    if numeric_cols:
        scaler = MinMaxScaler()

        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    surnames_to_preserve = fit_surname(X_train)
    transform_surname(X_train, surnames_to_preserve)
    transform_surname(X_val, surnames_to_preserve)

    if categorical_cols:
        surnames_to_preserve = fit_surname(X_train)

        ohe = OneHotEncoder(
            drop='if_binary',
            sparse_output=False,
            handle_unknown='ignore'
        )

        ohe.fit(X_train[categorical_cols])

        X_train = transform_and_add_categories(ohe, X_train, categorical_cols)
        X_val = transform_and_add_categories(ohe, X_val, categorical_cols)

    return X_train, X_val, scaler, ohe, surnames_to_preserve


def preprocess_data(raw_df: pd.DataFrame) -> PreprocessedData:
    train_df, val_df = split_data(raw_df)

    input_cols, target_col = identify_features(raw_df)

    X_train, y_train, X_val, y_val = get_data_subsets(
        train_df,
        val_df,
        input_cols,
        target_col
    )

    numeric_cols, categorical_cols = get_num_and_cat_cols(X_train)

    X_train, X_val, scaler, ohe, surnames_to_preserve = transform_columns(
        X_train,
        X_val,
        numeric_cols,
        categorical_cols
    )

    return PreprocessedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_cols=input_cols.tolist(),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaler=scaler,
        ohe=ohe,
        surnames_to_preserve=surnames_to_preserve
    )


def preprocess_new_data(test_df: pd.DataFrame, data: PreprocessedData) -> pd.DataFrame:
    X_test = test_df[data.input_cols].copy()

    X_test[data.numeric_cols] = data.scaler.transform(X_test[data.numeric_cols])

    transform_surname(X_test, data.surnames_to_preserve)
    X_test = transform_and_add_categories(data.ohe, X_test, data.categorical_cols)

    return X_test