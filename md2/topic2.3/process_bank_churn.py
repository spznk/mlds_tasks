import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dataclasses import dataclass

@dataclass
class PreprocessedData:
    """
    Container for storing preprocessed datasets and fitted transformers.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    input_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    scaler: MinMaxScaler | None
    ohe: OneHotEncoder | None


def split_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw dataset into training and validation subsets.

    The split preserves the distribution of the target variable using
    stratification to ensure both datasets contain a similar proportion
    of target classes.

    Args:
        raw_df (pd.DataFrame): Original dataset containing features and target.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - train_df: Training dataset (80% of original data).
            - val_df: Validation dataset (20% of original data).
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=0.2,
        random_state=91,
        stratify=raw_df.Exited
    )

    return train_df, val_df


def identify_features(raw_df: pd.DataFrame) -> tuple[pd.Index, str]:
    """
    Identify input features and target column names.

    Columns that contain identifiers or the target variable are excluded
    from the model input features.

    Args:
        raw_df (pd.DataFrame): Original dataset.

    Returns:
        tuple[pd.Index, str]:
            - input_cols: Names of columns used as model inputs.
            - target_col: Name of the target variable.
    """
    input_cols = raw_df.columns.drop(['id', 'CustomerId', 'Surname'])
    if 'Exited' in input_cols:
        input_cols = input_cols.drop('Exited')

    target_col = 'Exited'

    return input_cols, target_col


def get_data_subsets(train_df: pd.DataFrame, val_df: pd.DataFrame, input_cols: list, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separate datasets into input features (X) and target values (y).

    Creates independent copies of the data to avoid modifying the original
    training and validation DataFrames.

    Args:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame): Validation dataset.
        input_cols (list or pd.Index): Feature column names.
        target_col (str): Target column name.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train: Training input features.
            - y_train: Training target values.
            - X_val: Validation input features.
            - y_val: Validation target values.
    """
    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    return X_train, y_train, X_val, y_val


def get_num_and_cat_cols(X_train: pd.DataFrame) -> tuple[list, list]:
    """
    Identify numerical and categorical columns in the dataset.

    Numerical columns are detected based on pandas numeric data types.
    Categorical columns include object and category data types.

    Args:
        X_train (pd.DataFrame): Training feature dataset.

    Returns:
        tuple[list, list]:
            - numeric_cols: Names of numerical feature columns.
            - categorical_cols: Names of categorical feature columns.
    """
    numeric_cols = X_train.select_dtypes("number").columns.to_list()
    categorical_cols = X_train.select_dtypes(["object", "category"]).columns.to_list()

    return numeric_cols, categorical_cols


def transform_and_add_categories(fitted_encoder: OneHotEncoder, X: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Transform categorical features using a fitted OneHotEncoder and add them to the dataset.

    The function applies the encoder to the specified categorical columns,
    creating new one-hot encoded columns. The original categorical columns
    are dropped from the dataset.

    Args:
        fitted_encoder (OneHotEncoder): Pre-fitted OneHotEncoder instance.
        X (pd.DataFrame): Dataset containing categorical features.
        categorical_cols (list): Names of categorical columns to transform.

    Returns:
        pd.DataFrame: Dataset with one-hot encoded features added and original
                      categorical columns removed.
    """
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
    """
    Apply preprocessing transformations to numerical and categorical features.

    Numerical features are scaled using MinMaxScaler.
    Categorical features are encoded using OneHotEncoder.
    The fitted transformers are returned for applying the same transformations
    to future datasets.

    Args:
        X_train (pd.DataFrame): Training input features.
        X_val (pd.DataFrame): Validation input features.
        numeric_cols (list): Names of numerical columns.
        categorical_cols (list): Names of categorical columns.

    Returns:
        tuple:
            - X_train (pd.DataFrame): Transformed training features.
            - X_val (pd.DataFrame): Transformed validation features.
            - scaler (MinMaxScaler or None): Fitted numerical scaler.
            - ohe (OneHotEncoder or None): Fitted categorical encoder.
    """
    scaler = None
    ohe = None

    if numeric_cols:
        scaler = MinMaxScaler()

        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    if categorical_cols:
        ohe = OneHotEncoder(
            drop='if_binary',
            sparse_output=False,
            handle_unknown='ignore'
        )

        ohe.fit(X_train[categorical_cols])

        X_train = transform_and_add_categories(ohe, X_train, categorical_cols)

        X_val = transform_and_add_categories(ohe, X_val, categorical_cols)

    return X_train, X_val, scaler, ohe


def preprocess_data(raw_df: pd.DataFrame) -> PreprocessedData:
    """
    Execute the complete preprocessing pipeline.

    The pipeline performs:
        1. Train/validation split.
        2. Feature and target identification.
        3. Dataset separation into X and y.
        4. Numerical and categorical feature detection.
        5. Feature scaling and categorical encoding.

    Args:
        raw_df (pd.DataFrame): Original dataset.

    Returns:
        PreprocessedData: Container holding all preprocessed data and fitted transformers.

    """
    train_df, val_df = split_data(raw_df)

    input_cols, target_col = identify_features(raw_df)

    X_train, y_train, X_val, y_val = get_data_subsets(
        train_df,
        val_df,
        input_cols,
        target_col
    )

    numeric_cols, categorical_cols = get_num_and_cat_cols(X_train)

    X_train, X_val, scaler, ohe = transform_columns(
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
        ohe=ohe
    )

def preprocess_new_data(test_df: pd.DataFrame, data: PreprocessedData) -> pd.DataFrame:
    """
    Preprocess new data using the same transformations as the training data.

    This function applies the fitted scaler and encoder from the training data
    to ensure consistency in feature representation.

    Args:
        test_df (pd.DataFrame): New dataset to preprocess.
        data (PreprocessedData): Preprocessed training data containing fitted transformers.

    Returns:
        pd.DataFrame: Preprocessed new data.
    """
    X_test = test_df[data.input_cols].copy()

    X_test[data.numeric_cols] = data.scaler.transform(X_test[data.numeric_cols])

    X_test = transform_and_add_categories(data.ohe, X_test, data.categorical_cols)

    return X_test