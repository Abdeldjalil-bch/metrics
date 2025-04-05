import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb

# Streamlit page configuration
st.set_page_config(page_title="Machine Learning Model Selector")

# File uploader
file = st.file_uploader(
    label="Upload Train and Test Datasets",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="First upload train dataset, then test dataset (optional)"
)

# Initialize dataframes
train = pd.DataFrame()
test = pd.DataFrame()

# File reading logic
if file:
    if len(file) == 1:
        try:
            # Read single file
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            st.write("Dataset provided:")
            st.write(train.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

    elif len(file)  >= 2:
        try:
            # Read train and test files
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            test = pd.read_csv(file[1]) if file[1].type == "text/csv" else pd.read_excel(file[1])

            st.write("Train Dataset:")
            st.write(train.head())
            st.write("Test Dataset:")
            st.write(test.head())
        except Exception as e:
            st.error(f"Error reading files: {e}")

    else:
        st.warning("Please upload at least one dataset")

# Feature and target selection
if not train.empty:
    features = st.multiselect(label="Select Features", options=train.columns)
    target = st.selectbox(label="Select Target", options=train.columns)
    if features and target:
        x = train[features]
        y = train[target]


        def create_custom_preprocessor(train):
            """
            Create a customizable preprocessing pipeline with Streamlit widgets
            """
            # Separate numeric and categorical columns
            numeric_columns = train.select_dtypes(include=[np.number]).columns
            categorical_columns = train.select_dtypes(exclude=[np.number]).columns

            st.subheader("Numeric Column Preprocessing")

            # Numeric Imputation Strategy
            numeric_imputation_options = [
                "Mean",
                "Median",
                "Most Frequent",
                "Constant",
                "KNN Imputer"
            ]
            num_impute_method = st.selectbox(
                "Numeric Imputation Method",
                numeric_imputation_options
            )

            # Numeric Scaling Strategy
            scaling_options = [
                "Standard Scaler",
                "MinMax Scaler",
                "Robust Scaler",
                "No Scaling"
            ]
            num_scaling_method = st.selectbox(
                "Numeric Scaling Method",
                scaling_options
            )

            st.subheader("Categorical Column Preprocessing")

            # Categorical Imputation Strategy
            cat_imputation_options = [
                "Most Frequent",
                "Constant",
                "Simple Imputer"
            ]
            cat_impute_method = st.selectbox(
                "Categorical Imputation Method",
                cat_imputation_options
            )

            # Categorical Encoding Strategy
            encoding_options = [
                "One-Hot Encoding",
                "Label Encoding",
                "No Encoding"
            ]
            cat_encoding_method = st.selectbox(
                "Categorical Encoding Method",
                encoding_options
            )

            # Construct Numeric Transformer
            numeric_transformer_steps = []

            # Imputation for Numeric Columns
            if num_impute_method == "Mean":
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="mean")
                )
            elif num_impute_method == "Median":
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="median")
                )
            elif num_impute_method == "Most Frequent":
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="most_frequent")
                )
            elif num_impute_method == "Constant":
                constant_value = st.number_input(
                    "Constant Imputation Value",
                    value=0.0
                )
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="constant", fill_value=constant_value)
                )
            elif num_impute_method == "KNN Imputer":
                knn_neighbors = st.slider(
                    "KNN Imputer Neighbors",
                    min_value=1,
                    max_value=30,
                    value=15
                )
                numeric_transformer_steps.append(
                    KNNImputer(n_neighbors=knn_neighbors)
                )

            # Scaling for Numeric Columns
            if num_scaling_method == "Standard Scaler":
                numeric_transformer_steps.append(StandardScaler())
            elif num_scaling_method == "MinMax Scaler":
                numeric_transformer_steps.append(MinMaxScaler())
            elif num_scaling_method == "Robust Scaler":
                numeric_transformer_steps.append(RobustScaler())

            # Construct Categorical Transformer
            categorical_transformer_steps = []

            # Imputation for Categorical Columns
            if cat_impute_method == "Most Frequent":
                categorical_transformer_steps.append(
                    SimpleImputer(strategy="most_frequent")
                )
            elif cat_impute_method == "Constant":
                constant_value = st.text_input(
                    "Constant Categorical Imputation Value",
                    value="Unknown"
                )
                categorical_transformer_steps.append(
                    SimpleImputer(strategy="constant", fill_value=constant_value)
                )

            # Encoding for Categorical Columns
            if cat_encoding_method == "One-Hot Encoding":
                categorical_transformer_steps.append(
                    OneHotEncoder(handle_unknown='ignore')
                )
            elif cat_encoding_method == "Label Encoding":
                categorical_transformer_steps.append(
                    LabelEncoder()
                )

            # Create pipelines
            numeric_transformer = make_pipeline(*numeric_transformer_steps) if numeric_transformer_steps else None
            categorical_transformer = make_pipeline(
                *categorical_transformer_steps) if categorical_transformer_steps else None

            # Create column transformer
            preprocessor = make_column_transformer(
                (numeric_transformer, make_column_selector(dtype_include=np.number)) if numeric_transformer else None,
                (categorical_transformer,
                 make_column_selector(dtype_exclude=np.number)) if categorical_transformer else None,
                remainder='passthrough'
            )

            return preprocessor


        # Example usage in main Streamlit app
        if not train.empty:
            # Call the function to create custom preprocessor
            preprocessor = create_custom_preprocessor(train)

        # Importations supplémentaires nécessaires
        from sklearn.svm import SVR, SVC
        from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        import lightgbm as lgb


        def create_model_with_manual_params(model_type, model_name, preprocessor):
            """
            Create a machine learning model with manually input parameters
            """
            st.subheader(f"Model Parameters for {model_name}")

            if model_type == "Regression":
                regression_models = {
                    "Linear Regression": LinearRegression,
                    "KNN Regression": KNeighborsRegressor,
                    "XGBoost Regression": xgb.XGBRegressor,
                    "SVM Regression": SVR,
                    "AdaBoost Regression": AdaBoostRegressor,
                    "Random Forest Regression": RandomForestRegressor,
                    "Gradient Boosting Regression": GradientBoostingRegressor,
                    "LightGBM Regression": lgb.LGBMRegressor
                }

                # Paramètres génériques pour SVR
                if model_name == "SVM Regression":
                    st.write("SVM Regression Parameters")
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    c_value = st.number_input("Regularization Parameter (C)", min_value=0.0, value=1.0)
                    epsilon = st.number_input("Epsilon", min_value=0.0, value=0.1)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', SVR(
                            kernel=kernel,
                            C=c_value,
                            epsilon=epsilon
                        ))
                    ])
                # Paramètres génériques pour KNN Regression
                elif model_name == "KNN Regression":
                    # Manual input for KNN Regression
                    st.write("KNN Regression Parameters")
                    n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
                    weights = st.text_input("Weights (uniform/distance)", value="uniform")
                    algorithm = st.text_input("Algorithm (auto/ball_tree/kd_tree/brute)", value="auto")
                    metric = st.text_input("Distance Metric", value="minkowski")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', KNeighborsRegressor(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            metric=metric
                        ))
                    ])
                # Paramètres génériques pour XGBoost Regression
                elif model_name == "XGBoost Regression":
                    # Manual input for XGBoost Regression
                    st.write("XGBoost Regression Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1, format="%.3f")
                    max_depth = st.number_input("Max Depth", min_value=1, value=6)
                    subsample = st.number_input("Subsample", min_value=0.0, max_value=1.0, value=0.8, format="%.2f")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth),
                            subsample=subsample
                        ))
                    ])

                elif model_name == "AdaBoost Regression":
                    st.write("AdaBoost Regression Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=50)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=1.0)
                    loss = st.selectbox("Loss Function", ["linear", "square", "exponential"])

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', AdaBoostRegressor(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            loss=loss
                        ))
                    ])

                # Paramètres génériques pour Random Forest Regression
                elif model_name == "Random Forest Regression":
                    st.write("Random Forest Regression Parameters")
                    n_estimators = st.number_input("Number of Trees", min_value=1, value=100)
                    max_depth = st.number_input("Max Depth", min_value=1, value=None,
                                                help="None means nodes are expanded until all leaves are pure")
                    min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth) if max_depth is not None else None,
                            min_samples_split=min_samples_split
                        ))
                    ])

                # Paramètres génériques pour Gradient Boosting Regression
                elif model_name == "Gradient Boosting Regression":
                    st.write("Gradient Boosting Regression Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1)
                    max_depth = st.number_input("Max Depth", min_value=1, value=3)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth)
                        ))
                    ])

                # Paramètres génériques pour LightGBM Regression
                elif model_name == "LightGBM Regression":
                    st.write("LightGBM Regression Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1)
                    max_depth = st.number_input("Max Depth", min_value=-1, value=-1, help="-1 means no limit")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', lgb.LGBMRegressor(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth)
                        ))
                    ])

                # Modèles existants (Linear, KNN, XGBoost) restent inchangés
                else:
                    # Utiliser la logique existante pour les autres modèles
                    model = create_existing_model(model_name, preprocessor, model_type)

            else:  # Classification
                classification_models = {
                    "Logistic Regression": LogisticRegression,
                    "KNN Classification": KNeighborsClassifier,
                    "XGBoost Classification": xgb.XGBClassifier,
                    "SVM Classification": SVC,
                    "AdaBoost Classification": AdaBoostClassifier,
                    "Random Forest Classification": RandomForestClassifier,
                    "Gradient Boosting Classification": GradientBoostingClassifier,
                    "LightGBM Classification": lgb.LGBMClassifier
                }
                # Paramètres génériques pour logistic Classification
                if model_name == "Logistic Regression":
                    # Manual input for Logistic Regression
                    st.write("Logistic Regression Parameters")
                    penalty = st.text_input("Penalty (l2/l1/elasticnet/none)", value="l2")
                    c_value = st.number_input("Regularization Strength (C)", min_value=0.0, value=1.0, format="%.3f")
                    solver = st.text_input("Solver (lbfgs/newton-cg/liblinear/sag/saga)", value="lbfgs")
                    multi_class = st.text_input("Multi-class Strategy (auto/ovr/multinomial)", value="auto")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(
                            penalty=penalty,
                            C=c_value,
                            solver=solver,
                            multi_class=multi_class
                        ))
                    ])
                # Paramètres génériques pour SVM Classification
                elif model_name == "SVM Classification":
                    st.write("SVM Classification Parameters")
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    c_value = st.number_input("Regularization Parameter (C)", min_value=0.0, value=1.0)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', SVC(
                            kernel=kernel,
                            C=c_value
                        ))
                    ])
                # Paramètres génériques pour KNN Classification
                elif model_name == "KNN Classification":
                    # Manual input for KNN Classification
                    st.write("KNN Classification Parameters")
                    n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
                    weights = st.text_input("Weights (uniform/distance)", value="uniform")
                    algorithm = st.text_input("Algorithm (auto/ball_tree/kd_tree/brute)", value="auto")
                    metric = st.text_input("Distance Metric", value="minkowski")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            metric=metric
                        ))
                    ])
                # Paramètres génériques pour XGBoost Classification
                elif model_name == "XGBoost Classification":
                    # Manual input for XGBoost Classification
                    st.write("XGBoost Classification Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1, format="%.3f")
                    max_depth = st.number_input("Max Depth", min_value=1, value=6)
                    subsample = st.number_input("Subsample", min_value=0.0, max_value=1.0, value=0.8, format="%.2f")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', xgb.XGBClassifier(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth),
                            subsample=subsample
                        ))
                    ])
                # Paramètres génériques pour AdaBoost Classification
                elif model_name == "AdaBoost Classification":
                    st.write("AdaBoost Classification Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=50)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=1.0)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', AdaBoostClassifier(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate
                        ))
                    ])

                # Paramètres génériques pour Random Forest Classification
                elif model_name == "Random Forest Classification":
                    st.write("Random Forest Classification Parameters")
                    n_estimators = st.number_input("Number of Trees", min_value=1, value=100)
                    max_depth = st.number_input("Max Depth", min_value=1, value=None,
                                                help="None means nodes are expanded until all leaves are pure")
                    min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth) if max_depth is not None else None,
                            min_samples_split=min_samples_split
                        ))
                    ])

                # Paramètres génériques pour Gradient Boosting Classification
                elif model_name == "Gradient Boosting Classification":
                    st.write("Gradient Boosting Classification Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1)
                    max_depth = st.number_input("Max Depth", min_value=1, value=3)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', GradientBoostingClassifier(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth)
                        ))
                    ])

                # Paramètres génériques pour LightGBM Classification
                elif model_name == "LightGBM Classification":
                    st.write("LightGBM Classification Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1)
                    max_depth = st.number_input("Max Depth", min_value=-1, value=-1, help="-1 means no limit")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', lgb.LGBMClassifier(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth)
                        ))
                    ])

                # Modèles existants (Logistic, KNN, XGBoost) restent inchangés
                else:
                    # Utiliser la logique existante pour les autres modèles
                    model = create_existing_model(model_name, preprocessor, model_type)

            return model


        # Fonction auxiliaire pour gérer les modèles existants
        def create_existing_model(model_name, preprocessor, model_type):
            # Implémentation des modèles existants (comme dans votre code précédent)
            if model_type == "Regression":
                if model_name == "XGBoost Regression":
                    # Manual input for XGBoost Regression
                    st.write("XGBoost Regression Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1, format="%.3f")
                    max_depth = st.number_input("Max Depth", min_value=1, value=6)
                    subsample = st.number_input("Subsample", min_value=0.0, max_value=1.0, value=0.8, format="%.2f")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor(
                        n_estimators=int(n_estimators),
                        learning_rate=learning_rate,
                        max_depth=int(max_depth),
                        subsample=subsample
                        ))
                    ])
                # Paramètres génériques pour lénaire reg
                elif model_name == "Linear Regression":
                    # Manual input for Linear Regression
                    st.write("Linear Regression Parameters")
                    fit_intercept = st.checkbox("Fit Intercept", value=True)
                    normalize = st.checkbox("Normalize", value=False)

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', LinearRegression(
                            fit_intercept=fit_intercept,
                            normalize=normalize
                        ))
                    ])
                # Paramètres génériques pour KNN Regression
                elif model_name == "KNN Regression":
                    # Manual input for KNN Regression
                    st.write("KNN Regression Parameters")
                    n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
                    weights = st.text_input("Weights (uniform/distance)", value="uniform")
                    algorithm = st.text_input("Algorithm (auto/ball_tree/kd_tree/brute)", value="auto")
                    metric = st.text_input("Distance Metric", value="minkowski")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', KNeighborsRegressor(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            metric=metric
                        ))
                    ])
            # Paramètres génériques pour AdaBoost Regression
            else:  # Classification
                if model_name == "Logistic Regression":
                    # Manual input for Logistic Regression
                    st.write("Logistic Regression Parameters")
                    penalty = st.text_input("Penalty (l2/l1/elasticnet/none)", value="l2")
                    c_value = st.number_input("Regularization Strength (C)", min_value=0.0, value=1.0, format="%.3f")
                    solver = st.text_input("Solver (lbfgs/newton-cg/liblinear/sag/saga)", value="lbfgs")
                    multi_class = st.text_input("Multi-class Strategy (auto/ovr/multinomial)", value="auto")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(
                            penalty=penalty,
                            C=c_value,
                            solver=solver,
                            multi_class=multi_class
                        ))
                    ])

                elif model_name == "KNN Classification":
                    # Manual input for KNN Classification
                    st.write("KNN Classification Parameters")
                    n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
                    weights = st.text_input("Weights (uniform/distance)", value="uniform")
                    algorithm = st.text_input("Algorithm (auto/ball_tree/kd_tree/brute)", value="auto")
                    metric = st.text_input("Distance Metric", value="minkowski")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            metric=metric
                        ))
                    ])

                elif model_name == "XGBoost Classification":
                    # Manual input for XGBoost Classification
                    st.write("XGBoost Classification Parameters")
                    n_estimators = st.number_input("Number of Estimators", min_value=1, value=100)
                    learning_rate = st.number_input("Learning Rate", min_value=0.0, value=0.1, format="%.3f")
                    max_depth = st.number_input("Max Depth", min_value=1, value=6)
                    subsample = st.number_input("Subsample", min_value=0.0, max_value=1.0, value=0.8, format="%.2f")

                    model = ImbPipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', xgb.XGBClassifier(
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            max_depth=int(max_depth),
                            subsample=subsample
                        ))
                    ])




        # Main model selection and training section
        if not train.empty and features and target:
            # Model type selection
            model_types = ["Regression", "Classification"]
            model_type = st.selectbox(label="Select Model Type", options=model_types)

            # Expanded model lists
            if model_type == "Regression":
                regression_models = [
                    "Linear Regression",
                    "KNN Regression",
                    "XGBoost Regression",
                    "SVM Regression",
                    "AdaBoost Regression",
                    "Random Forest Regression",
                    "Gradient Boosting Regression",
                    "LightGBM Regression"
                ]
                model_sel = st.selectbox(label="Select Regression Model", options=regression_models)
            else:
                classification_models = [
                    "Logistic Regression",
                    "KNN Classification",
                    "XGBoost Classification",
                    "SVM Classification",
                    "AdaBoost Classification",
                    "Random Forest Classification",
                    "Gradient Boosting Classification",
                    "LightGBM Classification"
                ]
                model_sel = st.selectbox(label="Select Classification Model", options=classification_models)

            # Create model with manually input parameters
            model = create_model_with_manual_params(
                model_type=model_type,
                model_name=model_sel,
                preprocessor=preprocessor
            )
            # Train model button
            if st.button("Train Model"):
                try:
                    model.fit(x, y)
                    st.success("Model trained successfully!")

                    # Model evaluation section
                    st.header("Model Evaluation")

                    # Get predictions on training data for initial evaluation
                    train_predictions = model.predict(x)

                    # Check if target is categorical or continuous
                    is_classification = len(
                    np.unique(y)) < 10 or y.dtype == 'bool' or pd.api.types.is_categorical_dtype(y)

                    # Display evaluation metrics based on problem type
                    if is_classification:
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                                confusion_matrix, classification_report, roc_auc_score, roc_curve
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        # Calculate metrics on training data
                        st.subheader("Training Data Metrics")
                        accuracy = accuracy_score(y, train_predictions)

                        try:
                            # Only for binary classification
                            if len(np.unique(y)) == 2:
                                precision = precision_score(y, train_predictions)
                                recall = recall_score(y, train_predictions)
                                f1 = f1_score(y, train_predictions)

                                # Display metrics in columns
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{accuracy:.4f}")
                                col2.metric("Precision", f"{precision:.4f}")
                                col3.metric("Recall", f"{recall:.4f}")
                                col4.metric("F1 Score", f"{f1:.4f}")

                                # Display confusion matrix
                                st.subheader("Confusion Matrix (Training Data)")
                                cm = confusion_matrix(y, train_predictions)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)

                                # ROC Curve if the model has predict_proba
                                if hasattr(model, 'predict_proba'):
                                    st.subheader("ROC Curve (Training Data)")
                                    y_proba = model.predict_proba(x)[:, 1]
                                    fpr, tpr, _ = roc_curve(y, y_proba)
                                    auc = roc_auc_score(y, y_proba)

                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
                                    ax.plot([0, 1], [0, 1], 'k--')
                                    ax.set_xlabel('False Positive Rate')
                                    ax.set_ylabel('True Positive Rate')
                                    ax.set_title('ROC Curve')
                                    ax.legend(loc='lower right')
                                    st.pyplot(fig)
                            else:
                                # Multiclass classification
                                st.metric("Accuracy", f"{accuracy:.4f}")

                                # Classification report
                                st.subheader("Classification Report")
                                report = classification_report(y, train_predictions, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df)

                                # Display confusion matrix
                                st.subheader("Confusion Matrix (Training Data)")
                                cm = confusion_matrix(y, train_predictions)
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not calculate some metrics: {e}")
                            st.metric("Accuracy", f"{accuracy:.4f}")

                    else:
                        # Regression metrics
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        import matplotlib.pyplot as plt

                        st.subheader("Training Data Metrics")
                        mse = mean_squared_error(y, train_predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y, train_predictions)
                        r2 = r2_score(y, train_predictions)

                        # Display metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MSE", f"{mse:.4f}")
                        col2.metric("RMSE", f"{rmse:.4f}")
                        col3.metric("MAE", f"{mae:.4f}")
                        col4.metric("R² Score", f"{r2:.4f}")

                        # Actual vs Predicted plot
                        st.subheader("Actual vs Predicted (Training Data)")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y, train_predictions, alpha=0.5)
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title('Actual vs Predicted Values')
                        st.pyplot(fig)

                        # Residuals plot
                        st.subheader("Residuals Plot (Training Data)")
                        residuals = y - train_predictions
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(train_predictions, residuals, alpha=0.5)
                        ax.axhline(y=0, color='r', linestyle='--')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Residuals')
                        ax.set_title('Residuals Plot')
                        st.pyplot(fig)

                    # Test data evaluation if available
                    if test is not None and not test.empty and target in test.columns:
                        st.header("Test Data Evaluation")
                        X_test = test[features]
                        y_test = test[target]
                        test_predictions = model.predict(X_test)

                        if is_classification:
                            # Classification metrics
                            test_accuracy = accuracy_score(y_test, test_predictions)

                            try:
                                if len(np.unique(y_test)) == 2:
                                    test_precision = precision_score(y_test, test_predictions)
                                    test_recall = recall_score(y_test, test_predictions)
                                    test_f1 = f1_score(y_test, test_predictions)

                                    # Display metrics in columns
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Test Accuracy", f"{test_accuracy:.4f}")
                                    col2.metric("Test Precision", f"{test_precision:.4f}")
                                    col3.metric("Test Recall", f"{test_recall:.4f}")
                                    col4.metric("Test F1 Score", f"{test_f1:.4f}")

                                    # Test confusion matrix
                                    st.subheader("Confusion Matrix (Test Data)")
                                    test_cm = confusion_matrix(y_test, test_predictions)
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                                    ax.set_xlabel('Predicted')
                                    ax.set_ylabel('Actual')
                                    st.pyplot(fig)

                                    # Test ROC Curve if the model has predict_proba
                                    if hasattr(model, 'predict_proba'):
                                        st.subheader("ROC Curve (Test Data)")
                                        test_proba = model.predict_proba(X_test)[:, 1]
                                        test_fpr, test_tpr, _ = roc_curve(y_test, test_proba)
                                        test_auc = roc_auc_score(y_test, test_proba)

                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        ax.plot(test_fpr, test_tpr, label=f'AUC = {test_auc:.4f}')
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Test Data)')
                                        ax.legend(loc='lower right')
                                        st.pyplot(fig)
                                else:
                                    # Multiclass
                                    st.metric("Test Accuracy", f"{test_accuracy:.4f}")

                                    # Test classification report
                                    st.subheader("Classification Report (Test Data)")
                                    test_report = classification_report(y_test, test_predictions, output_dict=True)
                                    test_report_df = pd.DataFrame(test_report).transpose()
                                    st.dataframe(test_report_df)
                            except Exception as e:
                                st.warning(f"Could not calculate some test metrics: {e}")
                                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                        else:
                            # Regression test metrics
                            test_mse = mean_squared_error(y_test, test_predictions)
                            test_rmse = np.sqrt(test_mse)
                            test_mae = mean_absolute_error(y_test, test_predictions)
                            test_r2 = r2_score(y_test, test_predictions)

                            # Display metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Test MSE", f"{test_mse:.4f}")
                            col2.metric("Test RMSE", f"{test_rmse:.4f}")
                            col3.metric("Test MAE", f"{test_mae:.4f}")
                            col4.metric("Test R² Score", f"{test_r2:.4f}")

                            # Test Actual vs Predicted plot
                            st.subheader("Actual vs Predicted (Test Data)")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(y_test, test_predictions, alpha=0.5)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.set_title('Actual vs Predicted Values (Test Data)')
                            st.pyplot(fig)
                    elif test is not None and not test.empty:
                        # If test data exists but no target column
                        st.header("Test Data Predictions")
                        X_test = test[features]
                        test_predictions = model.predict(X_test)

                        # Show predictions
                        pred_df = test.copy()
                        pred_df['Predicted'] = test_predictions
                        st.write("Test Data with Predictions:")
                        st.dataframe(pred_df.head(10))

                        # Download predictions button
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Download Test Predictions as CSV",
                            data=csv,
                            file_name="test_predictions.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"Error during model training or evaluation: {e}")
        # Requirements note
        st.sidebar.info("""
            Required Libraries:
            - streamlit
            - pandas
            - scikit-learn
            - xgboost
            - lightgbm
            - imbalanced-learn
        """)
