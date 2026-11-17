
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PROFESSIONAL THEME-AWARE CSS
# ==========================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* Main page spacing */
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    border-right: 1px solid var(--border-color);
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-color);
    font-weight: 700;
}

/* Sidebar navigation buttons */
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    text-align: left;
    border: 1px solid var(--border-color);
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
}

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border-color);
    border-radius: 12px;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    border-radius: 8px;
    font-weight: 600;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    border-radius: 7px;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Section divider */
hr {
    border-color: var(--border-color);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DATASET LOADING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "credit_risk_dataset.csv"


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


if not DATA_FILE.exists():
    st.error("Dataset file not found!")
    st.info(
        "Place credit_risk_dataset.csv in the same folder "
        "as app.py."
    )
    st.stop()


try:
    df = load_data(str(DATA_FILE))
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


TARGET = "loan_status"

if TARGET not in df.columns:
    st.error("The dataset must contain a loan_status column.")
    st.stop()


# Remove rows without a target value
df = df.dropna(subset=[TARGET]).copy()

# Ensure target is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

if set(df[TARGET].unique()) != {0, 1}:
    st.error("The loan_status column must contain 0 and 1.")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

pages = [
    "Dashboard",
    "Data Analysis",
    "Model Training",
    "Prediction",
    "About Project"
]

with st.sidebar:

    st.markdown("# CREDIT RISK")
    st.markdown("**ANALYSIS PLATFORM**")

    st.divider()

    st.markdown("### NAVIGATION")

    for page in pages:

        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            )
        ):
            st.session_state.page = page
            st.rerun()

    st.divider()

    st.caption("Credit Risk Classification")
    st.caption("Machine Learning Application")


page = st.session_state.page


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_features(data):
    return data.drop(columns=[TARGET])


def train_models(data):

    X = get_features(data)
    y = data[TARGET]

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    trained = {}
    results = {}

    for name, algorithm in algorithms.items():

        model = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", algorithm)
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC AUC": roc_auc_score(
                y_test, probabilities
            ),
            "Predictions": predictions,
            "Probabilities": probabilities,
            "Actual": y_test,
            "Confusion Matrix": confusion_matrix(
                y_test, predictions
            ),
            "Report": classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        }

        trained[name] = model

    return trained, results


def display_metrics(data):

    col1, col2, col3, col4 = st.columns(4)

    total = len(data)

    defaults = int((data[TARGET] == 1).sum())

    non_defaults = int((data[TARGET] == 0).sum())

    with col1:
        st.metric("Total Applications", f"{total:,}")

    with col2:
        st.metric("Default Applications", f"{defaults:,}")

    with col3:
        st.metric("Non-Default Applications", f"{non_defaults:,}")

    with col4:
        st.metric(
            "Default Rate",
            f"{(defaults / total) * 100:.2f}%"
        )


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.title("Credit Risk Dashboard")

    st.write(
        "Analyze loan applications and understand "
        "credit default patterns using Machine Learning."
    )

    st.divider()

    display_metrics(df)

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Preview**")
        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    with col2:
        st.write("**Loan Status Distribution")

        status_counts = df[TARGET].value_counts().sort_index()
        status_counts.index = [
            "Non-Default (0)" if x == 0 else "Default (1)"
            for x in status_counts.index
        ]

        st.bar_chart(status_counts)

    st.divider()

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")

    with col3:
        missing = int(df.isnull().sum().sum())
        st.metric("Missing Values", f"{missing:,}")

    st.subheader("Key Dataset Details")

    st.write("**Target Variable:** loan_status")

    st.write("""
    - 0 represents a non-default loan.
    - 1 represents a default loan.
    - The model learns patterns from the available applicant information.
    """)


# ==========================================
# DATA ANALYSIS PAGE
# ==========================================

elif page == "Data Analysis":

    st.title("Data Analysis")

    st.write(
        "Explore the dataset, identify missing values "
        "and understand relationships between features."
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    missing_values = missing_values[missing_values > 0]

    if len(missing_values) > 0:
        st.bar_chart(missing_values)
    else:
        st.success("No missing values found.")

    st.subheader("Loan Status Distribution")

    status_counts = df[TARGET].value_counts().sort_index()

    status_counts.index = [
        "Non-Default (0)" if x == 0 else "Default (1)"
        for x in status_counts.index
    ]

    st.bar_chart(status_counts)

    numeric_df = df.select_dtypes(include="number")

    st.subheader("Correlation Analysis")

    if numeric_df.shape[1] >= 2:

        correlation = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))

        image = ax.imshow(
            correlation,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.columns)))

        ax.set_xticklabels(
            correlation.columns,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(correlation.columns)

        fig.colorbar(image, ax=ax)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Not enough numerical columns for correlation analysis.")

    st.subheader("Numerical Feature Distribution")

    numeric_columns = numeric_df.columns.tolist()

    selected_column = st.selectbox(
        "Select a numerical feature",
        numeric_columns
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(
        df[selected_column].dropna(),
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {selected_column}")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# MODEL TRAINING PAGE
# ==========================================

elif page == "Model Training":

    st.title("Machine Learning Model Training")

    st.write(
        "Train and compare three classification algorithms "
        "to predict loan default."
    )

    st.divider()

    st.subheader("Algorithms Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Logistic Regression**")
        st.write("A classification algorithm used to estimate the probability of default.")

    with col2:
        st.markdown("**Decision Tree**")
        st.write("A tree-based algorithm that learns decision rules.")

    with col3:
        st.markdown("**Random Forest**")
        st.write("An ensemble algorithm that combines multiple decision trees.")

    st.divider()

    st.subheader("Train Models")

    st.write(
        "The dataset is divided into 80% training data "
        "and 20% testing data."
    )

    if st.button(
        "Train Machine Learning Models",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Training models. Please wait..."):

                trained, results = train_models(df)

                st.session_state.trained_models = trained
                st.session_state.model_results = results

            st.success("All three models trained successfully!")

        except Exception as error:
            st.error(f"Model training failed: {error}")

    if st.session_state.model_results is not None:

        results = st.session_state.model_results

        st.divider()

        st.subheader("Model Performance Comparison")

        comparison = pd.DataFrame({
            name: {
                metric: values[metric]
                for metric in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC AUC"
                ]
            }
            for name, values in results.items()
        }).T

        st.dataframe(
            comparison.style.format("{:.4f}"),
            use_container_width=True
        )

        st.subheader("Accuracy Comparison")

        st.bar_chart(comparison["Accuracy"])

        best_model = comparison["F1 Score"].idxmax()

        st.success(
            f"Best model based on F1 Score: {best_model}"
        )

        st.subheader("Confusion Matrix")

        selected_model = st.selectbox(
            "Select model",
            list(results.keys())
        )

        matrix = results[selected_model]["Confusion Matrix"]

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.imshow(matrix, cmap="Blues")

        ax.set_title(f"{selected_model} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Non-Default", "Default"])
        ax.set_yticklabels(["Non-Default", "Default"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="black"
                )

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")

        st.code(
            results[selected_model]["Report"],
            language="text"
        )

    else:
        st.info(
            "Click the Train Machine Learning Models button "
            "to train and evaluate the algorithms."
        )


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Prediction":

    st.title("Credit Risk Prediction")

    st.write(
        "Enter applicant information to estimate "
        "the probability of loan default."
    )

    st.divider()

    if st.session_state.trained_models is None:

        st.warning(
            "Please visit Model Training and train the models first."
        )

    else:

        trained_models = st.session_state.trained_models

        selected_model = st.selectbox(
            "Choose Machine Learning Model",
            list(trained_models.keys())
        )

        X = get_features(df)

        st.subheader("Applicant Information")

        applicant = {}

        with st.form("prediction_form"):

            columns = st.columns(2)

            for index, column in enumerate(X.columns):

                with columns[index % 2]:

                    if pd.api.types.is_numeric_dtype(X[column]):

                        median_value = X[column].median()

                        if pd.isna(median_value):
                            median_value = 0

                        min_value = X[column].min()
                        max_value = X[column].max()

                        if pd.isna(min_value):
                            min_value = 0

                        if pd.isna(max_value):
                            max_value = 100

                        min_value = float(min_value)
                        max_value = float(max_value)

                        if min_value == max_value:
                            min_value -= 1
                            max_value += 1

                        applicant[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=min_value,
                            max_value=max_value,
                            value=float(median_value),
                            key=f"input_{column}"
                        )

                    else:

                        options = (
                            X[column]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        options = sorted(options)

                        if not options:
                            options = ["Unknown"]

                        applicant[column] = st.selectbox(
                            column.replace("_", " ").title(),
                            options,
                            key=f"input_{column}"
                        )

            submitted = st.form_submit_button(
                "Predict Credit Risk",
                use_container_width=True,
                type="primary"
            )

        if submitted:

            try:

                input_df = pd.DataFrame(
                    [applicant],
                    columns=X.columns
                )

                model = trained_models[selected_model]

                prediction = model.predict(input_df)[0]

                probability = model.predict_proba(input_df)[0][
                    list(model.classes_).index(1)
                ]

                st.divider()

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Default Probability",
                        f"{probability * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Non-Default Probability",
                        f"{(1 - probability) * 100:.2f}%"
                    )

                if prediction == 1:
                    st.error(
                        "Model Prediction: Potential Loan Default"
                    )
                else:
                    st.success(
                        "Model Prediction: Non-Default"
                    )

                st.caption(
                    "This is an educational ML prediction, "
                    "not a real lending or credit approval decision."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("About the Project")

    st.write(
        "Credit Risk Analysis is a Machine Learning project "
        "that studies loan applicant information and predicts "
        "the possibility of loan default."
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        "The objective is to build classification models "
        "that identify whether a loan applicant may default."
    )

    st.subheader("Technology Stack")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Scikit-learn",
            "Streamlit"
        ],
        "Purpose": [
            "Programming language",
            "Data analysis",
            "Numerical operations",
            "Data visualization",
            "Machine Learning",
            "Web application"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Machine Learning Workflow")

    workflow = [
        "Dataset collection",
        "Data preprocessing",
        "Exploratory Data Analysis",
        "Feature preparation",
        "Model training",
        "Model evaluation",
        "Credit risk prediction"
    ]

    for number, item in enumerate(workflow, start=1):
        st.write(f"**{number}.** {item}")

    st.divider()

    st.subheader("Project Disclaimer")

    st.write(
        "This application is developed for educational "
        "and academic purposes. Its predictions should not "
        "be used as the sole basis for actual financial decisions."
    )

    st.caption("Credit Risk Analysis | Machine Learning Project")
