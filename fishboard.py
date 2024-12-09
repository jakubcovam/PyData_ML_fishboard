import functools
import io
import logging
import pathlib
from typing import List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from altair import value
from pyarrow.dataset import dataset
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from streamlit.delta_generator import DeltaGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

footer_html = """
<style>.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgb(211, 211, 211);
    color: black;
    text-align: center;
}
</style>
<div class='footer'>
   By <a href="https://github.com/jakubcovam/" target="_blank">jakubcovam</a> using Streamlit.
</div>
 """


# typ pro Streamlit kontejnery
StContainer = DeltaGenerator


# adresář s daty
DATA_DIR = pathlib.Path("data")
DATA_FILES = {
    "Fishes 🐟🐠🐡 - regression": {
        "filename": "fish_data.csv",
        "type": "regression"
    },
    "Penguins 🐧🐧🐧 - regression": {
        "filename": "penguins_size_nona.csv",
        "type": "regression"
    },
    "Iris 🌻🌺🌼 - classification": {
        "filename": "Iris.csv",
        "type": "classification"
    },
}

# slovník s názvy modelů pro regresi
# u každého modelu je třeba definovat class - třídu, která se použije
# a hyperparams, který obsahuje slovník názvů hyperparametrů a funkcí pro vytvoření streamlit widgetu
# předpokládá se, že třídy mají scikit-learn API
REGRESSION_MODELS = {
    "LinearRegression": {
        "class": LinearRegression,
        "hyperparams": {},
    },
    "Lasso": {
        "class": Lasso,
        "hyperparams": {
            "alpha": functools.partial(st.slider, "alpha", 0.0, 1.0, 0.0)
        },
    },
    "SVR": {
        "class": SVR,
        "hyperparams": {
            "kernel": functools.partial(
                st.selectbox, "kernel", ["linear", "poly", "rbf", "sigmoid"], index=2
            ),
            "C": functools.partial(st.number_input, "C", 0.0, None, 1.0),
        },
    },
}

CLASSIFICATION_MODELS = {
    "KNeighbors": {
        "class": KNeighborsClassifier,
        "hyperparams": {
            "n_neighbors": functools.partial(st.slider, "n_neighbors", 1, 10, 1, step=1)
        },
    },
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "hyperparams": {
            "max_depth": functools.partial(st.slider, "max_depth", 1, 20, 5, step=1)
        },
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "hyperparams": {
            "n_estimators": functools.partial(st.slider, "n_estimators", 1, 100, 5, step=1)
        },
    },
    "SVC": {
        "class": SVC,
        "hyperparams": {
            "C": functools.partial(st.slider, "C", 1, 1000, 100, step=1),
        "kernel": functools.partial(
                st.selectbox, "kernel", ["linear", "poly", "rbf", "sigmoid"], index=2
            ),
        },
    },
}


# názvy metrik a příslušné funkce pro výpočet
METRICS = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}
METRICS_CLASSIFICATION  = {"Precision": precision_score, "Recall": recall_score, "F1": f1_score}


@st.cache_data
def load_data(csv_file: Union[str, pathlib.Path, io.IOBase]) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col=0)


@st.cache_data
def preprocess(
    data: pd.DataFrame, drop_columns: Optional[List] = None, get_dummies: bool = False
) -> pd.DataFrame:
    if drop_columns:
        data = data.drop(columns=drop_columns)
    if get_dummies:
        data = pd.get_dummies(data)
    return data


def regression(
    col1: StContainer,
    col2: StContainer,
    learning_data: pd.DataFrame,
    target: str,
    test_size: float,
    stratify: str,
) -> None:
    """Regrese v dashboardu"""

    # rozdělení na trénovací a testovací data
    y = learning_data[target]
    X = learning_data.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)

    with col1.expander("Model selection", expanded=True):
        model = st.selectbox("Regression model", list(REGRESSION_MODELS))
        # hodnoty hyperparametrů si uložíme do slovníku typu {jméno hyperparametru: hodnota}
        hyperparams = {
            hyperparam: widget() for hyperparam, widget in REGRESSION_MODELS[model]["hyperparams"].items()
        }
    with col1.expander("Accuracy criteria", expanded=True):
        metric = st.selectbox("Metric", list(METRICS))

    # REGRESSION_MODELS[model]["class"] vrací třídu regresoru, např. LinearRegression
    # ve slovníku hyperparams máme uložené hodnoty hyperparametrů od uživatele
    # takto tedy můžeme vytvořit příslušný regresor
    regressor = REGRESSION_MODELS[model]["class"](**hyperparams)
    # zkusíme natrénovat model
    try:
        regressor.fit(X_train, y_train)
    except Exception as prediction_error:
        # v případě chyby ukážeme uživateli co se stalo
        st.error(f"Model fitting error: {prediction_error}")
        # a nebudeme už nic dalšího zobrazovat
        return

    # predikce pomocí natrénovaného modelu
    y_predicted = regressor.predict(X_test)
    prediction_error = METRICS[metric](y_predicted, y_test)

    col2.header(f"Scatter plot: {model}")
    col2.write(f"{metric}: {prediction_error:.3g}")

    # vytvoříme pomocný dataframe se sloupcem s predikcí
    predicted_target_column = f"{target} - predicted"
    complete_data = learning_data.assign(**{predicted_target_column: regressor.predict(X)})
    # vykreslíme správné vs predikované body
    fig = px.scatter(
        complete_data,
        x=target,
        y=predicted_target_column,
    ).update_traces(marker=dict(color='red'))
    # přidáme čáru ukazující ideální predikci
    fig.add_trace(
        go.Scatter(
            x=[complete_data[target].min(), complete_data[target].max()],
            y=[complete_data[target].min(), complete_data[target].max()],
            mode="lines",
            line=dict(width=2, color="DarkSlateGrey"),
            name="Ideal prediction",
        )
    )
    col2.write(fig)


def classification(
    col1: StContainer,
    col2: StContainer,
    learning_data: pd.DataFrame,
    target: str,
    test_size: float,
    stratify: str,
) -> None:
    """Klasifikace v dashboardu"""
    # rozdělení na trénovací a testovací data
    y = learning_data[target]
    X = learning_data.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)
    with col1.expander("Model selection", expanded=True):
        model = st.selectbox("Classification model", list(CLASSIFICATION_MODELS))
        # hodnoty hyperparametrů si uložíme do slovníku typu {jméno hyperparametru: hodnota}
        hyperparams = {
            hyperparam: widget() for hyperparam, widget in CLASSIFICATION_MODELS[model]["hyperparams"].items()
        }
    with col1.expander("Accuracy criteria", expanded=True):
        metric = st.selectbox("Metric", list(METRICS_CLASSIFICATION))

    # REGRESSION_MODELS[model]["class"] vrací třídu regresoru, např. LinearRegression
    # ve slovníku hyperparams máme uložené hodnoty hyperparametrů od uživatele
    # takto tedy můžeme vytvořit příslušný regresor
    klasifikator = CLASSIFICATION_MODELS[model]["class"](**hyperparams)
    # zkusíme natrénovat model
    try:
        klasifikator.fit(X_train, y_train)
    except Exception as prediction_error:
        # v případě chyby ukážeme uživateli co se stalo
        st.error(f"Model fitting error: {prediction_error}")
        # a nebudeme už nic dalšího zobrazovat
        return

        # predikce pomocí natrénovaného modelu
    y_predicted = klasifikator.predict(X_test)
    prediction_error = METRICS_CLASSIFICATION[metric](y_test, y_predicted)

    col2.header(f"Confusion matrix: {model}")
    col2.write(f"{metric}: {prediction_error:.3g}")

    # zobraz matici záměn
    cm = confusion_matrix(y_predicted, y_test)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )
    display.plot()
    col2.pyplot(plt.gcf())

def main() -> None:
    # základní vlastnosti aplikace: jméno, široké rozložení
    st.set_page_config(page_title="PyData", layout="wide")
    st.title("Regression and classification problems")
    st.header("Exploratory data analysis and machine learning models")

    # add horizontal rule
    st.divider()

    # použijeme dva sloupce
    col1, col2 = st.columns(2)

    with col1.expander("Data selection", expanded=True):
        datas = st.selectbox("Dataset", list(DATA_FILES.keys()))
        # Extract filename and data type
        selected_file = DATA_FILES[datas]["filename"]
        data_type = DATA_FILES[datas]["type"]
    source_data = load_data(DATA_DIR / selected_file)

    with col1.expander("Exploratory data analysis", expanded=True):
        dist_plot_type = st.selectbox("Type of plot", ["boxplot", "histogram", "violin"])
        color = st.selectbox("Color for plots", source_data.columns)
        use_color = st.checkbox(f"Use color ({color})", value=True)
        target_plot = st.selectbox("Column for analysis", source_data.columns)

    col2.header(f"EDA: {dist_plot_type}")

    with col2:
        if dist_plot_type == "boxplot":
            st.write(px.box(source_data, x=target_plot, color=color if use_color else None))
        elif dist_plot_type == "histogram":
            st.write(px.histogram(source_data, x=target_plot, color=color if use_color else None))
        elif dist_plot_type == "violin":
            st.write(px.violin(source_data, x=target_plot, color=color if use_color else None))
        else:
            st.error("Invalid plot type")

    with col1.expander("Preprocessing for machine learning models", expanded=True):
        drop_columns = st.multiselect("Drop columns", source_data.columns)
        get_dummies = st.checkbox("Get dummies", value=True)
    learning_data = preprocess(source_data, drop_columns, get_dummies)
    target = col1.selectbox("Response column", learning_data.columns)

    with col1.expander("Display data"):
        display_preprocessed = st.checkbox("Display preprocessed data", value=False)
        if display_preprocessed:
            displayed_data = learning_data
            # st.dataframe(displayed_data)
        else:
            displayed_data = source_data
            # st.dataframe(displayed_data)
        st.dataframe(displayed_data)

    with col1.expander("Splitting into test and training data", expanded=True):
        test_size = st.slider("Test set ratio", 0.0, 1.0, 0.25, 0.05)
        #stratify_column = st.selectbox("Stratify", [None] + list(source_data.columns))
    #if stratify_column is not None:
    #    stratify = source_data[stratify_column]
    #else:
    #    stratify = None

    stratify = None
    if data_type == "regression":
        regression(col1, col2, learning_data, target, test_size, stratify)
    elif data_type == "classification":
        classification(col1, col2, learning_data, target, test_size, stratify)

    # Render the footer
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    logging.basicConfig()
    main()
