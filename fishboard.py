import functools
import io
import logging
import pathlib
from typing import List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from streamlit.delta_generator import DeltaGenerator


# typ pro Streamlit kontejnery
StContainer = DeltaGenerator


# adresář s daty
DATA_DIR = pathlib.Path("data")


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
        "hyperparams": {"alpha": functools.partial(st.slider, "alpha", 0.0, 1.0, 0.0)},
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


# názvy metrik a příslušné funkce pro výpočet
METRICS = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}


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

    with col1.expander("Výběr modelu"):
        model = st.selectbox("Regresní model", list(REGRESSION_MODELS))
        # hodnoty hyperparametrů si uložíme do slovníku typu {jméno hyperparametru: hodnota}
        hyperparams = {
            hyperparam: widget() for hyperparam, widget in REGRESSION_MODELS[model]["hyperparams"].items()
        }
        metric = st.selectbox("Metrika", list(METRICS))

    # REGRESSION_MODELS[model]["class"] vrací třídu regresoru, např. LinearRegression
    # ve slovníku hyperparams máme uložené hodnoty hyperparametrů od uživatele
    # takto tedy můžeme vytvořit příslušný regresor
    regressor = REGRESSION_MODELS[model]["class"](**hyperparams)
    # zkusíme natrénovat model
    try:
        regressor.fit(X_train, y_train)
    except Exception as prediction_error:
        # v případě chyby ukážeme uživateli co se stalo
        st.error(f"Chyba při fitování modelu: {prediction_error}")
        # a nebudeme už nic dalšího zobrazovat
        return

    # predikce pomocí natrénovaného modelu
    y_predicted = regressor.predict(X_test)
    prediction_error = METRICS[metric](y_predicted, y_test)

    col2.header(f"Výsledek modelu {model}")
    col2.write(f"{metric}: {prediction_error:.3g}")

    # vytvoříme pomocný dataframe s se sloupcem s predikcí
    predicted_target_column = f"{target} - predicted"
    complete_data = learning_data.assign(**{predicted_target_column: regressor.predict(X)})
    # vykreslíme správné vs predikované body
    fig = px.scatter(complete_data, x=target, y=predicted_target_column)
    # přidáme čáru ukazující ideální predikci
    fig.add_trace(
        go.Scatter(
            x=[complete_data[target].min(), complete_data[target].max()],
            y=[complete_data[target].min(), complete_data[target].max()],
            mode="lines",
            line=dict(width=2, color="DarkSlateGrey"),
            name="ideal prediction",
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
    st.error("Tohle ještě chybí")


def main() -> None:
    # základní vlastnosti aplikace: jméno, široké rozložení
    st.set_page_config(page_title="Fishboard", layout="wide")
    st.title("Fishboard")

    # použijeme dva sloupce
    col1, col2 = st.columns(2)

    with col1.expander("Výběr dat"):
        # TODO použít file upload for načtení uživatelských dat
        st.write("Vstupní data jsou ze souboru fish_data.csv")
    source_data = load_data(DATA_DIR / "fish_data.csv")

    with col1.expander("Preprocessing"):
        drop_columns = st.multiselect("Drop columns", source_data.columns)
        get_dummies = st.checkbox("Get dummies")
    learning_data = preprocess(source_data, drop_columns, get_dummies)

    with col1.expander("Zobrazení dat"):
        display_preprocessed = st.checkbox("Zobrazit preprocesovaná data", value=False)
        if display_preprocessed:
            displayed_data = learning_data
            # st.dataframe(displayed_data)
        else:
            displayed_data = source_data
            # st.dataframe(displayed_data)
        # TODO přidat grafy
        st.dataframe(displayed_data)

    target = col1.selectbox("Sloupec s odezvou", learning_data.columns)

    with col1.expander("Rozdělení na testovací a trénovací data"):
        test_size = st.slider("Poměr testovací sady", 0.0, 1.0, 0.25, 0.05)
        stratify_column = st.selectbox("Stratify", [None] + list(source_data.columns))
    if stratify_column is not None:
        stratify = source_data[stratify_column]
    else:
        stratify = None

    regression(col1, col2, learning_data, target, test_size, stratify)


if __name__ == "__main__":
    logging.basicConfig()
    main()
