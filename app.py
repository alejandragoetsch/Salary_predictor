import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


PALETTE = [
    "#916b5e",
    "#bab78c",
    "#ecdfcd",
    "#be8c6b",
    "#c2a88f",
    "#686961",
    "#a4927a",
    "#604a33",
    "#8e6c49",
    "#879281",
]

BG_BODY = "#ecdfcd"   
TEXT_COLOR = "#604a33" 
PLOT_BG = "#f5eee2"

pio.templates.default = "plotly_white"

COMMON_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT_COLOR),
    margin=dict(l=50, r=40, t=60, b=50),
    xaxis=dict(showgrid=True, zeroline=False),
    yaxis=dict(showgrid=True, zeroline=False),
)

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": f"1px solid {PALETTE[7]}",
    "borderRadius": "8px",
    "padding": "10px 16px",
    "minWidth": "220px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
}


def categorize_job_title(title):
    t = str(title).lower()
    if "junior" in t:
        return "Junior"
    if "senior" in t:
        return "Senior"
    if "director" in t:
        return "Director"
    if "manager" in t:
        return "Manager"
    if "analyst" in t or "data" in t:
        return "Analyst"
    if "engineer" in t:
        return "Engineer"
    if "developer" in t or "software" in t:
        return "Software developer"
    if "scientist" in t:
        return "Scientist"
    if "assistant" in t:
        return "Assistant"
    if "driver" in t:
        return "Driver"
    return "Other"


def normalize_education_level(x):
    t = str(x).strip().lower()
    if "high school" in t:
        return "High School"
    if "bachelor" in t:
        return "Bachelor's degree"
    if "master" in t:
        return "Master's degree"
    if "phd" in t or "doctor" in t:
        return "PhD"
    return "Other"


def load_data():
    df = pd.read_csv("Salary_Data.csv")

    df["Job Category"] = df["Job Title"].apply(categorize_job_title)
    df["Education Level"] = df["Education Level"].apply(normalize_education_level)

    for col in ["Age", "Salary", "Years of Experience"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    age_bins = [18, 29, 39, 60]
    age_labels = ["18–29", "30–39", "40–60"]
    df["Age Range"] = pd.cut(
        df["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
        right=True,
    )

    return df


def load_world_salary_data():
    """
    Carga el dataset global con columnas:
    country_name, continent_name, wage_span, median_salary, ...
    """
    dfw = pd.read_csv("world_salary.csv")  
    dfw = dfw[["country_name", "continent_name", "median_salary"]].dropna()
    return dfw


df_world = load_world_salary_data()


def encode_for_model(df):
    df = df.copy()
    mappings = {}

    for col in ["Gender", "Education Level", "Job Category"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, mappings


def train_model(df_model):
    X = df_model[
        ["Age", "Gender", "Education Level", "Years of Experience", "Job Category"]
    ]
    y = df_model["Salary"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    df_y = pd.DataFrame({"Actual Salary": y_test, "Predicted Salary": y_pred})
    return model, df_y, rmse, r2


df_raw = load_data()
df_model, mappings = encode_for_model(df_raw)
model, df_y, rmse, r2 = train_model(df_model)
mean_salary = df_model["Salary"].mean()

print("=== Modelo cargado ===")
print("RMSE:", rmse, "| R2:", r2)


def fig_box_edu(df):
    fig = px.box(
        df,
        x="Education Level",
        y="Salary",
        title="Salary by Education Level",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Education Level",
        yaxis_title="Salary (USD)",
    )
    return fig


def fig_world_map(dfw):
    fig = px.choropleth(
        dfw,
        locations="country_name",
        locationmode="country names",
        color="median_salary",
        title="Median Monthly Salary by Country",
        color_continuous_scale=[PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[7]],
        labels={"median_salary": "Median salary (USD/month)"},
    )
    fig.update_layout(
        **COMMON_LAYOUT,
        height=500,
    )
    return fig


def fig_world_continents(dfw):
    continent_mean = (
        dfw.groupby("continent_name")["median_salary"].mean().reset_index()
    )

    fig = px.bar(
        continent_mean,
        x="continent_name",
        y="median_salary",
        title="Median Monthly Salary by Continent",
        color="median_salary",
        color_continuous_scale=[PALETTE[2], PALETTE[3], PALETTE[0], PALETTE[7]],
        labels={
            "continent_name": "Continent",
            "median_salary": "Median salary (USD/month)",
        },
    )
    fig.update_layout(
        **COMMON_LAYOUT,
        height=420,
    )
    return fig


def fig_edu_by_gender(df):
    counts = (
        df.groupby(["Gender", "Education Level"])
        .size()
        .reset_index(name="Count")
    )
    total_by_gender = counts.groupby("Gender")["Count"].transform("sum")
    counts["Percent"] = counts["Count"] / total_by_gender * 100

    fig = px.bar(
        counts,
        x="Percent",
        y="Gender",
        color="Education Level",
        orientation="h",
        title="Education level distribution by Gender",
        text=counts["Percent"].map(lambda x: f"{x:.1f}%"),
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Percentage of people",
        yaxis_title="Gender",
        barmode="stack",
    )
    fig.update_xaxes(range=[0, 100])
    return fig


def fig_salary_age(df):
    fig = px.scatter(
        df,
        x="Age",
        y="Salary",
        title="Salary by Age",
        color_discrete_sequence=[PALETTE[7]],
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Age",
        yaxis_title="Salary (USD)",
    )
    return fig


def fig_salary_exp(df):
    fig = px.scatter(
        df,
        x="Years of Experience",
        y="Salary",
        title="Salary by Years of Experience",
        color_discrete_sequence=[PALETTE[8]],
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Years of Experience",
        yaxis_title="Salary (USD)",
    )
    return fig


def fig_age_gender_salary(df):
    agg = (
        df.groupby(["Age Range", "Gender"])["Salary"]
        .sum()
        .reset_index()
        .dropna(subset=["Age Range"])
    )

    fig = px.bar(
        agg,
        x="Age Range",
        y="Salary",
        color="Gender",
        barmode="group",
        title="Total salary by Age Range and Gender",
        color_discrete_sequence=[PALETTE[0], PALETTE[3], PALETTE[9]],
    )
    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Age Range",
        yaxis_title="Total salary (USD)",
    )
    return fig


def fig_corr(df):
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Matrix",
        aspect="auto",
        color_continuous_scale="PuBu",
    )
    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Variables",
        yaxis_title="Variables",
    )
    return fig


def fig_actual_pred(df_y):
    fig = px.scatter(
        df_y,
        x="Actual Salary",
        y="Predicted Salary",
        title="Actual vs Predicted Salary",
        opacity=0.7,
        color_discrete_sequence=[PALETTE[0]],
    )

    min_val = min(df_y.min())
    max_val = max(df_y.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color=PALETTE[7], dash="dash"),
            name="y = x",
        )
    )

    fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="Actual Salary (USD)",
        yaxis_title="Predicted Salary (USD)",
    )
    return fig


app = dash.Dash(__name__)
server = app.server

TAB_STYLE = {
    "padding": "10px 20px",
    "backgroundColor": BG_BODY,
    "color": TEXT_COLOR,
    "border": "1px solid " + PALETTE[7],
    "fontWeight": "normal",
}

TAB_SELECTED_STYLE = {
    "padding": "10px 20px",
    "backgroundColor": "#ffffff",
    "color": TEXT_COLOR,
    "border": "2px solid " + PALETTE[7],
    "fontWeight": "bold",
}

app.layout = html.Div(
    [
        html.H1(
            "SALARY PREDICTOR",
            style={
                "color": TEXT_COLOR,
                "textAlign": "center",
                "letterSpacing": "2px",
                "marginBottom": "20px",
            },
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Introducción",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.H2(
                            "Bienvenido/a a la aplicación de análisis de salarios",
                            style={"color": TEXT_COLOR},
                        ),
                        html.P(
                            "Esta aplicación utiliza un dataset de salarios en Estados Unidos para "
                            "explorar cómo influyen variables como la edad, el nivel educativo, "
                            "el género y los años de experiencia en el salario. "
                            "Además, se apoya en datos globales de salarios para situar estos resultados en contexto."
                        ),
                        html.P(
                            "A continuación puedes ver cómo se distribuye el salario mediano mensual en el mundo "
                            "y qué continentes presentan niveles salariales más altos."
                        ),

                        html.Div(
                            dcc.Graph(figure=fig_world_map(df_world)),
                            style={"marginTop": "10px", "marginBottom": "30px"},
                        ),
                        html.Div(
                            dcc.Graph(figure=fig_world_continents(df_world)),
                            style={"width": "70%", "margin": "0 auto"},
                        ),
                    ],
                ),

                dcc.Tab(
                    label="Análisis de salarios",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.H2("Análisis de salarios"),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(figure=fig_box_edu(df_raw)),
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(figure=fig_edu_by_gender(df_raw)),
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "marginLeft": "4%",
                                        "verticalAlign": "top",
                                    },
                                ),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(figure=fig_salary_age(df_raw)),
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(figure=fig_salary_exp(df_raw)),
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "marginLeft": "4%",
                                        "verticalAlign": "top",
                                    },
                                ),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                        html.Div(
                            [
                                html.H3("Salario total por rango de edad y género"),
                                html.P(
                                    "Esta gráfica resume cuánto salario total se concentra en cada rango de edad "
                                    "y cómo se reparte entre hombres, mujeres y otras identidades de género."
                                ),
                                dcc.Graph(figure=fig_age_gender_salary(df_raw)),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                    ],
                ),

                dcc.Tab(
                    label="Predicción de salario",
                    style=TAB_STYLE,
                    selected_style=TAB_SELECTED_STYLE,
                    children=[
                        html.H2("Introduce tus datos"),

                        html.Div(
                            style={"display": "flex", "gap": "40px", "alignItems": "flex-start"},
                            children=[
                                html.Div(
                                    style={"width": "40%"},
                                    children=[
                                        html.Label("Edad:"),
                                        dcc.Input(id="age", type="number", style={"width": "100%"}),

                                        html.Br(),
                                        html.Br(),
                                        html.Label("Años de experiencia:"),
                                        dcc.Input(id="exp", type="number", style={"width": "100%"}),

                                        html.Br(),
                                        html.Br(),
                                        html.Label("Género:"),
                                        dcc.Dropdown(
                                            id="gender",
                                            options=[
                                                {"label": g, "value": g}
                                                for g in mappings["Gender"].keys()
                                            ],
                                            style={"width": "100%"},
                                        ),

                                        html.Br(),
                                        html.Label("Nivel educativo:"),
                                        dcc.Dropdown(
                                            id="edu",
                                            options=[
                                                {"label": e, "value": e}
                                                for e in mappings["Education Level"].keys()
                                            ],
                                            style={"width": "100%"},
                                        ),

                                        html.Br(),
                                        html.Label("Job Category:"),
                                        dcc.Dropdown(
                                            id="job",
                                            options=[
                                                {"label": j, "value": j}
                                                for j in mappings["Job Category"].keys()
                                            ],
                                            style={"width": "100%"},
                                        ),

                                        html.Br(),
                                        html.Button("Predecir", id="predict", n_clicks=0),
                                    ],
                                ),

                                html.Div(
                                    style={"width": "60%"},
                                    children=[
                                        html.H3("Resultados de la predicción"),
                                        html.Div(id="prediction-output"),
                                        html.Br(),
                                        dcc.Graph(
                                            id="comparison-bar",
                                            style={"height": "320px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        html.H3("Tu punto sobre Salary vs Age"),
                        dcc.Graph(id="pred-plot-solo", figure=fig_salary_age(df_raw)),
                    ],
                ),
            ],
            colors={
                "border": PALETTE[7],
                "primary": PALETTE[7],
                "background": BG_BODY,
            },
        ),
    ],
    style={
        "backgroundColor": BG_BODY,
        "color": TEXT_COLOR,
        "minHeight": "100vh",
        "padding": "20px 40px",
    },
)


@app.callback(
    Output("prediction-output", "children"),
    Output("pred-plot-solo", "figure"),
    Output("comparison-bar", "figure"),
    Input("predict", "n_clicks"),
    State("age", "value"),
    State("exp", "value"),
    State("gender", "value"),
    State("edu", "value"),
    State("job", "value"),
)
def predict_salary_callback(n, age, exp, gender, edu, job):
    fig_base = fig_salary_age(df_raw)

    empty_bar = go.Figure().update_layout(
        **COMMON_LAYOUT,
        xaxis_title="",
        yaxis_title="",
        annotations=[
            dict(
                text="Introduce tus datos y pulsa Predecir",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        ],
    )

    if n == 0:
        msg = html.P("Introduce datos y pulsa Predecir.")
        return msg, fig_base, empty_bar

    if None in (age, exp, gender, edu, job):
        msg = html.P("Faltan datos por completar.")
        return msg, fig_base, empty_bar

    encoded = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [mappings["Gender"][gender]],
            "Education Level": [mappings["Education Level"][edu]],
            "Years of Experience": [exp],
            "Job Category": [mappings["Job Category"][job]],
        }
    )

    pred_usd = float(model.predict(encoded)[0])
    diff = pred_usd - mean_salary
    diff_pct = diff / mean_salary * 100

    usd_to_eur = 0.92
    pred_eur = pred_usd * usd_to_eur
    spain_factor = 0.5
    pred_spain_eur = pred_eur * spain_factor

    fig_base.add_trace(
        go.Scatter(
            x=[age],
            y=[pred_usd],
            mode="markers",
            marker=dict(
                size=14,
                color=PALETTE[9],          
                line=dict(width=1.5, color="black"),
            ),
            name="Tu predicción",
        )
    )

    cards = html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
        children=[
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.Strong("Salario EE. UU."),
                    html.Br(),
                    html.Span(f"{pred_usd:,.0f} USD"),
                    html.Br(),
                    html.Small(f"≈ {pred_eur:,.0f} €"),
                ],
            ),
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.Strong("Media dataset EE. UU."),
                    html.Br(),
                    html.Span(f"{mean_salary:,.0f} USD"),
                    html.Br(),
                    html.Small(f"Diferencia: {diff:,.0f} USD ({diff_pct:+.1f} %)"),
                ],
            ),
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.Strong("Estimación equivalente en España"),
                    html.Br(),
                    html.Span(f"{pred_spain_eur:,.0f} €"),
                    html.Br(),
                    html.Small("Aproximando que se cobra ~la mitad que en EE. UU."),
                ],
            ),
        ],
    )

    comp_df = pd.DataFrame(
        {
            "Categoria": [
                "Tu salario en EE. UU.",
                "Media EE. UU. dataset",
                "Estimación España (€/año)",
            ],
            "Valor": [pred_usd, mean_salary, pred_spain_eur],
        }
    )

    comp_fig = px.bar(
        comp_df,
        x="Categoria",
        y="Valor",
        title="Comparativa de tu salario frente a la media y España",
        color="Categoria",
        color_discrete_sequence=[PALETTE[0], PALETTE[3], PALETTE[9]],
    )
    comp_fig.update_layout(
        **COMMON_LAYOUT,
        xaxis_title="",
        yaxis_title="Importe (USD / € aprox.)",
    )

    return cards, fig_base, comp_fig


if __name__ == "__main__":
    app.run(debug=True)
