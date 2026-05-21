import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Prediccion de Ventas - Rossmann",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f7fb 0%, #eef2f7 100%);
        color: #1f2937;
    }

    .main-header {
        background: linear-gradient(135deg, #16324f 0%, #2563eb 100%);
        padding: 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px rgba(22, 50, 79, 0.18);
    }

    .main-header h1 {
        font-size: 2.25rem;
        margin-bottom: 0.45rem;
        font-weight: 760;
        letter-spacing: 0;
    }

    .main-header p {
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 1120px;
        color: #e8eef8;
    }

    .section-title {
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid #dbe4ef;
        color: #16324f;
        font-size: 1.35rem;
        font-weight: 760;
    }

    .summary-card,
    .metric-card,
    .guide-card,
    .control-panel,
    .info-box {
        background: #ffffff;
        border: 1px solid #dbe4ef;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }

    .summary-card {
        padding: 1.15rem;
        height: 100%;
    }

    .summary-card h4,
    .metric-card h4 {
        margin: 0;
        color: #2563eb;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
    }

    .summary-card p {
        margin-top: 0.45rem;
        margin-bottom: 0;
        font-size: 1rem;
        font-weight: 650;
        color: #111827;
    }

    .metric-card {
        padding: 1.1rem;
        min-height: 118px;
    }

    .metric-card h2 {
        margin-top: 0.35rem;
        margin-bottom: 0;
        color: #16324f;
        font-size: 1.45rem;
    }

    .info-box {
        border-left: 6px solid #2563eb;
        padding: 1.05rem 1.2rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    .success-box {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: #14532d;
        font-weight: 650;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    .control-panel {
        padding: 1.1rem 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    .guide-card {
        padding: 1rem 1.1rem;
        min-height: 130px;
    }

    .guide-card h4 {
        margin: 0 0 0.45rem 0;
        color: #16324f;
        font-size: 1rem;
    }

    .guide-card p {
        margin: 0;
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .small-note {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


RUTA_DATOS = "../01_data/datos_dashboard.csv"
RUTA_METRICAS = "../01_data/metricas_modelos.csv"
RUTA_ESCENARIOS = "../01_data/resumen_escenarios.csv"

COLUMNAS_DATOS = [
    "Date",
    "Real",
    "Prediccion_Prophet",
    "Prediccion_ARIMA",
    "Prediccion_LSTM",
    "Escenario_Base",
    "Escenario_Optimista",
    "Escenario_Pesimista",
]

COLUMNAS_METRICAS = ["Modelo", "RMSE", "MAE", "MAPE (%)"]


@st.cache_data
def cargar_csv(ruta):
    return pd.read_csv(ruta)


def mostrar_error_archivos():
    st.error(
        "No se encontro el archivo requerido. Verifique que los notebooks hayan sido "
        "ejecutados y que los CSV esten en la carpeta 01_data."
    )
    st.stop()


def validar_columnas(df, columnas_requeridas):
    return all(columna in df.columns for columna in columnas_requeridas)


def formato_numero(valor):
    if pd.isna(valor):
        return "No disponible"
    return f"{valor:,.2f}"


def formato_porcentaje(valor):
    if pd.isna(valor):
        return "No disponible"
    return f"{valor:,.2f}%"


def limpiar_datos(datos_originales):
    datos_limpios = datos_originales.copy()
    datos_limpios["Date"] = pd.to_datetime(datos_limpios["Date"], errors="coerce")
    datos_limpios = datos_limpios.dropna(subset=["Date"]).sort_values("Date")

    for columna in COLUMNAS_DATOS:
        if columna != "Date":
            datos_limpios[columna] = pd.to_numeric(datos_limpios[columna], errors="coerce")

    return datos_limpios


def limpiar_metricas(metricas_originales):
    metricas_limpias = metricas_originales.copy()
    for columna in ["RMSE", "MAE", "MAPE (%)"]:
        metricas_limpias[columna] = pd.to_numeric(metricas_limpias[columna], errors="coerce")
    return metricas_limpias


MAPA_PREDICCIONES = {
    "ARIMA": "Prediccion_ARIMA",
    "Prophet": "Prediccion_Prophet",
    "LSTM": "Prediccion_LSTM",
}


def obtener_metricas_modelo(metricas_limpias, modelo):
    fila_modelo = metricas_limpias[
        metricas_limpias["Modelo"].astype(str).str.lower() == modelo.lower()
    ]

    if fila_modelo.empty:
        return pd.Series({"Modelo": modelo, "RMSE": None, "MAE": None, "MAPE (%)": None})

    return fila_modelo.iloc[0]


def configurar_grafica(ax, titulo, eje_y="Ventas"):
    ax.set_title(titulo, fontsize=13, fontweight="bold", color="#16324f", pad=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(eje_y)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.xticks(rotation=35)
    plt.tight_layout()


def graficar_lineas(df, columnas, titulo):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    colores = {
        "Real": "#111827",
        "Prediccion_ARIMA": "#dc2626",
        "Prediccion_Prophet": "#2563eb",
        "Prediccion_LSTM": "#7c3aed",
        "Escenario_Base": "#2563eb",
        "Escenario_Optimista": "#16a34a",
        "Escenario_Pesimista": "#f59e0b",
    }
    etiquetas = {
        "Real": "Ventas reales",
        "Prediccion_ARIMA": "Prediccion ARIMA",
        "Prediccion_Prophet": "Prediccion Prophet",
        "Prediccion_LSTM": "Prediccion LSTM",
        "Escenario_Base": "Escenario base",
        "Escenario_Optimista": "Escenario optimista",
        "Escenario_Pesimista": "Escenario pesimista",
    }

    for columna in columnas:
        ax.plot(
            df["Date"],
            df[columna],
            label=etiquetas.get(columna, columna),
            color=colores.get(columna, "#334155"),
            linewidth=2.2,
        )

    configurar_grafica(ax, titulo)
    st.pyplot(fig)
    plt.close(fig)


try:
    datos = cargar_csv(RUTA_DATOS)
    metricas = cargar_csv(RUTA_METRICAS)
    resumen_escenarios = cargar_csv(RUTA_ESCENARIOS)
except Exception:
    mostrar_error_archivos()

if not validar_columnas(datos, COLUMNAS_DATOS):
    mostrar_error_archivos()

if not validar_columnas(metricas, COLUMNAS_METRICAS):
    mostrar_error_archivos()

datos = limpiar_datos(datos)
metricas = limpiar_metricas(metricas)
modelo_seleccionado_final = "Prophet"
metricas_seleccionado = obtener_metricas_modelo(metricas, modelo_seleccionado_final)
columna_seleccionada = MAPA_PREDICCIONES[modelo_seleccionado_final]


st.markdown(
    """
    <div class="main-header">
        <h1>Prediccion de Ventas con Series de Tiempo - Rossmann</h1>
        <p>
            Dashboard de mineria de datos basado en CRISP-DM para estimar demanda futura
            e integrar variables externas de planificacion de marketing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        <div class="summary-card">
            <h4>Dataset</h4>
            <p>Rossmann Store Sales Dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="summary-card">
            <h4>Modelos</h4>
            <p>ARIMA, Prophet y LSTM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="summary-card">
            <h4>Metricas</h4>
            <p>RMSE, MAE y MAPE</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="summary-card">
            <h4>Mejor modelo</h4>
            <p>{modelo_seleccionado_final}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        """
        <div class="summary-card">
            <h4>Funcionalidades</h4>
            <p>Visualizacion, prediccion y escenarios</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown('<div class="section-title">Resumen del proyecto</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="info-box">
        El proyecto estima la demanda futura de ventas usando series de tiempo.
        Se evaluaron ARIMA, Prophet y LSTM. {modelo_seleccionado_final} fue seleccionado como modelo
        principal por permitir analizar tendencia, estacionalidad y variables externas de forma interpretable. Las variables
        externas utilizadas provienen del dataset Rossmann, como promociones,
        festivos, apertura de tiendas, Promo2 y distancia de competencia.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-title">Guia rapida para entender el dashboard</div>', unsafe_allow_html=True)

guia_1, guia_2, guia_3, guia_4 = st.columns(4)

with guia_1:
    st.markdown(
        """
        <div class="guide-card">
            <h4>1. Datos historicos</h4>
            <p>Primero se miran las ventas reales de Rossmann en el tiempo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with guia_2:
    st.markdown(
        """
        <div class="guide-card">
            <h4>2. Modelos</h4>
            <p>ARIMA, Prophet y LSTM intentan copiar el comportamiento de esas ventas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with guia_3:
    st.markdown(
        """
        <div class="guide-card">
            <h4>3. Errores</h4>
            <p>RMSE, MAE y MAPE dicen que tan lejos quedo cada prediccion de la realidad.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with guia_4:
    st.markdown(
        f"""
        <div class="guide-card">
            <h4>4. Escenarios</h4>
            <p>Con {modelo_seleccionado_final} se prueba que pasaria si la demanda sube o baja.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown('<div class="section-title">Panel de control</div>', unsafe_allow_html=True)

fecha_minima = datos["Date"].min().date()
fecha_maxima = datos["Date"].max().date()

st.markdown(
    """
    <div class="control-panel">
        Use este filtro para revisar el dashboard en un periodo especifico.
        Todas las graficas y la tabla final se actualizan con este rango.
    </div>
    """,
    unsafe_allow_html=True,
)

col_fecha_inicio, col_fecha_fin = st.columns(2)

with col_fecha_inicio:
    fecha_inicio = st.date_input(
        "Fecha inicial",
        value=fecha_minima,
        min_value=fecha_minima,
        max_value=fecha_maxima,
        key="fecha_inicio_global",
    )

with col_fecha_fin:
    fecha_fin = st.date_input(
        "Fecha final",
        value=fecha_maxima,
        min_value=fecha_minima,
        max_value=fecha_maxima,
        key="fecha_fin_global",
    )

if fecha_inicio > fecha_fin:
    st.warning("La fecha inicial no puede ser mayor que la fecha final.")
    st.stop()

datos_filtrados = datos[
    (datos["Date"].dt.date >= fecha_inicio)
    & (datos["Date"].dt.date <= fecha_fin)
].copy()

if datos_filtrados.empty:
    st.warning("No hay datos disponibles para el rango de fechas seleccionado.")
    st.stop()


(
    tab_metricas,
    tab_temporal,
    tab_estacionalidad,
    tab_prediccion,
    tab_modelos,
    tab_escenarios,
    tab_tabla,
) = st.tabs(
    [
        "Metricas",
        "Visualizacion temporal",
        "Estacionalidad",
        "Dashboard de prediccion",
        "Comparacion por modelo",
        "Simulacion de escenarios",
        "Tabla final",
    ]
)


with tab_metricas:
    st.markdown('<div class="section-title">Metricas de evaluacion</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.dataframe(
            metricas.style.format(
                {
                    "RMSE": "{:,.2f}",
                    "MAE": "{:,.2f}",
                    "MAPE (%)": "{:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>RMSE {modelo_seleccionado_final}</h4>
                    <h2>{formato_numero(metricas_seleccionado["RMSE"])}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>MAE {modelo_seleccionado_final}</h4>
                    <h2>{formato_numero(metricas_seleccionado["MAE"])}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>MAPE {modelo_seleccionado_final}</h4>
                    <h2>{formato_porcentaje(metricas_seleccionado["MAPE (%)"])}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="success-box">
                {modelo_seleccionado_final} fue seleccionado como modelo principal por su interpretabilidad,
                manejo de estacionalidad y facilidad para la simulacion de escenarios.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-box">
            Nota tecnica: estas metricas se calculan a partir de las predicciones
            generadas por los notebooks. El dashboard no entrena modelos ni modifica
            manualmente los resultados.
        </div>
        """,
        unsafe_allow_html=True,
    )

    exp_rmse, exp_mae, exp_mape = st.columns(3)

    with exp_rmse:
        st.markdown(
            """
            <div class="guide-card">
                <h4>RMSE</h4>
                <p>Mide errores grandes. Si el modelo falla mucho en algunos dias, este valor sube.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_mae:
        st.markdown(
            """
            <div class="guide-card">
                <h4>MAE</h4>
                <p>Dice el error promedio en ventas: en promedio, cuanto se equivoco el modelo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_mape:
        st.markdown(
            """
            <div class="guide-card">
                <h4>MAPE</h4>
                <p>Muestra el error en porcentaje. Sirve para explicar el error de forma sencilla.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab_temporal:
    st.markdown('<div class="section-title">Visualizacion temporal</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Esta grafica permite observar el comportamiento real de las ventas durante
            el periodo de prueba del proyecto.
        </div>
        """,
        unsafe_allow_html=True,
    )

    graficar_lineas(datos_filtrados, ["Real"], "Ventas reales en el periodo seleccionado")


with tab_estacionalidad:
    st.markdown('<div class="section-title">Analisis de estacionalidad</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Estacionalidad significa que las ventas pueden repetirse con patrones.
            Por ejemplo, una tienda puede vender mas en ciertos dias de la semana
            o en ciertos meses. Esta seccion resume esos patrones usando las ventas reales.
        </div>
        """,
        unsafe_allow_html=True,
    )

    datos_estacionalidad = datos_filtrados.copy()
    datos_estacionalidad["DiaSemana"] = datos_estacionalidad["Date"].dt.dayofweek
    datos_estacionalidad["Mes"] = datos_estacionalidad["Date"].dt.month

    nombres_dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miercoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sabado",
        6: "Domingo",
    }
    nombres_meses = {
        1: "Ene",
        2: "Feb",
        3: "Mar",
        4: "Abr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Ago",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dic",
    }

    ventas_por_dia = (
        datos_estacionalidad.groupby("DiaSemana")["Real"]
        .mean()
        .reindex(range(7))
        .dropna()
    )
    ventas_por_mes = (
        datos_estacionalidad.groupby("Mes")["Real"]
        .mean()
        .reindex(range(1, 13))
        .dropna()
    )

    col_dias, col_meses = st.columns(2)

    with col_dias:
        fig_dia, ax_dia = plt.subplots(figsize=(7, 4.5))
        ax_dia.bar(
            [nombres_dias[indice] for indice in ventas_por_dia.index],
            ventas_por_dia.values,
            color="#2563eb",
        )
        ax_dia.set_title(
            "Venta promedio por dia de la semana",
            fontsize=12,
            fontweight="bold",
            color="#16324f",
        )
        ax_dia.set_xlabel("Dia de la semana")
        ax_dia.set_ylabel("Venta promedio")
        ax_dia.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig_dia)
        plt.close(fig_dia)

    with col_meses:
        fig_mes, ax_mes = plt.subplots(figsize=(7, 4.5))
        ax_mes.bar(
            [nombres_meses[indice] for indice in ventas_por_mes.index],
            ventas_por_mes.values,
            color="#16a34a",
        )
        ax_mes.set_title(
            "Venta promedio por mes",
            fontsize=12,
            fontweight="bold",
            color="#16324f",
        )
        ax_mes.set_xlabel("Mes")
        ax_mes.set_ylabel("Venta promedio")
        ax_mes.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig_mes)
        plt.close(fig_mes)

    st.markdown(
        """
        <div class="success-box">
            Esta parte cubre el requisito tecnico de analisis de estacionalidad:
            ayuda a ver si las ventas cambian segun el dia o el mes.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_prediccion:
    st.markdown('<div class="section-title">Dashboard de prediccion</div>', unsafe_allow_html=True)

    col_selector, col_texto = st.columns([0.35, 0.65])

    with col_selector:
        modelo_seleccionado = st.selectbox(
            "Seleccione el modelo a visualizar",
            ["Todos", "ARIMA", "Prophet", "LSTM"],
        )

    with col_texto:
        st.markdown(
            """
            <div class="small-note">
                Esta seccion compara las ventas reales con las predicciones generadas
                por los tres modelos de series de tiempo.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if modelo_seleccionado == "Todos":
        columnas_grafica = [
            "Real",
            "Prediccion_ARIMA",
            "Prediccion_Prophet",
            "Prediccion_LSTM",
        ]
        titulo_grafica = "Comparacion de ventas reales y predicciones"
    elif modelo_seleccionado == "ARIMA":
        columnas_grafica = ["Real", "Prediccion_ARIMA"]
        titulo_grafica = "Ventas reales vs prediccion ARIMA"
    elif modelo_seleccionado == "Prophet":
        columnas_grafica = ["Real", "Prediccion_Prophet"]
        titulo_grafica = "Ventas reales vs prediccion Prophet"
    else:
        columnas_grafica = ["Real", "Prediccion_LSTM"]
        titulo_grafica = "Ventas reales vs prediccion LSTM"

    graficar_lineas(datos_filtrados, columnas_grafica, titulo_grafica)


with tab_modelos:
    st.markdown('<div class="section-title">Comparacion individual por modelo</div>', unsafe_allow_html=True)

    tab_arima, tab_prophet, tab_lstm = st.tabs(["ARIMA", "Prophet", "LSTM"])

    with tab_arima:
        st.markdown(
            """
            <div class="info-box">
                ARIMA se utilizo como modelo base de serie temporal, trabajando con
                el comportamiento historico de las ventas.
            </div>
            """,
            unsafe_allow_html=True,
        )
        graficar_lineas(
            datos_filtrados,
            ["Real", "Prediccion_ARIMA"],
            "Ventas reales vs prediccion ARIMA",
        )

    with tab_prophet:
        st.markdown(
            """
            <div class="info-box">
                Prophet integro variables externas del dataset y permitio comparar
                un modelo especializado en series temporales con ARIMA y LSTM.
            </div>
            """,
            unsafe_allow_html=True,
        )
        graficar_lineas(
            datos_filtrados,
            ["Real", "Prediccion_Prophet"],
            "Ventas reales vs prediccion Prophet",
        )

    with tab_lstm:
        st.markdown(
            """
            <div class="info-box">
                LSTM se aplico como modelo de red neuronal para series de tiempo,
                incorporando variables externas preparadas en el proceso de mineria.
            </div>
            """,
            unsafe_allow_html=True,
        )
        graficar_lineas(
            datos_filtrados,
            ["Real", "Prediccion_LSTM"],
            "Ventas reales vs prediccion LSTM",
        )


with tab_escenarios:
    st.markdown('<div class="section-title">Simulacion de escenarios</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="info-box">
            La simulacion permite analizar posibles variaciones de la demanda para
            apoyar la planificacion de marketing. No es un modelo nuevo: es una
            variacion aplicada sobre la prediccion de {modelo_seleccionado_final}, seleccionado como
            modelo final por su interpretabilidad y manejo de estacionalidad.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_opt, col_pes = st.columns(2)

    with col_opt:
        porcentaje_optimista = st.slider(
            "Escenario optimista",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
            format="%d%%",
        )

    with col_pes:
        porcentaje_pesimista = st.slider(
            "Escenario pesimista",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
            format="%d%%",
        )

    datos_escenarios = datos_filtrados.copy()
    datos_escenarios["Escenario_Base"] = datos_escenarios[columna_seleccionada]
    datos_escenarios["Escenario_Optimista"] = datos_escenarios[columna_seleccionada] * (
        1 + porcentaje_optimista / 100
    )
    datos_escenarios["Escenario_Pesimista"] = datos_escenarios[columna_seleccionada] * (
        1 - porcentaje_pesimista / 100
    )

    graficar_lineas(
        datos_escenarios,
        ["Escenario_Base", "Escenario_Optimista", "Escenario_Pesimista"],
        f"Simulacion de escenarios de demanda usando {modelo_seleccionado_final}",
    )

    col_base, col_optimista, col_pesimista = st.columns(3)

    with col_base:
        st.metric(
            "Promedio escenario base",
            formato_numero(datos_escenarios["Escenario_Base"].mean()),
        )

    with col_optimista:
        st.metric(
            "Promedio escenario optimista",
            formato_numero(datos_escenarios["Escenario_Optimista"].mean()),
        )

    with col_pesimista:
        st.metric(
            "Promedio escenario pesimista",
            formato_numero(datos_escenarios["Escenario_Pesimista"].mean()),
        )

    if not resumen_escenarios.empty:
        st.markdown("**Resumen de escenarios generado por los notebooks**")
        st.dataframe(resumen_escenarios, use_container_width=True, hide_index=True)


with tab_tabla:
    st.markdown('<div class="section-title">Tabla final de resultados</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            La tabla muestra ventas reales, predicciones y escenarios para el rango
            elegido en el panel de control. Los escenarios de esta tabla se actualizan
            con los porcentajes definidos en la seccion de simulacion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabla_filtrada = datos_escenarios[
        [
            "Date",
            "Real",
            "Prediccion_ARIMA",
            "Prediccion_Prophet",
            "Prediccion_LSTM",
            "Escenario_Base",
            "Escenario_Optimista",
            "Escenario_Pesimista",
        ]
    ].copy()

    tabla_filtrada["Date"] = tabla_filtrada["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        tabla_filtrada.style.format(
            {
                "Real": "{:,.2f}",
                "Prediccion_ARIMA": "{:,.2f}",
                "Prediccion_Prophet": "{:,.2f}",
                "Prediccion_LSTM": "{:,.2f}",
                "Escenario_Base": "{:,.2f}",
                "Escenario_Optimista": "{:,.2f}",
                "Escenario_Pesimista": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


st.markdown("---")
st.markdown(
    """
    <div class="small-note">
        Flujo del proyecto: Dataset historico Rossmann -> modelos predictivos ->
        predicciones -> dashboard -> simulacion de escenarios.
    </div>
    """,
    unsafe_allow_html=True,
)
