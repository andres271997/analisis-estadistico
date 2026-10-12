import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIGURACIÓN INICIAL (Como preparar el salón de clases antes de empezar)
# =============================================================================
st.set_page_config(page_title="Análisis Estadístico Descriptivo", layout="wide")

# Datos de ejemplo (como tener ejercicios de práctica en el tablero)
SAMPLES = {
    "Calificaciones": "3.2 4.1 3.8 2.9 4.5 3.6 3.9 4.0 2.5 3.3 4.7 3.1 3.7 4.2 3.5 2.8 4.4 3.9 3.0 3.6 4.8 3.4 3.8 2.7 4.1 3.2 3.9 4.6 3.5 3.3 1.2 4.0 3.7 3.1 4.3 3.6 2.9 3.8 4.9 3.4",
    "Edades": "18 19 20 20 21 21 21 22 22 22 22 23 23 23 24 24 25 25 25 26 26 27 27 28 29 30 31 33 35 38 41 45 19 20 22 21",
    "Ingresos": "1300 1450 1600 1160 2100 1800 1750 1500 2500 1400 1650 1900 3200 1550 1700 2000 1350 1850 2200 1600 7800 1480 1720 2400 1950 1580 2800 1680 1520 9500",
    "Lenguajes (cualitativa)": "Python Java Python JavaScript C++ Python Java JavaScript Python Go Python Java C++ JavaScript Python Rust Java Python JavaScript Python Go Java Python C++ JavaScript",
}

# Valores que consideramos vacíos o inválidos (como borrar tareas mal escritas)
MISSING = {"", "na", "n/a", "null", "nan", "-", "?"}

# =============================================================================
# LÓGICA ESTADÍSTICA (Las fórmulas que la profe enseñó)
# =============================================================================

def parse_tokens(tokens):
    """
    Analogía: Como separar los ingredientes de una receta.
    Toma una lista de textos y decide si son números (para calcular) o palabras (categorías).
    Devuelve un diccionario con la información limpia.
    """
    tokens = [str(t).strip() for t in tokens if str(t).strip()]
    nums, texts, invalid = [], [], 0

    for t in tokens:
        if t.lower() in MISSING:
            invalid += 1
            continue
        try:
            v = float(t)
            if math.isfinite(v):
                nums.append(v)
            else:
                texts.append(t)
        except ValueError:
            texts.append(t)

    # Si hay más palabras que números, asumimos que es una variable cualitativa
    if len(texts) > len(nums):
        return {"kind": "cualitativa", "values": [], "cats": texts, "invalid": invalid + len(nums), "total": len(tokens)}

    # Determinamos si es discreta (enteros) o continua (decimales)
    kind = "discreta" if all(float(n).is_integer() for n in nums) else "continua"
    return {"kind": kind, "values": nums, "cats": [], "invalid": invalid + len(texts), "total": len(tokens)}


def parse_text(text):
    """
    Analogía: Como picar todos los ingredientes en trozos pequeños.
    Toma un texto largo y lo divide por comas, espacios o saltos de línea.
    """
    import re
    return parse_tokens(re.split(r"[,;\t\s]+", text))


def quantile(s, p):
    """
    Analogía: Encontrar la posición exacta de un corredor en una carrera.
    Calcula el cuantil 'p' (ej: 0.25 para Q1) usando la fórmula de posición:
    pos = p * (N + 1)
    Si la posición no es entera, interpola entre los dos valores vecinos.
    """
    n = len(s)
    pos = p * (n + 1)
    if pos <= 1:
        return s[0]
    if pos >= n:
        return s[-1]
    lo = int(math.floor(pos))
    f = pos - lo
    # Interpolación lineal (como promediar dos notas cercanas)
    return s[lo - 1] + f * (s[lo] - s[lo - 1])


def compute(values, stats_type="Poblacional"):
    """
    Analogía: La cocina completa. Aquí se procesan todos los datos
    para obtener las medidas de tendencia central, dispersión y frecuencia.

    El parámetro 'stats_type' decide si usamos fórmulas poblacionales (dividir entre N)
    o muestrales (dividir entre n-1) para la varianza y la desviación estándar.

    Fórmulas:
    - Varianza poblacional:  σ² = Σ(xi - μ)² / N
    - Varianza muestral:     s² = Σ(xi - x̄)² / (n - 1)
    """
    s = sorted(values)
    N = len(s)

    # --- MEDIDAS DE TENDENCIA CENTRAL ---
    # Media: suma de todos / N (como repartir una cuenta en partes iguales)
    # Poblacional: μ = Σxi / N     |     Muestral: x̄ = Σxi / n
    mean = sum(s) / N

    # Mediana: el valor del medio (si N es par, promedio de los dos centrales)
    median = s[(N - 1) // 2] if N % 2 else (s[N // 2 - 1] + s[N // 2]) / 2

    # Moda: el valor que más se repite (como el sabor de helado favorito)
    counts = pd.Series(s).value_counts().sort_index()
    maxc = counts.max()
    modes = list(counts[counts == maxc].index)

    if maxc == 1 or len(modes) == len(counts):
        modes, mtype = [], "Amodal"
    else:
        mtype = {1: "Unimodal", 2: "Bimodal"}.get(len(modes), "Multimodal")

    # --- MEDIDAS DE POSICIÓN (CUARTILES) ---
    # Q1, Q2, Q3 dividen los datos en 4 partes iguales (como cortar una torta)
    q1, q2, q3 = quantile(s, .25), quantile(s, .5), quantile(s, .75)
    iqr = q3 - q1  # Rango Intercuartílico (la parte central de la torta)

    # --- MEDIDAS DE DISPERSIÓN (Fórmulas según tipo de estadística) ---
    # Momentos centrales: miden qué tan lejos están los datos del promedio
    # El divisor cambia según sea poblacional (N) o muestral (n-1)
    divisor = N if stats_type == "Poblacional" else (N - 1)

    def mom(p):
        return sum((x - mean) ** p for x in s) / divisor

    m2, m3, m4 = mom(2), mom(3), mom(4)

    # Desviación estándar: raíz de la varianza
    sd = math.sqrt(m2)

    # Coeficiente de Variación: compara la dispersión con el promedio (como medir la consistencia)
    cv = sd / abs(mean) * 100 if mean else float("nan")

    # Asimetría y Curtosis (forma de la distribución)
    g1 = m3 / sd**3 if sd else 0
    g2 = m4 / sd**4 - 3 if sd else 0

    # Límites para valores atípicos (Regla de Tukey: 1.5 * RIC)
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = [v for v in s if v < lf or v > uf]

    # --- CONSTRUCCIÓN DE INTERVALOS (Regla de Sturges) ---
    # Como agrupar los datos en cajas de tamaño similar
    # Fórmula: k = 1 + 3.322 * log10(N)
    rng = s[-1] - s[0]
    k = max(1, math.ceil(1 + 3.322 * math.log10(N)))
    discrete = all(float(v).is_integer() for v in s)

    if discrete:
        # Amplitud entera: evita clases vacías "entre" enteros
        w = max(1, math.ceil(rng / k))
        k = math.ceil((rng + 1) / w)  # clases necesarias para cubrir todos los enteros
    else:
        w = rng / k if rng else 1

    rows, F = [], 0
    for i in range(k):
        li = s[0] + i * w
        if discrete:
            ls = li + w
            fi = sum(1 for v in s if li <= v < ls)
            xi = (li + ls - 1) / 2  # centro real de los enteros de la clase
        else:
            ls = s[-1] + (0 if rng else 1) if i == k - 1 else s[0] + (i + 1) * w
            fi = sum(1 for v in s if v >= li and (v <= ls if i == k - 1 else v < ls))
            xi = (li + ls) / 2
        F += fi
        rows.append({
            "Li": li, "Ls": ls, "xi": xi, "fi": fi,
            "hi": fi / N, "Fi": F, "Hi": F / N
        })

    return dict(
        N=N, s=s, sum=sum(s), mean=mean, median=median, modes=modes, mtype=mtype,
        min=s[0], max=s[-1], range=rng, q1=q1, q2=q2, q3=q3, iqr=iqr,
        var=m2, sd=sd, cv=cv, g1=g1, g2=g2, m1=mom(1), m2=m2, m3=m3, m4=m4,
        lf=lf, uf=uf, out=out, k=k, w=w, classes=pd.DataFrame(rows), counts=counts,
        stats_type=stats_type
    )


def fmt(x, d=4):
    """Analogía: Como redondear una nota en el boletín."""
    if x is None or not math.isfinite(x):
        return "—"
    return f"{round(x, d):,.{d}f}".rstrip("0").rstrip(".") if d else f"{x:,.0f}"


# =============================================================================
# GRÁFICOS (Las representaciones visuales que pidió la profe)
# =============================================================================

def histogram(r):
    """
    Analogía: Un dibujo de barras que muestra cuántos datos caen en cada caja.
    También dibuja la línea del polígono de frecuencias.
    """
    c = r["classes"]
    fig = go.Figure(go.Bar(x=c["xi"], y=c["fi"], width=r["w"] * .98, name="fi", marker_color="#1f4e79"))
    fig.add_trace(go.Scatter(x=c["xi"], y=c["fi"], mode="lines+markers", name="Polígono", line_color="#e07a1f"))
    fig.update_layout(title="Histograma y polígono de frecuencias", height=380, margin=dict(t=40, b=20))
    return fig


def ogive(r):
    """
    Analogía: La línea que sube y muestra cuántos datos llevamos acumulados.
    """
    c = r["classes"]
    fig = go.Figure(go.Scatter(
        x=[c["Li"].iloc[0]] + list(c["Ls"]),
        y=[0] + list(c["Fi"]),
        mode="lines+markers",
        line_color="#2a9d8f"
    ))
    fig.update_layout(title="Ojiva (Fi)", height=380, margin=dict(t=40, b=20))
    return fig


# =============================================================================
# INTERFAZ DE USUARIO (La página web que ve el estudiante)
# =============================================================================

st.title("Análisis Estadístico Descriptivo")

# Inicializar el estado de la sesión (como guardar la página donde ibas)
if "parsed" not in st.session_state:
    st.session_state.parsed = parse_text(SAMPLES["Calificaciones"])
    st.session_state.source = "Calificaciones"
    st.session_state.stats_type = "Poblacional"

# Barra lateral (el menú de navegación)
with st.sidebar:
    st.header("Pasos")
    step = st.radio("Ir a", [
        "1. Gestor de datos",
        "2. Frecuencias"
    ], label_visibility="collapsed")

# Procesar los datos cargados
p = st.session_state.parsed
r = None
if p["kind"] != "cualitativa" and len(p["values"]) >= 2:
    r = compute(p["values"], stats_type=st.session_state.stats_type)

# --- PASO 1: GESTOR DE DATOS ---
if step.startswith("1"):
    st.subheader("Paso 1 · Gestor de datos")
    t1, t2, t3 = st.tabs([" Manual", " Archivo CSV/Excel", " Ejemplos"])

    with t1:
        txt = st.text_area("Valores separados por espacios, comas o saltos de línea", height=150)
        if st.button("Cargar datos") and txt.strip():
            st.session_state.parsed = parse_text(txt)
            st.session_state.source = "Manual"
            st.rerun()

    with t2:
        f = st.file_uploader("Sube un archivo", type=["csv", "xlsx", "xls"])
        if f:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            col = st.selectbox("Columna", df.columns)
            if st.button("Usar columna"):
                st.session_state.parsed = parse_tokens(df[col].astype(str).tolist())
                st.session_state.source = f"{f.name} · {col}"
                st.rerun()

    with t3:
        name = st.selectbox("Conjunto", list(SAMPLES))
        if st.button("Cargar ejemplo"):
            st.session_state.parsed = parse_text(SAMPLES[name])
            st.session_state.source = name
            st.rerun()

    # --- BOTÓN PARA VERIFICAR TIPO DE ESTADÍSTICA ---
    st.markdown("---")
    st.markdown("### Tipo de Estadística")
    st.write(
        "Selecciona manualmente si el ejercicio corresponde a una **población** "
        "(todos los datos) o a una **muestra** (una parte de la población). "
        "Esto cambia las fórmulas de varianza y desviación estándar."
    )

    tipo = st.radio(
        "¿Los datos representan una población o una muestra?",
        ["Poblacional", "Muestral"],
        index=0 if st.session_state.stats_type == "Poblacional" else 1,
        horizontal=True
    )

    if st.button("✅ Verificar tipo de estadística"):
        st.session_state.stats_type = tipo
        if tipo == "Poblacional":
            st.success(
                "Se usará **estadística poblacional**: la varianza y la desviación estándar "
                "dividen entre **N** (el total de datos). Fórmulas: σ² = Σ(xi - μ)² / N"
            )
        else:
            st.success(
                "Se usará **estadística muestral**: la varianza y la desviación estándar "
                "dividen entre **n - 1** (para corregir el sesgo). Fórmulas: s² = Σ(xi - x̄)² / (n - 1)"
            )
        st.rerun()

    # Mostrar el tipo actual
    st.info(f"Tipo de estadística activo: **{st.session_state.stats_type}**")

    # --- RESUMEN DE DATOS CARGADOS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Tipo de variable", p["kind"].capitalize())
    c2.metric("Datos válidos (N)", len(p["values"]) or len(p["cats"]))
    c3.metric("Filtrados", p["invalid"])

    if p["invalid"]:
        st.warning(f"Se eliminaron {p['invalid']} valores vacíos o no válidos.")

    st.dataframe(
        pd.DataFrame({"Dato": p["values"] or p["cats"]}),
        height=250,
        use_container_width=True
    )

# --- VARIABLE CUALITATIVA ---
elif p["kind"] == "cualitativa":
    st.subheader("Variable cualitativa")
    st.info("Solo aplican frecuencias, moda y gráficos categóricos.")
    vc = pd.Series(p["cats"]).value_counts()
    N = vc.sum()
    df = pd.DataFrame({
        "Categoría": vc.index,
        "fi": vc.values,
        "hi": vc.values / N,
        "Fi": vc.cumsum().values,
        "Hi": vc.cumsum().values / N
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.metric("Moda", ", ".join(vc[vc == vc.max()].index))

    a, b = st.columns(2)
    a.plotly_chart(
        go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color="#1f4e79")).update_layout(title="Barras"),
        use_container_width=True
    )
    b.plotly_chart(
        go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=.4)).update_layout(title="Circular"),
        use_container_width=True
    )

elif r is None:
    st.warning("Carga al menos 2 datos numéricos en el paso 1.")

# --- PASO 2: FRECUENCIAS ---
elif step.startswith("2"):
    st.subheader("Paso 2 · Tabla de frecuencias")

    order = st.radio("Orden de datos", ["Ascendente", "Descendente"], horizontal=True)
    st.write(", ".join(fmt(v, 2) for v in (r["s"] if order == "Ascendente" else r["s"][::-1])))

    # Fórmula de Sturges (siempre se usa en esta fase)
    st.latex(r"k = 1 + 3.322\log_{10}N")
    st.caption(f"k = {r['k']} clases · amplitud A = {fmt(r['w'])}")

    st.dataframe(
        r["classes"].style.format({
            "Li": "{:.3f}", "Ls": "{:.3f}", "xi": "{:.3f}",
            "hi": "{:.4f}", "Hi": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    if p["kind"] == "discreta":
        st.markdown("**Frecuencia por valor (variable discreta)**")
        st.dataframe(
            pd.DataFrame({"x": r["counts"].index, "fi": r["counts"].values}),
            hide_index=True
        )

    a, b = st.columns(2)
    a.plotly_chart(histogram(r), use_container_width=True)
    b.plotly_chart(ogive(r), use_container_width=True)