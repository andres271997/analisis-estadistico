import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Análisis Estadístico Descriptivo", page_icon="📊", layout="wide")

SAMPLES = {
    "Calificaciones": "3.2 4.1 3.8 2.9 4.5 3.6 3.9 4.0 2.5 3.3 4.7 3.1 3.7 4.2 3.5 2.8 4.4 3.9 3.0 3.6 4.8 3.4 3.8 2.7 4.1 3.2 3.9 4.6 3.5 3.3 1.2 4.0 3.7 3.1 4.3 3.6 2.9 3.8 4.9 3.4",
    "Edades": "18 19 20 20 21 21 21 22 22 22 22 23 23 23 24 24 25 25 25 26 26 27 27 28 29 30 31 33 35 38 41 45 19 20 22 21",
    "Ingresos": "1300 1450 1600 1160 2100 1800 1750 1500 2500 1400 1650 1900 3200 1550 1700 2000 1350 1850 2200 1600 7800 1480 1720 2400 1950 1580 2800 1680 1520 9500",
    "Lenguajes (cualitativa)": "Python Java Python JavaScript C++ Python Java JavaScript Python Go Python Java C++ JavaScript Python Rust Java Python JavaScript Python Go Java Python C++ JavaScript",
}
MISSING = {"", "na", "n/a", "null", "nan", "-", "?"}

# ---------------- Lógica estadística (fórmulas poblacionales, divide por N) ----------------
def parse_tokens(tokens):
    tokens = [str(t).strip() for t in tokens if str(t).strip()]
    nums, texts, invalid = [], [], 0
    for t in tokens:
        if t.lower() in MISSING:
            invalid += 1; continue
        try:
            v = float(t)
            if math.isfinite(v): nums.append(v)
            else: texts.append(t)
        except ValueError:
            texts.append(t)
    if len(texts) > len(nums):
        return {"kind": "cualitativa", "values": [], "cats": texts, "invalid": invalid + len(nums), "total": len(tokens)}
    kind = "discreta" if all(float(n).is_integer() for n in nums) else "continua"
    return {"kind": kind, "values": nums, "cats": [], "invalid": invalid + len(texts), "total": len(tokens)}

def parse_text(text):
    import re
    return parse_tokens(re.split(r"[,;\t\s]+", text))

def quantile(s, p):
    n = len(s); pos = p * (n + 1)
    if pos <= 1: return s[0]
    if pos >= n: return s[-1]
    lo = int(math.floor(pos)); f = pos - lo
    return s[lo - 1] + f * (s[lo] - s[lo - 1])

def compute(values, rule="Sturges"):
    s = sorted(values); N = len(s); mean = sum(s) / N
    median = s[(N - 1) // 2] if N % 2 else (s[N // 2 - 1] + s[N // 2]) / 2
    counts = pd.Series(s).value_counts().sort_index()
    maxc = counts.max(); modes = list(counts[counts == maxc].index)
    if maxc == 1 or len(modes) == len(counts): modes, mtype = [], "Amodal"
    else: mtype = {1: "Unimodal", 2: "Bimodal"}.get(len(modes), "Multimodal")
    q1, q2, q3 = quantile(s, .25), quantile(s, .5), quantile(s, .75); iqr = q3 - q1
    mom = lambda p: sum((x - mean) ** p for x in s) / N
    m2, m3, m4 = mom(2), mom(3), mom(4); sd = math.sqrt(m2)
    cv = sd / abs(mean) * 100 if mean else float("nan")
    g1 = m3 / sd**3 if sd else 0; g2 = m4 / sd**4 - 3 if sd else 0
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = [v for v in s if v < lf or v > uf]
    k = max(1, math.ceil(1 + 3.322 * math.log10(N)) if rule == "Sturges" else math.ceil(math.sqrt(N)))
    rng = s[-1] - s[0]; w = rng / k if rng else 1
    rows, F = [], 0
    for i in range(k):
        li = s[0] + i * w; ls = s[-1] + (0 if rng else 1) if i == k - 1 else s[0] + (i + 1) * w
        fi = sum(1 for v in s if v >= li and (v <= ls if i == k - 1 else v < ls)); F += fi
        rows.append({"Li": li, "Ls": ls, "xi": (li + ls) / 2, "fi": fi, "hi": fi / N, "Fi": F, "Hi": F / N})
    return dict(N=N, s=s, sum=sum(s), mean=mean, median=median, modes=modes, mtype=mtype, min=s[0], max=s[-1],
                range=rng, q1=q1, q2=q2, q3=q3, iqr=iqr, var=m2, sd=sd, cv=cv, g1=g1, g2=g2, m1=mom(1), m2=m2,
                m3=m3, m4=m4, lf=lf, uf=uf, out=out, k=k, w=w, classes=pd.DataFrame(rows), counts=counts)

def fmt(x, d=4):
    if x is None or not math.isfinite(x): return "—"
    return f"{round(x, d):,.{d}f}".rstrip("0").rstrip(".") if d else f"{x:,.0f}"

def skew_label(g):
    if abs(g) < .1: return "Simétrica", "Media, mediana y moda tienden a coincidir."
    return ("Asimetría positiva", "Cola alargada hacia la derecha.") if g > 0 else ("Asimetría negativa", "Cola alargada hacia la izquierda.")

def kurt_label(g):
    if abs(g) < .1: return "Mesocúrtica", "Apuntamiento similar a la normal."
    return ("Leptocúrtica", "Más apuntada que la normal, colas pesadas.") if g > 0 else ("Platicúrtica", "Más achatada que la normal.")

def cv_label(cv):
    if not math.isfinite(cv): return "indeterminada"
    return "baja" if cv < 10 else "moderada" if cv < 25 else "alta" if cv < 40 else "muy alta"

def interpretation(r):
    sk, ku = skew_label(r["g1"]), kurt_label(r["g2"])
    moda = "no existe un valor que se repita más que otros (amodal)" if r["mtype"] == "Amodal" else f"la distribución es {r['mtype'].lower()} con moda {', '.join(fmt(m,2) for m in r['modes'])}"
    return [
        f"Se analizó una población de N = {r['N']} datos, entre {fmt(r['min'],2)} y {fmt(r['max'],2)} (rango {fmt(r['range'],2)}).",
        f"El promedio poblacional es μ = {fmt(r['mean'],2)} y la mediana es {fmt(r['median'],2)}; {moda}.",
        f"Variabilidad {cv_label(r['cv'])}: CV = {fmt(r['cv'],2)}% y σ = {fmt(r['sd'],2)}.",
        f"El 50% central está entre Q₁ = {fmt(r['q1'],2)} y Q₃ = {fmt(r['q3'],2)} (RIC = {fmt(r['iqr'],2)}).",
        (f"Se detectaron {len(r['out'])} valor(es) atípico(s) fuera de [{fmt(r['lf'],2)}, {fmt(r['uf'],2)}]: {', '.join(fmt(v,2) for v in r['out'][:10])}."
         if r["out"] else "No se detectaron valores atípicos (criterio de Tukey 1.5 × RIC)."),
        f"Forma: {sk[0].lower()} (g₁ = {fmt(r['g1'],3)}) y {ku[0].lower()} (g₂ = {fmt(r['g2'],3)}). {sk[1]}",
    ]

# ---------------- Gráficos ----------------
def histogram(r):
    c = r["classes"]
    fig = go.Figure(go.Bar(x=c["xi"], y=c["fi"], width=r["w"] * .98, name="fi", marker_color="#1f4e79"))
    fig.add_trace(go.Scatter(x=c["xi"], y=c["fi"], mode="lines+markers", name="Polígono", line_color="#e07a1f"))
    fig.update_layout(title="Histograma y polígono de frecuencias", height=380, margin=dict(t=40, b=20))
    return fig

def ogive(r):
    c = r["classes"]
    fig = go.Figure(go.Scatter(x=[c["Li"].iloc[0]] + list(c["Ls"]), y=[0] + list(c["Fi"]), mode="lines+markers", line_color="#2a9d8f"))
    fig.update_layout(title="Ojiva (Fi)", height=380, margin=dict(t=40, b=20))
    return fig

def boxplot(r):
    fig = go.Figure(go.Box(x=r["s"], boxpoints="outliers", name="", marker_color="#1f4e79"))
    fig.update_layout(title="Diagrama de caja y bigotes", height=300, margin=dict(t=40, b=20))
    return fig

def density(r):
    s = np.array(r["s"]); N = len(s); h = 1.06 * (r["sd"] or 1) * N ** -0.2
    xs = np.linspace(s.min() - 2*h, s.max() + 2*h, 200)
    ys = np.exp(-0.5 * ((xs[:, None] - s) / h) ** 2).sum(1) / (N * h * math.sqrt(2*math.pi))
    fig = go.Figure(go.Scatter(x=xs, y=ys, fill="tozeroy", name="Densidad", line_color="#1f4e79"))
    for v, n, col in [(r["mean"], "μ", "#e63946"), (r["median"], "Me", "#2a9d8f")]:
        fig.add_vline(x=v, line_dash="dash", line_color=col, annotation_text=n)
    fig.update_layout(title="Curva de densidad", height=340, margin=dict(t=40, b=20))
    return fig

# ---------------- Interfaz ----------------
st.title("📊 Análisis Estadístico Descriptivo")
st.caption("UTP · Ingeniería en Sistemas y Computación · Estadística poblacional (divide por N)")

if "parsed" not in st.session_state:
    st.session_state.parsed = parse_text(SAMPLES["Calificaciones"]); st.session_state.source = "Calificaciones"

with st.sidebar:
    st.header("Pasos")
    step = st.radio("Ir a", ["1. Gestor de datos", "2. Frecuencias", "3. Tendencia y posición",
                             "4. Dispersión y atípicos", "5. Momentos y forma", "6. Dashboard e informe"], label_visibility="collapsed")
    rule = st.selectbox("Regla de clases", ["Sturges", "Raíz de N"])
    st.caption(f"Fuente: **{st.session_state.source}**")

p = st.session_state.parsed
r = compute(p["values"], rule) if p["kind"] != "cualitativa" and len(p["values"]) >= 2 else None

if step.startswith("1"):
    st.subheader("Paso 01 · Gestor de datos")
    t1, t2, t3 = st.tabs(["✍️ Manual", "📁 Archivo CSV/Excel", "🧪 Ejemplos"])
    with t1:
        txt = st.text_area("Valores separados por espacios, comas o saltos de línea", height=150)
        if st.button("Cargar datos") and txt.strip():
            st.session_state.parsed = parse_text(txt); st.session_state.source = "Manual"; st.rerun()
    with t2:
        f = st.file_uploader("Sube un archivo", type=["csv", "xlsx", "xls"])
        if f:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            col = st.selectbox("Columna", df.columns)
            if st.button("Usar columna"):
                st.session_state.parsed = parse_tokens(df[col].astype(str).tolist()); st.session_state.source = f"{f.name} · {col}"; st.rerun()
    with t3:
        name = st.selectbox("Conjunto", list(SAMPLES))
        if st.button("Cargar ejemplo"):
            st.session_state.parsed = parse_text(SAMPLES[name]); st.session_state.source = name; st.rerun()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tipo de variable", p["kind"].capitalize())
    c2.metric("Datos válidos (N)", len(p["values"]) or len(p["cats"]))
    c3.metric("Filtrados", p["invalid"])
    if p["invalid"]: st.warning(f"Se eliminaron {p['invalid']} valores vacíos o no válidos.")
    st.dataframe(pd.DataFrame({"Dato": p["values"] or p["cats"]}), height=250, use_container_width=True)

elif p["kind"] == "cualitativa":
    st.subheader("Variable cualitativa")
    st.info("Solo aplican frecuencias, moda y gráficos categóricos.")
    vc = pd.Series(p["cats"]).value_counts(); N = vc.sum()
    df = pd.DataFrame({"Categoría": vc.index, "fi": vc.values, "hi": vc.values / N, "Fi": vc.cumsum().values, "Hi": vc.cumsum().values / N})
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.metric("Moda", ", ".join(vc[vc == vc.max()].index))
    a, b = st.columns(2)
    a.plotly_chart(go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color="#1f4e79")).update_layout(title="Barras"), use_container_width=True)
    b.plotly_chart(go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=.4)).update_layout(title="Circular"), use_container_width=True)

elif r is None:
    st.warning("Carga al menos 2 datos numéricos en el paso 1.")

elif step.startswith("2"):
    st.subheader("Paso 02 · Tabla de frecuencias")
    order = st.radio("Orden de datos", ["Ascendente", "Descendente"], horizontal=True)
    st.write(", ".join(fmt(v, 2) for v in (r["s"] if order == "Ascendente" else r["s"][::-1])))
    st.latex(r"k = 1 + 3.322\log_{10}N" if rule == "Sturges" else r"k=\lceil\sqrt{N}\rceil")
    st.caption(f"k = {r['k']} clases · amplitud A = {fmt(r['w'])}")
    st.dataframe(r["classes"].style.format({"Li": "{:.3f}", "Ls": "{:.3f}", "xi": "{:.3f}", "hi": "{:.4f}", "Hi": "{:.4f}"}), use_container_width=True, hide_index=True)
    if p["kind"] == "discreta":
        st.markdown("**Frecuencia por valor (variable discreta)**")
        st.dataframe(pd.DataFrame({"x": r["counts"].index, "fi": r["counts"].values}), hide_index=True)
    a, b = st.columns(2); a.plotly_chart(histogram(r), use_container_width=True); b.plotly_chart(ogive(r), use_container_width=True)

elif step.startswith("3"):
    st.subheader("Paso 03 · Tendencia central y posición")
    c = st.columns(3)
    c[0].metric("Media μ", fmt(r["mean"])); c[0].latex(r"\mu=\frac{\sum x_i}{N}")
    c[1].metric("Mediana Me", fmt(r["median"]))
    c[2].metric(f"Moda ({r['mtype']})", ", ".join(fmt(m) for m in r["modes"]) or "—")
    c = st.columns(3)
    c[0].metric("Q₁", fmt(r["q1"])); c[1].metric("Q₂", fmt(r["q2"])); c[2].metric("Q₃", fmt(r["q3"]))
    st.latex(r"Q_k:\ \text{posición}=\frac{k(N+1)}{4}")
    pc = st.slider("Percentil P", 1, 99, 90)
    st.metric(f"P{pc}", fmt(quantile(r["s"], pc / 100)))
    st.plotly_chart(density(r), use_container_width=True)

elif step.startswith("4"):
    st.subheader("Paso 04 · Dispersión y atípicos")
    c = st.columns(4)
    c[0].metric("Rango", fmt(r["range"])); c[1].metric("Varianza σ²", fmt(r["var"]))
    c[2].metric("Desv. estándar σ", fmt(r["sd"])); c[3].metric("CV", f"{fmt(r['cv'],2)} %")
    st.latex(r"\sigma^2=\frac{\sum (x_i-\mu)^2}{N}\qquad CV=\frac{\sigma}{|\mu|}\cdot100")
    st.caption(f"Variabilidad {cv_label(r['cv'])}. RIC = {fmt(r['iqr'])}. Cercas: [{fmt(r['lf'])}, {fmt(r['uf'])}]")
    st.plotly_chart(boxplot(r), use_container_width=True)
    if r["out"]: st.error("Atípicos: " + ", ".join(fmt(v, 2) for v in r["out"]))
    else: st.success("Sin valores atípicos.")

elif step.startswith("5"):
    st.subheader("Paso 05 · Momentos y forma")
    c = st.columns(4)
    for i, k in enumerate(["m1", "m2", "m3", "m4"]): c[i].metric(f"Momento central {k}", fmt(r[k]))
    st.latex(r"m_r=\frac{\sum (x_i-\mu)^r}{N}\quad g_1=\frac{m_3}{\sigma^3}\quad g_2=\frac{m_4}{\sigma^4}-3")
    sk, ku = skew_label(r["g1"]), kurt_label(r["g2"])
    a, b = st.columns(2)
    a.metric("Asimetría g₁", fmt(r["g1"], 3), sk[0], delta_color="off"); a.caption(sk[1])
    b.metric("Curtosis g₂", fmt(r["g2"], 3), ku[0], delta_color="off"); b.caption(ku[1])
    st.plotly_chart(density(r), use_container_width=True)

else:
    st.subheader("Paso 06 · Dashboard e informe")
    c = st.columns(6)
    for col, (l, v) in zip(c, [("N", str(r["N"])), ("μ", fmt(r["mean"], 2)), ("Me", fmt(r["median"], 2)), ("σ", fmt(r["sd"], 2)), ("CV %", fmt(r["cv"], 2)), ("Atípicos", str(len(r["out"])))]):
        col.metric(l, v)
    a, b = st.columns(2)
    a.plotly_chart(histogram(r), use_container_width=True); b.plotly_chart(boxplot(r), use_container_width=True)
    a.plotly_chart(density(r), use_container_width=True); b.plotly_chart(ogive(r), use_container_width=True)
    st.markdown("### Interpretación automática")
    text = interpretation(r)
    for t in text: st.markdown(f"- {t}")
    summary = pd.DataFrame([(k, r[k]) for k in ["N","sum","mean","median","min","max","range","q1","q2","q3","iqr","var","sd","cv","g1","g2"]], columns=["Medida", "Valor"])
    report = "INFORME — ANÁLISIS ESTADÍSTICO DESCRIPTIVO\nFuente: " + st.session_state.source + "\n\n" + summary.to_string(index=False) + "\n\nTabla de frecuencias\n" + r["classes"].to_string(index=False) + "\n\nInterpretación\n" + "\n".join("- " + t for t in text)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Descargar informe (.txt)", report, "informe_estadistico.txt")
    d2.download_button("⬇️ Descargar tabla (.csv)", r["classes"].to_csv(index=False), "frecuencias.csv")
