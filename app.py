"""App Streamlit: Black-Litterman para asignación de volumen entre líneas de una familia de producto."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bl import data, model, report

st.set_page_config(page_title="Black-Litterman | Asignación de volumen", layout="wide")
st.title("Black-Litterman para asignación de volumen entre líneas")
st.caption("Retorno = varianza favorable vs costo estándar (USD/ton) = estándar − real. "
           "Positivo: la línea corre por debajo del estándar. Peso = participación en el volumen de la familia.")

# ------------------------------------------------------------------ 1. datos
with st.sidebar:
    st.header("1. Datos")
    fuente = st.radio("Fuente", ["Ejemplo generado", "Cargar CSV"])
    if fuente == "Ejemplo generado":
        n_lines = st.slider("Líneas", 2, 10, 4)
        n_months = st.slider("Meses de historia", 12, 60, 36)
        seed = int(st.number_input("Semilla", 0, 9999, 7, step=1))
        brk = st.checkbox("Simular reajuste de estándar en Línea 1", False)
        hist_long, lines_raw = data.generate_example(n_lines, n_months, seed, brk)
        sig = f"ej-{n_lines}-{n_months}-{seed}-{brk}"
    else:
        t_hist, t_lines = data.templates()
        st.download_button("Plantilla histórico.csv", t_hist, "historico.csv", "text/csv")
        st.download_button("Plantilla lineas.csv", t_lines, "lineas.csv", "text/csv")
        f_hist = st.file_uploader("Histórico mensual (CSV)", type="csv")
        f_lines = st.file_uploader("Parámetros de líneas (CSV)", type="csv")
        if not (f_hist and f_lines):
            st.info("Carga ambos archivos para continuar.")
            st.stop()
        hist_long, lines_raw = pd.read_csv(f_hist), pd.read_csv(f_lines)
        sig = f"csv-{f_hist.name}-{f_hist.size}-{f_lines.name}-{f_lines.size}"

try:
    wide, data_warns = data.to_wide(hist_long)
    lines_raw = data.validate_lines(lines_raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

names = [l for l in lines_raw["linea"] if l in wide.columns]
dropped = sorted(set(lines_raw["linea"]) ^ set(wide.columns))
if dropped:
    data_warns.append(f"Líneas sin correspondencia entre histórico y parámetros (se excluyen): {dropped}")
if len(names) < 2:
    st.error("Se requieren al menos 2 líneas con histórico y parámetros.")
    st.stop()
wide = wide[names]
lines_raw = lines_raw.set_index("linea").loc[names].reset_index()
flags = data.check_standard_breaks(wide)

tab_d, tab_v, tab_r, tab_f, tab_s, tab_c = st.tabs(
    ["Datos y líneas", "Vistas", "Resultados", "Frontera eficiente", "Sensibilidad", "Comentario y exportación"])

with tab_d:
    st.subheader("Parámetros de líneas (editables)")
    lines_df = st.data_editor(lines_raw, key=f"lines-{sig}", hide_index=True, num_rows="fixed",
                              disabled=["linea"], width="stretch")
    cap = model.capacity_ton(lines_df)
    vol = lines_df["volumen_actual_ton_mes"].to_numpy(float)
    if vol.sum() <= 0:
        st.error("El volumen actual total debe ser mayor que cero.")
        st.stop()
    w0 = vol / vol.sum()
    over = [names[i] for i in range(len(names)) if vol[i] > cap[i] + 1e-6]
    if over:
        st.warning(f"Volumen actual por encima de la capacidad efectiva en: {', '.join(over)}. "
                   "Revisar tasa, horas u OEE.")

# ------------------------------------------------------------------ 2. parámetros
with st.sidebar:
    st.header("2. Parámetros del modelo")
    cov_method = st.selectbox("Covarianza", ["auto", "muestral", "ledoit_wolf"],
                              help="auto usa Ledoit-Wolf cuando meses < 10 × líneas.")
    try:
        sigma, cov_used, shrink = model.estimate_covariance(wide.values, cov_method)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    T = len(wide)
    tau = st.number_input("τ (incertidumbre del equilibrio)", 0.001, 1.0, float(round(1 / T, 4)),
                          step=0.005, format="%.4f", key=f"tau-{sig}", help="Valor inicial = 1 / meses.")
    sig_p = float(np.sqrt(w0 @ sigma @ w0))
    hist_p = float(w0 @ wide.mean().to_numpy())
    default_target = max(0.01, round(hist_p if hist_p > 0 else 0.5 * sig_p, 2))
    target = st.number_input("Ahorro implícito de la asignación actual (USD/ton)", 0.01, 1000.0,
                             default_target, step=0.1, key=f"tgt-{sig}",
                             help="Calibra δ = ahorro / σ². Por defecto: promedio histórico ponderado "
                                  "(o 0.5σ si el histórico es desfavorable).")
    delta_auto = model.calibrate_delta(w0, sigma, target)
    if st.checkbox("Sobrescribir δ manualmente"):
        delta = st.number_input("δ (aversión al riesgo)", 1e-6, 1e6, float(round(delta_auto, 4)), format="%.4f")
    else:
        delta = delta_auto
        st.caption(f"δ calibrado = {delta:.4f}")

    st.header("3. Restricciones")
    demand = st.number_input("Demanda de la familia (ton/mes, SF/CFR)", 1.0, 1e9, float(vol.sum()),
                             step=100.0, key=f"dem-{sig}")
    use_cap = st.checkbox("Capacidad máxima (tasa × horas × OEE)", True)
    use_min = st.checkbox("Volumen mínimo por línea", True)
    use_band = st.checkbox("Cambio máximo vs asignación actual", True)
    band = st.slider("Cambio máximo (puntos de participación)", 1, 50, 10, disabled=not use_band)

lo, hi, cap, issues = model.volume_bounds(lines_df, demand, w0, band, use_cap, use_min, use_band)

with tab_d:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Varianza favorable mensual (USD/ton)")
        st.plotly_chart(px.line(wide, labels={"value": "USD/ton", "periodo": ""}), width="stretch")
    with c2:
        st.subheader("Correlación entre líneas")
        corr = pd.DataFrame(sigma, index=names, columns=names)
        d_ = np.sqrt(np.diag(sigma))
        corr = corr / np.outer(d_, d_)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu"),
                        width="stretch")
    stats = pd.DataFrame({"media_usd_ton": wide.mean(), "desv_usd_ton": np.sqrt(np.diag(sigma)),
                          "capacidad_ton_mes": cap, "volumen_actual_ton_mes": vol, "peso_actual": w0},
                         index=names)
    st.dataframe(stats.style.format({"media_usd_ton": "{:+.2f}", "desv_usd_ton": "{:.2f}",
                                     "capacidad_ton_mes": "{:,.0f}", "volumen_actual_ton_mes": "{:,.0f}",
                                     "peso_actual": "{:.1%}"}), width="stretch")
    st.caption(f"Covarianza: {cov_used}" + (f" (contracción {shrink:.2f})" if cov_used == "ledoit_wolf" else "")
               + f" · {T} meses · {len(names)} líneas")
    for w in data_warns:
        st.warning(w)
    for f in flags:
        st.warning("Posible reajuste de estándar. " + f)

# ------------------------------------------------------------------ vistas
means = wide.mean()
default_views = [{"tipo": "Absoluta", "linea_a": names[0], "linea_b": None,
                  "valor": round(float(means.iloc[0]) + 2.0, 2), "confianza_pct": 60.0,
                  "justificacion": "Ejemplo: nuevo llenador reduce scrap"},
                 {"tipo": "Relativa", "linea_a": names[1], "linea_b": names[0], "valor": 1.5,
                  "confianza_pct": 50.0, "justificacion": "Ejemplo: cambio de crewing"}]
with tab_v:
    st.subheader("Vistas del controller y de operaciones")
    st.markdown("Absoluta: la línea A tendrá *valor* USD/ton de varianza favorable. "
                "Relativa: la línea A superará a la B por *valor* USD/ton. "
                "Confianza (Idzorek): 0% ignora la vista; 100% la toma como cierta.")
    views_df = st.data_editor(
        pd.DataFrame(default_views), key=f"views-{sig}", num_rows="dynamic", hide_index=True,
        width="stretch",
        column_config={
            "tipo": st.column_config.SelectboxColumn("Tipo", options=["Absoluta", "Relativa"], required=True),
            "linea_a": st.column_config.SelectboxColumn("Línea A", options=names, required=True),
            "linea_b": st.column_config.SelectboxColumn("Línea B (relativa)", options=names),
            "valor": st.column_config.NumberColumn("Valor (USD/ton)", format="%.2f"),
            "confianza_pct": st.column_config.NumberColumn("Confianza %", min_value=0, max_value=100),
            "justificacion": st.column_config.TextColumn("Evidencia / justificación"),
        })

if issues:
    with tab_r:
        st.error("Restricciones infactibles. Ajusta demanda, mínimos, capacidad o banda de cambio:")
        for i in issues:
            st.write("• " + i)
    st.stop()

r = model.run(wide, w0, demand, views_df, tau, delta, sigma, lo, hi)
tab = report.allocation_table(r)

with tab_v:
    if len(r["Q"]):
        pv = pd.DataFrame({"vista": r["labels"], "prior_implícito": r["P"] @ r["pi"], "vista_Q": r["Q"],
                           "posterior": r["P"] @ r["mu"], "confianza": r["conf"], "omega": r["omegas"]})
        st.dataframe(pv.style.format({"prior_implícito": "{:+.2f}", "vista_Q": "{:+.2f}",
                                      "posterior": "{:+.2f}", "confianza": "{:.0%}", "omega": "{:.4g}"}),
                     width="stretch", hide_index=True)
    for w in r["warnings"]:
        st.warning(w)

# ------------------------------------------------------------------ resultados
with tab_r:
    if not r["ok"]:
        st.warning(f"El optimizador reportó: {r['msg']}. Revisa restricciones.")
    g_tot = demand * (r["w_bl"] - w0) @ r["mu"]
    g_con = demand * (r["w_constr"] - w0) @ r["mu"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ahorro esperado vs actual", f"{g_tot:+,.0f} USD/mes", f"{g_tot * 12:+,.0f} USD/año")
    k2.metric("Efecto restricciones", f"{g_con:+,.0f} USD/mes")
    k3.metric("Efecto vistas", f"{g_tot - g_con:+,.0f} USD/mes")
    k4.metric("Riesgo USD/ton", f"{r['stats_bl'][1]:.2f}", f"{r['stats_bl'][1] - r['stats_w0'][1]:+.2f}",
              delta_color="inverse")

    st.subheader("Retornos: prior (equilibrio implícito) vs posterior")
    rp = pd.DataFrame({"Prior Π": r["pi"], "Posterior": r["mu"], "Histórico": means.to_numpy()}, index=names)
    st.plotly_chart(px.bar(rp, barmode="group", labels={"value": "USD/ton", "index": ""}),
                    width="stretch")

    st.subheader("Asignación de volumen")
    fmt = {c: "{:.1%}" for c in tab.columns if c.startswith("peso")}
    fmt.update({c: "{:,.0f}" for c in tab.columns if c.startswith(("ton", "ahorro"))})
    fmt.update({c: "{:+.2f}" for c in tab.columns if c.startswith("varianza")})
    st.dataframe(tab.style.format(fmt), width="stretch", hide_index=True)
    wb = pd.DataFrame({"Actual": w0, "Solo restricciones": r["w_constr"], "Black-Litterman": r["w_bl"]},
                      index=names)
    st.plotly_chart(px.bar(wb, barmode="group", labels={"value": "Participación", "index": ""})
                    .update_yaxes(tickformat=".0%"), width="stretch")
    with st.expander("Referencia: pesos sin restricciones"):
        st.write("Resultado teórico sin límites ni suma = 100%. Pesos negativos o suma distinta de 100% "
                 "indican que las vistas empujan más allá de lo operable.")
        st.dataframe(pd.DataFrame({"peso_sin_restricciones": r["w_unc"]}, index=names)
                     .style.format("{:.1%}"))
        st.caption(f"Suma: {r['w_unc'].sum():.1%}")

# ------------------------------------------------------------------ frontera
frontier = model.efficient_frontier(r["mu"], sigma, lo, hi, 30)
with tab_f:
    fig = go.Figure()
    if len(frontier):
        fig.add_trace(go.Scatter(x=frontier.riesgo_usd_ton, y=frontier.ahorro_usd_ton, mode="lines",
                                 name="Frontera (posterior, con restricciones)"))
    for lbl, key, sym in [("Actual", "stats_w0", "circle"), ("Solo restricciones", "stats_constr", "square"),
                          ("Black-Litterman", "stats_bl", "star")]:
        fig.add_trace(go.Scatter(x=[r[key][1]], y=[r[key][0]], mode="markers", name=lbl,
                                 marker=dict(size=14, symbol=sym)))
    fig.update_layout(xaxis_title="Riesgo: desviación de la varianza de la familia (USD/ton)",
                      yaxis_title="Varianza favorable esperada (USD/ton)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Frontera calculada con retornos posteriores y las restricciones activas.")

# ------------------------------------------------------------------ sensibilidad
sens = None
with tab_s:
    if len(r["Q"]) == 0:
        st.info("Agrega al menos una vista válida.")
    else:
        valid_rows = [i for i, row in views_df.reset_index(drop=True).iterrows()
                      if any(lbl.startswith(f"V{i + 1}:") for lbl in r["labels"])]
        pick = st.selectbox("Vista a sensibilizar", valid_rows,
                            format_func=lambda i: next(l for l in r["labels"] if l.startswith(f"V{i + 1}:")))
        sens = model.sensitivity(wide, w0, demand, views_df, tau, delta, sigma, lo, hi, pick)
        long = sens.melt(id_vars=["confianza_pct", "ahorro_mes_usd"], var_name="línea", value_name="peso")
        st.plotly_chart(px.line(long, x="confianza_pct", y="peso", color="línea", markers=True,
                                labels={"confianza_pct": "Confianza de la vista (%)", "peso": "Participación"})
                        .update_yaxes(tickformat=".0%"), width="stretch")
        st.plotly_chart(px.line(sens, x="confianza_pct", y="ahorro_mes_usd", markers=True,
                                labels={"confianza_pct": "Confianza (%)", "ahorro_mes_usd": "USD/mes vs estándar"}),
                        width="stretch")

# ------------------------------------------------------------------ comentario y exportación
with tab_c:
    text = report.commentary(r, views_df, cov_used, flags)
    st.subheader("Comentario tipo controller")
    st.text_area("Editable antes de exportar", text, height=260, key="coment")
    params = {"fuente": fuente, "meses": T, "lineas": len(names), "covarianza": cov_used,
              "contraccion": round(shrink, 4), "tau": tau, "delta": delta,
              "ahorro_implicito_usd_ton": target, "demanda_ton_mes": demand,
              "restriccion_capacidad": use_cap, "restriccion_minimo": use_min,
              "banda_cambio_pts": band if use_band else "desactivada",
              "nota": "Datos de ejemplo ilustrativos" if fuente == "Ejemplo generado" else "Datos cargados por usuario"}
    xls = report.to_excel(r, views_df, lines_df, wide, frontier, sens, st.session_state.get("coment", text), params)
    st.download_button("Descargar resultados (Excel)", xls, "black_litterman_asignacion.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
