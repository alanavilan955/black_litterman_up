"""Núcleo del modelo Black-Litterman para asignación de volumen entre líneas.

Retorno de una línea = varianza favorable vs costo estándar, USD/ton (estándar − real).
Peso = participación de la línea en el volumen de la familia de producto.
Archivo único (sin subcarpetas) para que funcione al subirlo por la web de GitHub.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, minimize_scalar

# ======================================================================
# MODEL
# ======================================================================
OMEGA_ZERO_CONF = 1e10   # multiplicador de s = p·τΣ·p' para confianza 0%
OMEGA_FULL_CONF = 1e-10  # multiplicador de s para confianza 100%


# ---------------------------------------------------------------- covarianza
def estimate_covariance(returns: np.ndarray, method: str = "auto"):
    """Devuelve (Σ, método usado, intensidad de contracción)."""
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    if T < n + 2:
        raise ValueError(f"Se requieren al menos {n + 2} meses para {n} líneas; hay {T}.")
    chosen = method
    if method == "auto":
        chosen = "ledoit_wolf" if T < 10 * n else "muestral"
    if chosen == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        return lw.covariance_, "ledoit_wolf", float(lw.shrinkage_)
    return np.cov(X, rowvar=False, ddof=1), "muestral", 0.0


# ---------------------------------------------------------------- equilibrio
def calibrate_delta(w: np.ndarray, sigma: np.ndarray, implied_saving: float) -> float:
    """δ tal que la asignación actual 'rinda' implied_saving USD/ton: δ = ahorro / σp²."""
    var_p = float(w @ sigma @ w)
    if implied_saving <= 0 or var_p <= 0:
        raise ValueError("El ahorro implícito y la varianza del portafolio deben ser positivos.")
    return implied_saving / var_p


def implied_returns(delta: float, sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Π = δ Σ w: varianza favorable que la asignación actual supone óptima."""
    return delta * sigma @ w


# ---------------------------------------------------------------- vistas
def build_views(views_df: pd.DataFrame, lines: list[str]):
    """Convierte la tabla de vistas en P, Q, confianza y etiquetas. Devuelve también avisos."""
    P, Q, C, labels, warns = [], [], [], [], []
    idx = {l: i for i, l in enumerate(lines)}
    if views_df is None or len(views_df) == 0:
        return np.zeros((0, len(lines))), np.zeros(0), np.zeros(0), [], warns
    for r, row in views_df.reset_index(drop=True).iterrows():
        tipo = str(row.get("tipo", "")).strip()
        a, b = row.get("linea_a"), row.get("linea_b")
        val, conf = row.get("valor"), row.get("confianza_pct")
        if pd.isna(val) or pd.isna(conf) or a not in idx:
            warns.append(f"Vista {r + 1} ignorada: faltan línea A, valor o confianza.")
            continue
        p = np.zeros(len(lines))
        p[idx[a]] = 1.0
        if tipo == "Relativa":
            if b not in idx or b == a:
                warns.append(f"Vista {r + 1} ignorada: la vista relativa necesita una línea B distinta.")
                continue
            p[idx[b]] = -1.0
            labels.append(f"V{r + 1}: {a} − {b} = {float(val):+.2f}")
        else:
            labels.append(f"V{r + 1}: {a} = {float(val):+.2f}")
        P.append(p)
        Q.append(float(val))
        C.append(min(max(float(conf), 0.0), 100.0) / 100.0)
    if not P:
        return np.zeros((0, len(lines))), np.zeros(0), np.zeros(0), [], warns
    return np.array(P), np.array(Q), np.array(C), labels, warns


def omega_closed_form(s: float, c: float) -> float:
    """ω = s(1-c)/c. Equivale al método de Idzorek cuando la optimización no tiene restricciones."""
    if c <= 0:
        return s * OMEGA_ZERO_CONF
    if c >= 1:
        return s * OMEGA_FULL_CONF
    return s * (1.0 - c) / c


def idzorek_omega(pi, sigma, tau, delta, w_mkt, p, q, c) -> float:
    """Método de Idzorek (2005): busca ω tal que la inclinación de pesos de la vista
    sea c × la inclinación que tendría con 100% de confianza."""
    ts = tau * sigma
    s = float(p @ ts @ p)
    if c <= 0 or c >= 1:
        return omega_closed_form(s, c)
    gap = float(q - p @ pi)
    if abs(gap) < 1e-12:  # la vista coincide con el equilibrio: ω no mueve pesos
        return omega_closed_form(s, c)
    inv = np.linalg.inv(delta * sigma)

    def w_of(om):
        return inv @ (pi + ts @ p * (gap / (s + om)))

    w100 = w_of(0.0)
    target = w_mkt + c * (w100 - w_mkt)
    scale = float(np.sum((w100 - w_mkt) ** 2))

    def f(x):
        return float(np.sum((w_of(np.exp(x)) - target) ** 2)) / scale

    res = minimize_scalar(f, bounds=(np.log(s * 1e-10), np.log(s * 1e10)),
                          method="bounded", options={"xatol": 1e-10})
    return float(np.exp(res.x))


def posterior(pi, sigma, tau, P, Q, omega_diag):
    """Retornos y covarianza posteriores (He y Litterman, 1999)."""
    ts = tau * sigma
    if P.shape[0] == 0:
        return pi.copy(), sigma + ts
    A = P @ ts @ P.T + np.diag(omega_diag)
    K = ts @ P.T @ np.linalg.inv(A)
    mu = pi + K @ (Q - P @ pi)
    M = ts - K @ P @ ts
    return mu, sigma + M


# ---------------------------------------------------------------- restricciones
def capacity_ton(lines_df: pd.DataFrame) -> np.ndarray:
    """Capacidad efectiva ton/mes = tasa nominal (ton/h) × horas disponibles × OEE."""
    return (lines_df["tasa_nominal_ton_h"] * lines_df["horas_disponibles_mes"]
            * lines_df["oee_pct"] / 100.0).to_numpy(dtype=float)


def volume_bounds(lines_df, demand, w0, band_pts, use_cap=True, use_min=True, use_band=True):
    """Límites de participación por línea y lista de problemas de factibilidad."""
    n = len(lines_df)
    names = lines_df["linea"].tolist()
    lo, hi = np.zeros(n), np.ones(n)
    cap = capacity_ton(lines_df)
    if use_min:
        lo = np.maximum(lo, lines_df["minimo_ton_mes"].to_numpy(float) / demand)
    if use_cap:
        hi = np.minimum(hi, cap / demand)
    if use_band:
        b = band_pts / 100.0
        lo = np.maximum(lo, w0 - b)
        hi = np.minimum(hi, w0 + b)
    issues = []
    for i in range(n):
        if lo[i] > hi[i] + 1e-12:
            issues.append(f"{names[i]}: mínimo ({lo[i] * demand:,.0f} ton) mayor que máximo "
                          f"({hi[i] * demand:,.0f} ton) con las restricciones activas.")
    if lo.sum() > 1 + 1e-9:
        issues.append(f"La suma de mínimos ({lo.sum() * demand:,.0f} ton) excede la demanda "
                      f"({demand:,.0f} ton).")
    if hi.sum() < 1 - 1e-9:
        issues.append(f"La suma de máximos ({hi.sum() * demand:,.0f} ton) no cubre la demanda "
                      f"({demand:,.0f} ton).")
    return lo, hi, cap, issues


# ---------------------------------------------------------------- optimización
def optimize_allocation(mu, sigma, delta, lo, hi, w_start):
    """max w'μ − δ/2 w'Σw  s.a.  Σw = 1,  lo ≤ w ≤ hi."""
    n = len(mu)
    x0 = np.clip(w_start, lo, hi)
    res = minimize(
        lambda w: -(w @ mu - 0.5 * delta * w @ sigma @ w),
        x0,
        jac=lambda w: -(mu - delta * sigma @ w),
        bounds=list(zip(lo, hi)),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x, bool(res.success), res.message


def efficient_frontier(mu, sigma, lo, hi, n_points=30) -> pd.DataFrame:
    """Frontera de mínima varianza para ahorros objetivo entre el mínimo y el máximo factibles."""
    n = len(mu)
    bounds = list(zip(lo, hi))
    eq = dict(A_eq=np.ones((1, n)), b_eq=[1.0], bounds=bounds, method="highs")
    rmin, rmax = linprog(mu, **eq), linprog(-mu, **eq)
    if not (rmin.success and rmax.success):
        return pd.DataFrame(columns=["ahorro_usd_ton", "riesgo_usd_ton"])
    targets = np.linspace(mu @ rmin.x, mu @ rmax.x, n_points)
    rows, x0 = [], rmin.x
    for t in targets:
        res = minimize(
            lambda w: w @ sigma @ w, x0, jac=lambda w: 2 * sigma @ w, bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                         {"type": "eq", "fun": lambda w, t=t: w @ mu - t}],
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
        )
        if res.success:
            x0 = res.x
            rows.append({"ahorro_usd_ton": float(t), "riesgo_usd_ton": float(np.sqrt(res.x @ sigma @ res.x))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- orquestación
def run(returns_wide: pd.DataFrame, w0, demand, views_df, tau, delta, sigma, lo, hi):
    """Ejecuta el modelo completo con Σ ya estimada. Devuelve un diccionario de resultados."""
    lines = list(returns_wide.columns)
    pi = implied_returns(delta, sigma, w0)
    P, Q, C, labels, warns = build_views(views_df, lines)
    omegas = np.array([idzorek_omega(pi, sigma, tau, delta, w0, P[k], Q[k], C[k])
                       for k in range(len(Q))])
    mu, sigma_post = posterior(pi, sigma, tau, P, Q, omegas)
    # Convención de Idzorek (2005): se optimiza con Σ, no con Σ posterior, para que sin vistas
    # ni restricciones activas el modelo devuelva exactamente la asignación actual.
    w_constr, ok1, _ = optimize_allocation(pi, sigma, delta, lo, hi, w0)   # solo restricciones
    w_bl, ok2, msg = optimize_allocation(mu, sigma, delta, lo, hi, w0)     # restricciones + vistas
    w_unc = np.linalg.solve(delta * sigma, mu)                              # referencia sin restricciones

    def stats(w):
        return float(w @ mu), float(np.sqrt(w @ sigma @ w))

    return dict(
        lines=lines, pi=pi, mu=mu, sigma=sigma, sigma_post=sigma_post, P=P, Q=Q, conf=C,
        omegas=omegas, labels=labels, warnings=warns, w0=w0, w_constr=w_constr, w_bl=w_bl,
        w_unc=w_unc, ok=ok1 and ok2, msg=msg, demand=demand, delta=delta, tau=tau,
        stats_w0=stats(w0), stats_constr=stats(w_constr), stats_bl=stats(w_bl),
    )


def sensitivity(returns_wide, w0, demand, views_df, tau, delta, sigma, lo, hi, view_row, steps=11):
    """Recalcula la asignación óptima moviendo la confianza de una vista entre 0% y 100%."""
    rows = []
    for c in np.linspace(0, 100, steps):
        v = views_df.copy().reset_index(drop=True)
        v.loc[view_row, "confianza_pct"] = c
        r = run(returns_wide, w0, demand, v, tau, delta, sigma, lo, hi)
        row = {"confianza_pct": c, "ahorro_mes_usd": demand * r["stats_bl"][0]}
        row.update({l: r["w_bl"][i] for i, l in enumerate(r["lines"])})
        rows.append(row)
    return pd.DataFrame(rows)

# ======================================================================
# DATA
# ======================================================================
HIST_REQUIRED = ["periodo", "linea"]
HIST_COST_COLS = ["costo_estandar_usd_ton", "costo_real_usd_ton"]
HIST_VAR_COL = "varianza_favorable_usd_ton"
LINE_COLS = ["linea", "volumen_actual_ton_mes", "tasa_nominal_ton_h",
             "horas_disponibles_mes", "oee_pct", "minimo_ton_mes"]


def generate_example(n_lines: int = 4, n_months: int = 36, seed: int = 7,
                     standard_reset: bool = False):
    """Datos sintéticos: varianza mensual vs estándar con un factor común (utilities/commodities).

    Devuelve (histórico en formato largo, tabla de líneas). Los valores son ilustrativos.
    """
    rng = np.random.default_rng(seed)
    names = [f"Línea {i + 1}" for i in range(n_lines)]
    end = pd.Timestamp.today().to_period("M") - 1
    periods = pd.period_range(end=end, periods=n_months, freq="M").astype(str)

    mean = rng.normal(0.5, 1.5, n_lines)          # USD/ton favorable promedio
    vol = rng.uniform(2.0, 7.0, n_lines)          # USD/ton desviación estándar mensual
    load = rng.uniform(0.3, 0.8, n_lines)         # exposición al factor común
    std_cost = rng.uniform(150, 400, n_lines)     # costo estándar USD/ton
    factor = rng.standard_normal(n_months)
    eps = rng.standard_normal((n_months, n_lines))
    var = mean + vol * (factor[:, None] * load + eps * np.sqrt(1 - load ** 2))
    if standard_reset:  # reajuste de estándar a mitad de periodo en la línea 1
        var[n_months // 2:, 0] += 3.0 * vol[0]

    current = rng.uniform(800, 2500, n_lines)
    rate = rng.uniform(4, 10, n_lines)
    hours = np.full(n_lines, 600.0)
    oee = rng.uniform(55, 80, n_lines)
    cap = rate * hours * oee / 100
    need = current * rng.uniform(1.15, 1.5, n_lines)
    rate = np.where(cap < need, need / (hours * oee / 100), rate)

    rows = []
    for t, per in enumerate(periods):
        for i, l in enumerate(names):
            rows.append({
                "periodo": per, "linea": l,
                "costo_estandar_usd_ton": round(std_cost[i], 2),
                "costo_real_usd_ton": round(std_cost[i] - var[t, i], 2),
                "toneladas": round(current[i] * rng.uniform(0.85, 1.15), 1),
            })
    hist = pd.DataFrame(rows)
    lines = pd.DataFrame({
        "linea": names,
        "volumen_actual_ton_mes": current.round(0),
        "tasa_nominal_ton_h": rate.round(2),
        "horas_disponibles_mes": hours,
        "oee_pct": oee.round(1),
        "minimo_ton_mes": (current * rng.uniform(0.3, 0.5, n_lines)).round(0),
    })
    return hist, lines


def to_wide(hist: pd.DataFrame):
    """Convierte el histórico largo a matriz periodo × línea de varianza favorable USD/ton."""
    warns = []
    missing = [c for c in HIST_REQUIRED if c not in hist.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el histórico: {missing}")
    h = hist.copy()
    if HIST_VAR_COL not in h.columns:
        if not all(c in h.columns for c in HIST_COST_COLS):
            raise ValueError(f"El histórico necesita '{HIST_VAR_COL}' o ambas columnas {HIST_COST_COLS}.")
        h[HIST_VAR_COL] = h["costo_estandar_usd_ton"] - h["costo_real_usd_ton"]
    h["periodo"] = h["periodo"].astype(str)
    if "toneladas" in h.columns:  # promedio ponderado por toneladas si hay varias órdenes por mes
        h["_w"] = h[HIST_VAR_COL] * h["toneladas"]
        g = h.groupby(["periodo", "linea"])[["_w", "toneladas"]].sum()
        s = (g["_w"] / g["toneladas"]).rename(HIST_VAR_COL)
    else:
        s = h.groupby(["periodo", "linea"])[HIST_VAR_COL].mean()
    wide = s.unstack("linea").sort_index()
    n_before = len(wide)
    wide = wide.dropna()
    if len(wide) < n_before:
        warns.append(f"Se descartaron {n_before - len(wide)} periodos con líneas sin dato.")
    return wide, warns


def validate_lines(lines: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in LINE_COLS if c not in lines.columns]
    if missing:
        raise ValueError(f"Faltan columnas en la tabla de líneas: {missing}")
    out = lines[LINE_COLS].copy()
    for c in LINE_COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[LINE_COLS[1:]].isna().any().any():
        raise ValueError("La tabla de líneas tiene valores no numéricos o vacíos.")
    if (out["volumen_actual_ton_mes"] < 0).any():
        raise ValueError("El volumen actual no puede ser negativo.")
    return out


def check_standard_breaks(wide: pd.DataFrame, z: float = 3.0):
    """Marca líneas cuya media cambia entre la primera y la segunda mitad (posible reajuste de estándar)."""
    flags = []
    if len(wide) < 12:
        return flags
    h = len(wide) // 2
    a, b = wide.iloc[:h], wide.iloc[h:]
    for c in wide.columns:
        se = np.sqrt(a[c].var(ddof=1) / len(a) + b[c].var(ddof=1) / len(b))
        if se > 0 and abs(b[c].mean() - a[c].mean()) / se > z:
            flags.append(f"{c}: la media pasó de {a[c].mean():+.2f} a {b[c].mean():+.2f} USD/ton "
                         f"entre mitades del periodo. Validar si hubo reajuste de estándar o cambio de BOM.")
    return flags


def templates():
    """CSV de plantilla con una fila de ejemplo cada uno."""
    hist = pd.DataFrame([{"periodo": "2026-01", "linea": "Línea 1", "costo_estandar_usd_ton": 250.0,
                          "costo_real_usd_ton": 247.5, "toneladas": 1500}])
    lines = pd.DataFrame([{"linea": "Línea 1", "volumen_actual_ton_mes": 1500, "tasa_nominal_ton_h": 6.0,
                           "horas_disponibles_mes": 600, "oee_pct": 70.0, "minimo_ton_mes": 600}])
    return hist.to_csv(index=False).encode("utf-8"), lines.to_csv(index=False).encode("utf-8")

# ======================================================================
# REPORT
# ======================================================================
def allocation_table(r) -> pd.DataFrame:
    d = r["demand"]
    df = pd.DataFrame({
        "linea": r["lines"],
        "varianza_prior_usd_ton": r["pi"],
        "varianza_posterior_usd_ton": r["mu"],
        "peso_actual": r["w0"],
        "peso_solo_restricciones": r["w_constr"],
        "peso_black_litterman": r["w_bl"],
        "ton_actual": r["w0"] * d,
        "ton_black_litterman": r["w_bl"] * d,
    })
    df["ton_cambio"] = df["ton_black_litterman"] - df["ton_actual"]
    df["ahorro_mes_actual_usd"] = df["ton_actual"] * df["varianza_posterior_usd_ton"]
    df["ahorro_mes_bl_usd"] = df["ton_black_litterman"] * df["varianza_posterior_usd_ton"]
    return df


def commentary(r, views_df: pd.DataFrame, cov_used: str, flags: list[str]) -> str:
    d = r["demand"]
    tab = allocation_table(r)
    gain = d * (r["w_bl"] - r["w0"]) @ r["mu"]
    gain_constr = d * (r["w_constr"] - r["w0"]) @ r["mu"]
    gain_views = gain - gain_constr
    risk0, risk1 = r["stats_w0"][1], r["stats_bl"][1]
    donors = tab[tab.ton_cambio < -0.5].sort_values("ton_cambio")
    recv = tab[tab.ton_cambio > 0.5].sort_values("ton_cambio", ascending=False)
    moves = ", ".join(f"{x.linea} {x.ton_cambio:+,.0f} ton" for x in pd.concat([recv, donors]).itertuples())
    n_views = len(r["Q"])
    avg_conf = float(np.mean(r["conf"]) * 100) if n_views else 0.0
    kind = "estructural si las vistas se confirman" if n_views else "efecto de restricciones de capacidad"

    lines = [
        f"Reasignar volumen de la familia ({d:,.0f} ton/mes) genera {gain:+,.0f} USD/mes de varianza "
        f"favorable esperada vs estándar ({gain * 12:+,.0f} USD anualizado); driven by {moves or 'sin movimientos materiales'}.",
        f"Descomposición: restricciones operativas {gain_constr:+,.0f} USD/mes; vistas {gain_views:+,.0f} USD/mes.",
        f"Riesgo (desviación de la varianza de la familia): {risk0:.2f} → {risk1:.2f} USD/ton.",
        f"Root cause: REQUIRES VALIDATION. Resultado de modelo basado en {n_views} vista(s) con confianza "
        f"promedio {avg_conf:.0f}% y covarianza {cov_used}.",
        f"Impact: {kind}.",
        "Action: OWNER: TO BE CONFIRMED. Validar con Operaciones capacidad, crewing y changeovers de las "
        "líneas receptoras antes de mover volumen; validar con Planeación que la demanda SF/CFR usada es la vigente.",
        "Monitoring: varianza USD/ton mensual por línea contra el posterior del modelo; OEE y ton/hora de "
        "las líneas receptoras; revisión de confianza de vistas cada trimestre.",
    ]
    if flags:
        lines.append("Data quality: " + " ".join(flags))
    if r["warnings"]:
        lines.append("Vistas no usadas: " + " ".join(r["warnings"]))
    return "\n".join(lines)


def to_excel(r, views_df, lines_df, wide, frontier, sens, text, params: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"parametro": list(params), "valor": [str(v) for v in params.values()]}
                     ).to_excel(xw, sheet_name="Supuestos", index=False)
        allocation_table(r).to_excel(xw, sheet_name="Resultados", index=False)
        views_df.to_excel(xw, sheet_name="Vistas", index=False)
        pd.DataFrame({"vista": r["labels"], "omega": r["omegas"], "confianza": r["conf"]}
                     ).to_excel(xw, sheet_name="Omega", index=False)
        lines_df.to_excel(xw, sheet_name="Lineas", index=False)
        wide.to_excel(xw, sheet_name="Historico_USD_ton")
        pd.DataFrame(r["sigma"], index=r["lines"], columns=r["lines"]).to_excel(xw, sheet_name="Covarianza")
        frontier.to_excel(xw, sheet_name="Frontera", index=False)
        if sens is not None:
            sens.to_excel(xw, sheet_name="Sensibilidad", index=False)
        pd.DataFrame({"comentario": text.split("\n")}).to_excel(xw, sheet_name="Comentario", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 22
    return buf.getvalue()
