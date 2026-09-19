# Black-Litterman para asignación de volumen entre líneas

App en Streamlit que reparte el volumen de una familia de producto entre líneas intercambiables de una misma planta. Minimiza el costo por unidad combinando la asignación actual (equilibrio) con vistas de controller y operaciones ponderadas por confianza.

## Definición del modelo

| Elemento | Definición en la app |
|---|---|
| Activo | Línea que fabrica la familia de producto |
| Peso w | Participación de la línea en el volumen de la familia |
| Retorno | Varianza favorable vs estándar, USD/ton = costo estándar − costo real |
| Σ | Covarianza mensual de esa varianza (muestral o Ledoit-Wolf) |
| Equilibrio | Asignación actual: Π = δ Σ w₀ |
| δ | Calibrada: δ = ahorro implícito / (w₀' Σ w₀); editable |
| τ | 1 / meses por defecto; editable |
| Vistas | Absolutas o relativas en USD/ton, con confianza 0-100% |
| Ω | Método de Idzorek (2005), resuelto numéricamente por vista |
| Posterior | μ = Π + τΣP'(PτΣP' + Ω)⁻¹(Q − PΠ) |
| Optimización | max w'μ − δ/2 w'Σw, con Σw = 1 y límites por línea |

La optimización usa Σ y no la covarianza posterior (convención de Idzorek). Así, sin vistas y sin restricciones activas, el modelo devuelve la asignación actual exacta; cualquier cambio se explica por vistas o por restricciones.

Restricciones: capacidad máxima = tasa nominal (ton/h) × horas disponibles × OEE; volumen mínimo por línea; cambio máximo en puntos de participación contra la asignación actual; volumen total igual a la demanda capturada (SF/CFR). La app revisa factibilidad antes de optimizar.

## Salidas

Tablas prior vs posterior y asignación actual vs óptima en %, toneladas y USD; descomposición del ahorro en efecto restricciones y efecto vistas; frontera eficiente con los tres puntos marcados; sensibilidad de la asignación a la confianza de cada vista; comentario en formato controller y exportación a Excel (10 hojas).

## Datos

Por defecto la app genera datos sintéticos (2 a 10 líneas, 12 a 60 meses). Son ilustrativos. Para datos reales, carga dos CSV (plantillas descargables en la app):

`historico.csv`: periodo (AAAA-MM), linea, costo_estandar_usd_ton, costo_real_usd_ton, toneladas (opcional). Alternativa: columna varianza_favorable_usd_ton. Si hay varias filas por línea y mes, se promedian ponderando por toneladas.

`lineas.csv`: linea, volumen_actual_ton_mes, tasa_nominal_ton_h, horas_disponibles_mes, oee_pct, minimo_ton_mes.

La app marca líneas cuya media cambia más de 3 errores estándar entre la primera y la segunda mitad del histórico (posible reajuste de estándar o cambio de BOM).

## Ejecutar local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q tests        # 6 pruebas del núcleo matemático
```

## Publicar en GitHub y Streamlit Community Cloud

```bash
git init
git add .
git commit -m "Modelo Black-Litterman de asignación de volumen"
git branch -M main
git remote add origin https://github.com/<usuario>/bl-asignacion-volumen.git
git push -u origin main
```

En share.streamlit.io: Create app → repositorio → rama main → archivo app.py → Deploy.

No subas datos reales de costos al repositorio ni a una app pública sin aprobación de la política de datos de tu empresa.

## Límites

El modelo asume que la varianza vs estándar por línea es lineal en el volumen. Si la absorción de fijos cambia al mover toneladas, ese efecto no está capturado. Tampoco modela costos de changeover, curvas de aprendizaje ni restricciones de SKU específicos dentro de la familia.

## Referencias

Black, F. y Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal.
He, G. y Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios. Goldman Sachs.
Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model.
Ledoit, O. y Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis.
