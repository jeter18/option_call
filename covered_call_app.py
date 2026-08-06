"""
Covered Call Analyzer — versión GUI (Streamlit)

Misma lógica de cálculo que el script original (covered_call_iberdrola.py):
- Black-Scholes con dividendos continuos para la matriz teórica.
- Volatilidad histórica anualizada a partir de yfinance.
- Comparación contra la prima real introducida por el usuario.

Cambios respecto al original en ESTA entrega:
- Se sustituye el ticker hardcodeado por un buscador con autocompletado (yf.Search).
- Los meses a evaluar y el nº de acciones se seleccionan en la GUI en vez de estar
  hardcodeados / pedirse por input().
- El resto de la lógica (fórmulas, fallback de dividendo, comisión fija, etc.)
  se ha dejado TAL CUAL a propósito — esos puntos los vamos a corregir uno a uno
  en las siguientes iteraciones, tal y como acordamos.

Ejecutar con:  streamlit run covered_call_app.py
"""

import calendar
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
import yfinance as yf

st.set_page_config(page_title="Covered Call Analyzer", page_icon="📈", layout="wide")

# ==============================================================================
# FUNCIONES ORIGINALES (sin cambios de lógica respecto al script de partida)
# ==============================================================================

DIV_YIELD_FALLBACK = 0.033  # 3.3% estándar estimado, solo si no hay dato fiable


def normalizar_dividend_yield(raw):
    """
    yfinance ha cambiado de formato entre versiones para 'dividendYield': a veces
    es una fracción (0.033 = 3.3%) y otras veces ya viene multiplicado por 100
    (3.3 = 3.3%). Aquí se detecta heurísticamente cuál es, y SIEMPRE se devuelve
    también la 'fuente' para que quien lo use sepa si el dato es de fiar, en vez
    de aplicar un fallback en silencio (punto #1/#2 de la revisión).

    Devuelve: (valor_normalizado_como_fraccion, fuente, valor_bruto_original)
    """
    if raw is None or raw == 0:
        return DIV_YIELD_FALLBACK, "fallback (yfinance no devolvió dato)", raw

    valor = raw / 100 if raw > 1 else raw

    if valor > 0.20:  # >20% de dividendo anual sostenido es prácticamente imposible
        return DIV_YIELD_FALLBACK, f"sospechoso (yfinance devolvió {raw!r}, se descarta)", raw

    fuente = "yfinance (fracción)" if raw <= 1 else "yfinance (%, normalizado /100)"
    return valor, fuente, raw


def calcular_volatilidad_y_precio(ticker, dias=252):
    try:
        stock = yf.Ticker(ticker)
        # auto_adjust fijado explícitamente: yfinance ha cambiado su valor por defecto entre
        # versiones (punto #11 de la revisión). True = precios ajustados por splits/dividendos,
        # que es lo correcto para calcular rendimientos logarítmicos y volatilidad histórica.
        df = stock.history(period="2y", auto_adjust=True)
    except Exception as e:
        # Errores de red/API de Yahoo (timeouts, rate limit, etc.) — no confundir con
        # "ticker inválido", son cosas distintas y requieren reacciones distintas.
        raise ConnectionError(
            f"No se pudo conectar con Yahoo Finance para '{ticker}'. "
            f"Puede ser un problema de red temporal o que Yahoo esté limitando peticiones. "
            f"Detalle técnico: {e}"
        ) from e

    if df.empty:
        raise ValueError(
            f"Yahoo Finance no devolvió histórico para '{ticker}'. Revisa que el ticker "
            f"esté en formato Yahoo Finance (ej. IBE.MC para Iberdrola en Madrid, no solo 'IBE')."
        )

    if len(df) < 30:
        raise ValueError(
            f"'{ticker}' solo tiene {len(df)} sesiones de histórico — insuficiente para "
            f"calcular una volatilidad fiable. ¿Es un ticker recién salido a bolsa?"
        )

    df["Rendimientos"] = np.log(df["Close"] / df["Close"].shift(1))
    dias_disponibles = min(dias, len(df) - 1)
    vol_anualizada = df["Rendimientos"].tail(dias_disponibles).std() * np.sqrt(252)
    precio_ultimo = df["Close"].iloc[-1]

    if pd.isna(vol_anualizada) or vol_anualizada <= 0:
        raise ValueError(
            f"La volatilidad calculada para '{ticker}' no es válida ({vol_anualizada!r}). "
            f"Los datos de precio pueden estar corruptos o tener huecos."
        )

    try:
        info = stock.info
    except Exception:
        # El endpoint de info (dividendos, divisa...) es menos fiable que el de histórico.
        # Si falla, seguimos con fallbacks en vez de tirar todo el análisis por la borda.
        info = {}

    raw_div = info.get("dividendYield", None)
    div_yield, fuente_div, raw_div = normalizar_dividend_yield(raw_div)

    divisa = info.get("currency", "EUR")
    return vol_anualizada, precio_ultimo, divisa, div_yield, fuente_div, raw_div


def obtener_tercer_viernes(year, month):
    primer_dia_semana, _ = calendar.monthrange(year, month)
    primer_viernes = (4 - primer_dia_semana) % 7 + 1
    return datetime(year, month, primer_viernes + 14).date()


def black_scholes_call_con_dividendos(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def volatilidad_implicita_desde_precio(precio_mercado, S, K, T, r, q, tol=1e-4, max_iter=100):
    """
    Despeja la volatilidad implícita (IV) a partir de la prima REAL que ofrece el
    mercado, invirtiendo Black-Scholes por bisección (el precio BS es monótono
    creciente en sigma, así que la bisección es robusta y no necesita derivadas).

    Esto sustituye la comparación sesgada de la versión anterior (prima real vs.
    teórico con volatilidad HISTÓRICA — punto #6 de la revisión). La IV es la
    volatilidad que el mercado está pagando *de verdad* ahora mismo; compararla
    contra la histórica da contexto real en vez de una falsa señal de compra/venta.

    Devuelve None si el precio de mercado está por debajo del valor intrínseco
    (dato inconsistente) o si no hay tiempo a vencimiento.
    """
    if T <= 0 or precio_mercado <= 0:
        return None

    intrinseco = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    if precio_mercado <= intrinseco:
        return None

    lo, hi = 1e-4, 5.0  # búsqueda entre 0.01% y 500% de volatilidad anualizada
    if black_scholes_call_con_dividendos(S, K, T, r, q, hi) < precio_mercado:
        return None  # precio de mercado fuera de rango representable, algo raro en el dato

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        precio_bs = black_scholes_call_con_dividendos(S, K, T, r, q, mid)
        if abs(precio_bs - precio_mercado) < tol:
            return mid
        if precio_bs < precio_mercado:
            lo = mid
        else:
            hi = mid
    return mid


# ==============================================================================
# HELPERS PROPIOS DE LA GUI (no tocan la lógica financiera)
# ==============================================================================

@st.cache_data(show_spinner=False, ttl=300)
def buscar_tickers(query: str):
    """Autocompletado de tickers vía Yahoo Finance Search."""
    if not query or len(query) < 2:
        return []
    try:
        resultados = yf.Search(query, max_results=10).quotes
    except Exception:
        return []
    opciones = []
    for r in resultados:
        symbol = r.get("symbol")
        nombre = r.get("shortname") or r.get("longname") or ""
        exch = r.get("exchDisp", "")
        if symbol:
            opciones.append({"symbol": symbol, "label": f"{nombre} — {symbol} ({exch})"})
    return opciones


def proximos_meses(n=12):
    """Genera los próximos n (mes, año) a partir del mes actual."""
    hoy = date.today()
    meses = []
    m, y = hoy.month, hoy.year
    for _ in range(n):
        meses.append((m, y))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return meses


NOMBRES_MES = ["", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# ==============================================================================
# ESTADO DE SESIÓN
# ==============================================================================

for key, default in [
    ("ticker_confirmado", None),
    ("datos_base", None),
    ("tabla_teorica", pd.DataFrame()),
    ("opciones_mapeadas", {}),
    ("comparaciones", {}),  # dict keyed por ID de opción -> resultado evaluado
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================================
# UI — 1. BUSCADOR DE TICKER
# ==============================================================================

st.title("📈 Covered Call Analyzer")
st.caption("Matriz teórica Black-Scholes + contraste con la prima real de tu bróker.")

st.header("1. Elige el activo")

col_search, col_manual = st.columns([2, 1])

with col_search:
    query = st.text_input("Busca por nombre o ticker", placeholder="ej. Iberdrola, Santander, AAPL...")
    ticker_desde_busqueda = None
    if query:
        resultados = buscar_tickers(query)
        if resultados:
            etiquetas = [r["label"] for r in resultados]
            elegido = st.selectbox("Resultados", etiquetas, key="select_busqueda")
            ticker_desde_busqueda = next(r["symbol"] for r in resultados if r["label"] == elegido)
        else:
            st.caption("Sin resultados todavía (o sin conexión). Puedes escribir el ticker a mano →")

with col_manual:
    ticker_manual = st.text_input(
        "...o ticker exacto",
        value=ticker_desde_busqueda or "",
        placeholder="ej. IBE.MC",
        help="Formato Yahoo Finance. Para DeGiro/mercado español suele llevar sufijo .MC",
    )

ticker_final = ticker_manual or ticker_desde_busqueda

if st.button("Confirmar activo", type="primary", disabled=not ticker_final):
    with st.spinner(f"Descargando histórico de {ticker_final}..."):
        try:
            vol, precio, divisa, div_yield, fuente_div, raw_div = calcular_volatilidad_y_precio(ticker_final)
            st.session_state.ticker_confirmado = ticker_final
            st.session_state.datos_base = {
                "vol": vol, "precio": precio, "divisa": divisa,
                "div_yield": div_yield, "fuente_div": fuente_div, "raw_div": raw_div,
            }
            st.session_state.tabla_teorica = pd.DataFrame()
            st.session_state.opciones_mapeadas = {}
        except ConnectionError as e:
            st.error(f"🔌 Problema de conexión con Yahoo Finance: {e}")
            st.caption("Prueba a esperar un momento y reintentar — Yahoo a veces limita peticiones.")
        except ValueError as e:
            st.error(f"⚠️ Ticker no válido o sin datos suficientes: {e}")
        except Exception as e:
            st.error(f"❌ Error inesperado al procesar '{ticker_final}': {e}")
            with st.expander("Detalles técnicos (para depurar)"):
                st.exception(e)

if st.session_state.ticker_confirmado:
    d = st.session_state.datos_base
    st.success(f"Activo: **{st.session_state.ticker_confirmado}**")
    c1, c2 = st.columns(2)
    c1.metric("Precio actual", f"{d['precio']:.2f} {d['divisa']}")
    c2.metric("Volatilidad histórica anualizada", f"{d['vol']*100:.2f}%")

    es_fiable = d["fuente_div"].startswith("yfinance")
    aviso = st.warning if not es_fiable else st.caption
    aviso(f"Dividend yield — fuente: **{d['fuente_div']}** (dato bruto de yfinance: `{d['raw_div']!r}`)")

    div_yield_editado = st.number_input(
        "Dividend yield a usar en el cálculo (%) — revisa/corrige si no te cuadra",
        min_value=0.0, max_value=20.0,
        value=round(d["div_yield"] * 100, 3),
        step=0.05, format="%.3f",
    ) / 100
    st.session_state.datos_base["div_yield"] = div_yield_editado

# ==============================================================================
# UI — 2. PARÁMETROS Y MATRIZ TEÓRICA
# ==============================================================================

if st.session_state.ticker_confirmado:
    st.header("2. Matriz teórica (Black-Scholes)")

    c1, c2, c3 = st.columns(3)
    with c1:
        tasa_libre_riesgo = st.number_input(
            "Tasa libre de riesgo (%)", value=2.25, step=0.05, format="%.2f",
            help="Referencia: facilidad de depósito del BCE, 2,25% desde el 17-jun-2026. Ajusta si usas otra referencia (Euribor, letra del Tesoro).",
        ) / 100
    with c2:
        meses_disponibles = proximos_meses(12)
        etiquetas_meses = [f"{NOMBRES_MES[m]} {y}" for m, y in meses_disponibles]
        default_idx = [0, 1, 3] if len(meses_disponibles) > 3 else list(range(len(meses_disponibles)))
        seleccion_meses = st.multiselect(
            "Vencimientos a evaluar",
            etiquetas_meses,
            default=[etiquetas_meses[i] for i in default_idx],
        )
        meses_evaluar = [meses_disponibles[etiquetas_meses.index(s)] for s in seleccion_meses]
    with c3:
        precio_actual = st.session_state.datos_base["precio"]
        base_strike = round(precio_actual * 2) / 2
        # Rango amplio de candidatos en pasos de 0.5, el usuario elige cuáles evaluar
        strikes_candidatos = [round(base_strike + i * 0.5, 2) for i in range(-6, 17)]
        strikes_default = [round(base_strike + i * 0.5, 2) for i in range(-1, 5)]
        strikes_seleccionados = st.multiselect(
            "Strikes a evaluar (pasos de 0.5)",
            strikes_candidatos,
            default=[s for s in strikes_default if s in strikes_candidatos],
            help="Nota: estos strikes son una rejilla sintética de 0.5 en 0.5, no la cadena real "
                 "de tu bróker (punto #8 pendiente). Solo se calculan los que estén por encima "
                 "del precio actual (fuera de dinero).",
        )

    if st.button("Calcular matriz teórica") and meses_evaluar and strikes_seleccionados:
        try:
            d = st.session_state.datos_base
            precio_actual = d["precio"]
            fecha_hoy = date.today()

            strikes_omitidos_itm = []
            filas = []
            opciones_mapeadas = {}
            contador_id = 1
            for month, year in meses_evaluar:
                fecha_venc = obtener_tercer_viernes(year, month)
                dias_restantes = (fecha_venc - fecha_hoy).days
                if dias_restantes <= 0:
                    continue
                tiempo_anos = dias_restantes / 365.0

                for strike in strikes_seleccionados:
                    distancia_otm = ((strike - precio_actual) / precio_actual) * 100
                    if distancia_otm < 0:
                        if strike not in strikes_omitidos_itm:
                            strikes_omitidos_itm.append(strike)
                        continue

                    precio_teorico = black_scholes_call_con_dividendos(
                        precio_actual, strike, tiempo_anos,
                        tasa_libre_riesgo, d["div_yield"], d["vol"],
                    )
                    rend_anualizado = (precio_teorico / precio_actual) * (365.0 / dias_restantes) * 100.0

                    filas.append({
                        "ID": contador_id,
                        "Vencimiento": fecha_venc.strftime("%d-%b-%Y"),
                        "Días": dias_restantes,
                        "Strike": strike,
                        "Dist. OTM %": round(distancia_otm, 2),
                        "Teórico BS": round(precio_teorico, 3),
                        "Rend. Anualizado %": round(rend_anualizado, 2),
                    })
                    opciones_mapeadas[contador_id] = {
                        "vencimiento": fecha_venc.strftime("%d-%b-%Y"),
                        "dias": dias_restantes,
                        "strike": strike,
                        "otm": distancia_otm,
                        "teorico": precio_teorico,
                    }
                    contador_id += 1

            if strikes_omitidos_itm:
                st.info(
                    f"Strikes ignorados por estar en/por debajo del precio actual (no son OTM): "
                    f"{', '.join(str(s) for s in sorted(strikes_omitidos_itm))}"
                )

            if not filas:
                st.warning("Ningún vencimiento/strike seleccionado produjo resultados válidos. Revisa la selección.")

            st.session_state.tabla_teorica = pd.DataFrame(filas)
            st.session_state.opciones_mapeadas = opciones_mapeadas
        except Exception as e:
            st.error(f"❌ Error calculando la matriz teórica: {e}")
            with st.expander("Detalles técnicos (para depurar)"):
                st.exception(e)

    if not st.session_state.tabla_teorica.empty:
        st.dataframe(st.session_state.tabla_teorica, use_container_width=True, hide_index=True)

# ==============================================================================
# UI — 3. CONTRASTE CON PRIMA REAL DEL BRÓKER
# ==============================================================================

if st.session_state.opciones_mapeadas:
    st.header("3. Contraste con la prima real (DeGiro)")
    st.caption("Evalúa varias opciones y compáralas en la tabla de abajo antes de decidir.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ids_disponibles = list(st.session_state.opciones_mapeadas.keys())
        id_sel = st.selectbox(
            "Opción (ID de la tabla de arriba)",
            ids_disponibles,
            format_func=lambda i: (
                f"ID {i} — {st.session_state.opciones_mapeadas[i]['vencimiento']} "
                f"strike {st.session_state.opciones_mapeadas[i]['strike']:.2f}"
            ),
        )
    with c2:
        num_acciones = st.number_input("Acciones en cartera", min_value=0, value=100, step=1)
    with c3:
        precio_broker = st.number_input("Prima real de DeGiro", min_value=0.0, value=0.0, step=0.01, format="%.3f")

    with st.expander("Comisiones DEGIRO (revisa/ajusta si han cambiado)"):
        cc1, cc2 = st.columns(2)
        with cc1:
            comision_apertura = st.number_input(
                "Comisión por contrato al vender la call (€)",
                min_value=0.0, value=0.75, step=0.05, format="%.2f",
                help="Confirmado: opciones/futuros MEFF a 0,75€/contrato (tarifa vigente en 2026). "
                     "Revisa la página de tarifas de DEGIRO si operas en otro mercado.",
            )
        with cc2:
            comision_asignacion = st.number_input(
                "Comisión por ejercicio/asignación (€/contrato)",
                min_value=0.0, value=1.00, step=0.05, format="%.2f",
                help="DEGIRO cobra 1€/contrato si la opción se ejerce, se asigna o se liquida en "
                     "efectivo. No aplica si la call vence sin valor.",
            )
        incluir_asignacion = st.checkbox(
            "Incluir comisión de asignación en el ingreso neto (asume escenario de asignación)",
            value=False,
        )

    contratos = num_acciones // 100

    if 0 < num_acciones < 100:
        st.caption(f"⚠️ {num_acciones} acciones no llegan a cubrir 1 contrato (se necesitan 100). Contratos: 0.")

    puede_anadir = contratos > 0 and precio_broker > 0

    b1, b2 = st.columns([1, 1])
    with b1:
        anadir = st.button(
            "➕ Añadir a comparación", type="primary", use_container_width=True,
            disabled=not puede_anadir,
        )
    with b2:
        limpiar = st.button("🗑️ Limpiar comparación", use_container_width=True)

    if not puede_anadir:
        motivos = []
        if contratos == 0:
            motivos.append("necesitas al menos 100 acciones para cubrir 1 contrato")
        if precio_broker <= 0:
            motivos.append("introduce una prima real mayor que 0")
        st.caption("No se puede añadir: " + "; ".join(motivos) + ".")

    if limpiar:
        st.session_state.comparaciones = {}

    if anadir and puede_anadir:
        d = st.session_state.datos_base
        datos_opc = st.session_state.opciones_mapeadas[id_sel]
        dias = datos_opc["dias"]
        teorico = datos_opc["teorico"]
        precio_actual = d["precio"]
        vol_historica = d["vol"]

        rend_real = (precio_broker / precio_actual) * (365.0 / dias) * 100.0
        comision_total = comision_apertura * contratos
        if incluir_asignacion:
            comision_total += comision_asignacion * contratos
        ingreso_neto = (precio_broker * 100 * contratos) - comision_total

        # Volatilidad implícita despejada de TU prima real (punto #6): sustituye el
        # "edge vs. teórico histórico" (sesgado, casi siempre positivo por la prima
        # de riesgo de volatilidad) por un dato informativo real: qué volatilidad
        # está pagando el mercado ahora mismo, comparada con la histórica.
        tiempo_anos = dias / 365.0
        iv = volatilidad_implicita_desde_precio(
            precio_broker, precio_actual, datos_opc["strike"], tiempo_anos,
            tasa_libre_riesgo, d["div_yield"],
        )
        prima_riesgo_vol = (iv - vol_historica) * 100 if iv is not None else None

        # Clave única: misma opción con distinto nº de acciones/prima se trata como
        # entradas separadas para poder comparar escenarios sobre el mismo strike.
        clave = f"{id_sel}_{num_acciones}_{precio_broker}_{incluir_asignacion}"
        st.session_state.comparaciones[clave] = {
            "ID": id_sel,
            "Vencimiento": datos_opc["vencimiento"],
            "Días": dias,
            "Strike": datos_opc["strike"],
            "Dist. OTM %": round(datos_opc["otm"], 2),
            "Teórico BS (vol. hist.)": round(teorico, 3),
            "Prima Real": round(precio_broker, 3),
            "Rend. Anualizado Real %": round(rend_real, 2),
            "IV Implícita %": round(iv * 100, 2) if iv is not None else None,
            "Vol. Histórica %": round(vol_historica * 100, 2),
            "Prima Riesgo Vol. %": round(prima_riesgo_vol, 2) if prima_riesgo_vol is not None else None,
            "Contratos": contratos,
            "Ingreso Neto": round(ingreso_neto, 2),
        }
        if iv is None:
            st.warning(
                "No se ha podido despejar la volatilidad implícita con esa prima "
                "(precio de mercado inconsistente con el modelo, p.ej. por debajo del "
                "valor intrínseco). Revisa la prima introducida."
            )

    if st.session_state.comparaciones:
        st.subheader("Comparación de operaciones evaluadas")
        df_comp = pd.DataFrame(list(st.session_state.comparaciones.values()))
        df_comp = df_comp.sort_values("Rend. Anualizado Real %", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_comp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rend. Anualizado Real %": st.column_config.ProgressColumn(
                    "Rend. Anualizado Real %",
                    min_value=0,
                    max_value=max(30.0, float(df_comp["Rend. Anualizado Real %"].max())),
                    format="%.2f%%",
                ),
            },
        )
        st.caption(
            "**Prima Riesgo Vol. %** = IV implícita (despejada de tu prima real) menos volatilidad "
            "histórica. Suele ser positiva de forma estructural (el mercado casi siempre cobra algo "
            "extra sobre la volatilidad pasada) — no la interpretes por sí sola como 'oportunidad', "
            "es contexto, no una señal de compra/venta."
        )

        mejor = df_comp.iloc[0]
        st.success(
            f"Mejor rendimiento anualizado real: **ID {mejor['ID']}** "
            f"({mejor['Vencimiento']}, strike {mejor['Strike']:.2f}) con {mejor['Rend. Anualizado Real %']:.2f}%"
        )

        st.bar_chart(df_comp.set_index("ID")["Rend. Anualizado Real %"])
else:
    if st.session_state.ticker_confirmado:
        st.info("Calcula primero la matriz teórica (paso 2) para poder evaluar una opción concreta.")
