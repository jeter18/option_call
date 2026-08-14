# Covered Call Analyzer — Matemática de la aplicación

Este documento explica **todas las fórmulas** que usa `covered_call_app.py`, en el
mismo orden en que aparecen en la interfaz, para que puedas revisarlas variable a
variable. No es documentación de uso — es documentación de cálculo.

---

## 0. Notación usada en todo el documento

| Símbolo | Significado | Variable en el código |
|---|---|---|
| `S` | Precio actual de la acción | `precio_actual` |
| `K` | Strike de la call | `strike` |
| `T` | Tiempo a vencimiento, en años | `tiempo_anos` = `dias_restantes / 365` |
| `r` | Tasa libre de riesgo (anual, continua) | `tasa_libre_riesgo` |
| `q` | Dividend yield (anual, continuo) | `div_yield` |
| `σ` | Volatilidad anualizada | `vol` (histórica) o `sigma` (variable de búsqueda en IV) |
| `N(x)` | Función de distribución normal estándar acumulada | `norm.cdf(x)` (scipy) |

---

## 1. Volatilidad histórica anualizada

**Fuente de datos:** `yfinance`, histórico de 2 años (`period="2y"`), con
`auto_adjust=True` fijado explícitamente (precios ajustados por splits y
dividendos; yfinance ha cambiado este valor por defecto entre versiones, así que
no se deja implícito).

**Paso 1 — Rendimientos logarítmicos diarios:**

```
rendimiento[i] = ln( Close[i] / Close[i-1] )
```

**Paso 2 — Desviación típica anualizada**, usando hasta 252 sesiones (o todas las
disponibles si hay menos):

```
σ_histórica = std( rendimiento[últimas N sesiones] ) × √252
```

El factor `√252` anualiza una desviación típica diaria asumiendo ~252 sesiones
bursátiles al año (estándar del sector).

**Validación:** si el resultado es `NaN`, cero o negativo, la función lanza un
error explícito en vez de continuar con un número inválido.

---

## 2. Dividend yield — normalización

`yfinance` ha cambiado de formato entre versiones para el campo `dividendYield`:
a veces lo devuelve como fracción (`0.033` = 3,3%) y a veces ya multiplicado por
100 (`3.3` = 3,3%). La función `normalizar_dividend_yield()` decide cuál es:

```
si raw es None o 0:
    usar FALLBACK = 3.3%          # con aviso "fallback"
si no:
    valor = raw/100 si raw > 1, si no raw          # heurística de formato
    si valor > 20%:
        usar FALLBACK = 3.3%      # con aviso "sospechoso", el dato se descarta
    si no:
        usar valor                 # con aviso "yfinance (...)"
```

El resultado se muestra siempre en la interfaz junto con su origen, y es
**editable a mano** antes de calcular nada.

---

## 3. Precio de la call — Black-Scholes con dividendo continuo

Función `black_scholes_call_con_dividendos(S, K, T, r, q, σ)`:

```
d1 = [ ln(S/K) + (r − q + σ²/2)·T ] / (σ·√T)
d2 = d1 − σ·√T

Precio_call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
```

Caso límite: si `T ≤ 0` o `σ ≤ 0`, se devuelve el valor intrínseco descontado:

```
Precio_call = max( 0, S·e^(−qT) − K·e^(−rT) )
```

**Dónde se usa `σ`:** en la matriz teórica (sección 2 de la app) se usa siempre
`σ = σ_histórica` (paso 1). Este es el precio "teórico" que ves en la columna
`Teórico BS` — no es una predicción del precio de mercado, es lo que la fórmula
da con volatilidad **pasada**, no con lo que el mercado espera ahora.

---

## 4. Volatilidad implícita (IV) a partir de tu prima real

Cuando introduces la prima real de DeGiro (sección 3), la app despeja qué
volatilidad haría que Black-Scholes diera exactamente ese precio. Como el precio
de una call sube de forma estrictamente monótona con `σ`, esto se resuelve por
**bisección** (no hace falta calcular derivadas):

```
función volatilidad_implicita(precio_mercado, S, K, T, r, q):
    intrínseco = max(0, S·e^(−qT) − K·e^(−rT))
    si precio_mercado ≤ intrínseco:  devolver None   # dato inconsistente

    lo, hi = 0.0001, 5.0   # buscar entre 0.01% y 500% de vol. anual
    repetir hasta 100 veces:
        mid = (lo + hi) / 2
        precio_bs = BS(S, K, T, r, q, mid)
        si |precio_bs − precio_mercado| < 0.0001:  devolver mid
        si precio_bs < precio_mercado:  lo = mid
        si no:                          hi = mid
    devolver mid   # mejor aproximación tras 100 iteraciones
```

**Importante — relación algebraica con el "edge" clásico:** como Black-Scholes es
monótona en `σ`, se cumple exactamente:

```
prima_real > Teórico_BS(σ_histórica)   ⟺   IV_implícita > σ_histórica
```

Es decir, comparar la IV implícita contra la histórica es **la misma
comparación** que comparar la prima real contra el teórico en euros, solo que
expresada en puntos de volatilidad en vez de en euros. La ventaja de expresarlo
así es que es comparable entre strikes/vencimientos distintos (cada uno tiene
una sensibilidad — vega — distinta a la volatilidad). La IV se muestra como dato
informativo, **no** como semáforo de "cara/barata".

---

## 5. Fecha de vencimiento — tercer viernes del mes

Los contratos de opciones estándar vencen el tercer viernes de cada mes:

```
función tercer_viernes(year, month):
    primer_día_semana = día de la semana del día 1 del mes (0=lunes … 6=domingo)
    primer_viernes = (4 − primer_día_semana) mod 7 + 1
    devolver day = primer_viernes + 14   # el tercer viernes es el primero + 2 semanas
```

`DTE` (días a vencimiento) = `fecha_vencimiento − fecha_hoy`, en días naturales.

---

## 6. Strikes evaluados — rejilla sintética

**No son los strikes reales de la cadena de opciones de tu bróker** (no hay
fuente de datos con la cadena real de MEFF integrada). Es una rejilla generada
matemáticamente:

```
strike_base = round(precio_actual × 4) / 4        # redondeo a 0.25 más cercano
candidatos = { strike_base + i × 0.25 : i = −12 … +32 }
```

Tú seleccionas manualmente en la interfaz cuáles de esos candidatos evaluar.
Cualquier strike igual o por debajo del precio actual (no está OTM) se descarta
automáticamente, con aviso de cuáles se han ignorado.

---

## 7. Distancia OTM (Out of The Money)

```
OTM_% = (strike − precio_actual) / precio_actual × 100
```

Positivo = la call está por encima del precio actual (fuera de dinero, lo normal
al vender covered calls). Negativo = strike en/por debajo del precio (se
descarta, ver punto 6).

---

## 8. Rendimiento anualizado (matriz teórica)

Para cada fila de la matriz teórica (sección 2), usando el precio teórico BS
como si fuera la prima:

```
Rend_Anualizado_% = (Precio_teórico / precio_actual) × (365 / DTE) × 100
```

Es una anualización **lineal** simple (no compuesta) del rendimiento que
representaría esa prima sobre el precio de la acción, para poder comparar
vencimientos de distinta duración en la misma escala.

---

## 9. Resultado total por escenario — sección 3 (con tu prima real)

Estos cálculos usan tu **precio medio de entrada** (`precio_entrada`, lo que
pagaste de verdad por las acciones), no el precio de mercado actual — porque es
lo que determina tu resultado real de cartera.

Variables:
- `contratos = número_de_acciones // 100` (1 contrato = 100 acciones; el
  restante que no llega a 100 no cubre un contrato completo)
- `acciones_cubiertas = contratos × 100`
- `comision_apertura`, `comision_asignacion`: editables en la GUI (por defecto
  0,75 €/contrato y 1,00 €/contrato — tarifas MEFF/DEGIRO vigentes en 2026)

**Escenario A — la call vence sin valor** (el strike no se alcanza, conservas
las acciones):

```
Resultado_A = prima_real × 100 × contratos − comisión_apertura × contratos

Rend_%_A = Resultado_A / (precio_entrada × acciones_cubiertas) × 100
```

**Escenario B — te asignan** (vendes las acciones al strike):

```
Plusvalía_acción = (strike − precio_entrada) × acciones_cubiertas

Resultado_B = Plusvalía_acción
            + prima_real × 100 × contratos
            − (comisión_apertura + comisión_asignación) × contratos

Rend_%_B = Resultado_B / (precio_entrada × acciones_cubiertas) × 100
```

El denominador (`precio_entrada × acciones_cubiertas`) es el capital que tienes
realmente invertido en esas acciones — el % siempre se mide contra tu coste real,
no contra el valor de mercado.

---

## 10. Rendimiento total si asignado — versión teórica (matriz, sección 2)

Misma fórmula que el Escenario B del punto 9, pero:
- Usa el **precio teórico BS** en vez de una prima real.
- Se calcula **siempre para 1 contrato exacto (100 acciones)**, no para tu
  posición real completa.

```
comisión_1c = comisión_apertura + comisión_asignación   # una sola vez, no × contratos

Resultado_teórico = (strike − precio_entrada) × 100 + Precio_teórico × 100 − comisión_1c

Rend_%_teórico = Resultado_teórico / (precio_entrada × 100) × 100
```

**Por qué solo 1 contrato:** el resultado y el capital en juego escalan de forma
proporcional con el número de contratos, pero la comisión es un coste fijo por
contrato — así que el `%` resultante es *casi* independiente del tamaño de la
posición (la comisión se diluye más cuanto mayor sea la posición). Se usa 1
contrato como referencia estándar para poder explorar rápido toda la rejilla de
strikes/vencimientos; para la cifra exacta sobre tu posición real, la tabla de
comparación de la sección 3 (que sí usa tu número de acciones) es la que manda.

---

## 11. Tasa libre de riesgo y comisiones — valores por defecto

Estos son **valores de referencia editables**, no constantes fijas en el código:

| Parámetro | Valor por defecto | Fuente / fecha |
|---|---|---|
| Tasa libre de riesgo | 2,25% | Facilidad de depósito del BCE, vigente desde 17-jun-2026 |
| Comisión apertura | 0,75 €/contrato | Tarifa MEFF de DEGIRO, vigente en 2026 |
| Comisión asignación | 1,00 €/contrato | Tarifa de ejercicio/asignación de DEGIRO |

Revísalos periódicamente — no hay ninguna llamada automática que los actualice.

---

## Limitaciones conocidas (a propósito, no son bugs)

- **Strikes sintéticos** (punto 6): no son la cadena real de opciones de tu
  bróker, es una rejilla matemática en pasos de 0,25 alrededor del precio
  actual.
- **IV implícita informativa, no predictiva**: como se explica en el punto 4,
  la comparación IV vs. histórica contiene la misma información que comparar
  precio real vs. teórico — no es una señal de compra/venta nueva ni elimina el
  sesgo estructural de que el mercado casi siempre cobra algo por encima de la
  volatilidad histórica (prima de riesgo de volatilidad).
- **Sin persistencia entre sesiones**: al cerrar el navegador se pierde la
  comparación de operaciones evaluadas (descartado deliberadamente).
- **Dividendo continuo**: el modelo asume dividendo continuo (`q`), una
  aproximación razonable pero no exacta para una acción que paga dividendos
  discretos (ej. semestrales), especialmente relevante cerca de fechas
  ex-dividendo por riesgo de asignación anticipada.

---

## Cómo verificar los números a mano

Ejemplo con los valores usados durante el desarrollo (Iberdrola, aprox.):

```
S = 195.50, K = 205, T = 30/365, r = 2.25%, q = 3.29%, σ_hist = 28%
precio_entrada = 180.00, prima_real = 2.20, contratos = 1
comisión_apertura = 0.75, comisión_asignación = 1.00
```

- `d1 = [ln(195.50/205) + (0.0225 − 0.0329 + 0.28²/2)×(30/365)] / (0.28×√(30/365))`
- Teórico BS ≈ **1.85 €** (verificado numéricamente durante el desarrollo)
- Si vence sin valor: `2.20×100×1 − 0.75×1 = 219.25 €` → `219.25 / 18000 × 100 = 1.22%`
- Si asignan: `(205−180)×100 + 220 − 1.75 = 2718.25 €` → `2718.25 / 18000 × 100 = 15.10%`

Estos dos últimos resultados están verificados por ejecución directa en Python
durante el desarrollo de la app.
