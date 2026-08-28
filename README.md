# pobot — Señales de trading para opciones binarias (Pocket Option)

Bot de **señales** (no de ejecución automática) para opciones binarias: analiza velas
históricas y en vivo, hace backtesting y validación out-of-sample, y te dice **dirección**
(CALL/sube o PUT/baja), **momento** de entrada y **tiempo de expiración**. Tú ejecutas la
operación manualmente en Pocket Option; el bot nunca toca tu cuenta ni pide tus credenciales.

## Por qué así, y no de otra forma

Pocket Option no tiene API pública oficial. Automatizar la ejecución implicaría hacer
ingeniería inversa de su sesión o su WebSocket, lo que viola sus términos de servicio y puede
terminar en el bloqueo de la cuenta con el saldo dentro. Por eso este proyecto se detiene en la
señal: analiza, backtestea, y te avisa por consola y/o Telegram. La decisión y el clic son
tuyos.

## El principio que rige todo el sistema

En una opción binaria de payout `p` (ej. 0.85 = 85%), acertar paga `p × stake` y fallar cuesta
`stake`. Con probabilidad de acierto `w`, el valor esperado es:

```
EV = w·(1 + p) − 1
winrate de equilibrio = 1 / (1 + p)
```

Con payout 85% hay que acertar **más del 54.05%** de las veces solo para no perder dinero. Por
eso este bot no reporta "aciertos" sueltos: reporta el **límite inferior del intervalo de
confianza de Wilson al 95%** sobre el winrate observado en datos que la estrategia/modelo
**nunca vio durante su ajuste** (walk-forward). Si ese límite inferior no supera el winrate de
equilibrio, el reporte dice explícitamente que **no hay ventaja demostrada** — y hay que
creerle, no seguir insistiendo hasta que "salga positivo".

Esto es intencional: la mayoría de "bots de señales" que circulan por internet muestran
backtests in-sample sobreajustados que no se sostienen en vivo. Este proyecto prioriza no
mentirte a ti mismo sobre eficacia falsa.

## Instalación

No requiere dependencias de terceros — todo el núcleo usa la librería estándar de Python 3.11+.

```bash
git clone <este-repo>
cd Bot
cp .env.example .env   # rellena TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID si quieres alertas
```

`scikit-learn` es **opcional**: si está instalado, el comando `train` puede aprovechar un
modelo más potente (`HistGradientBoostingClassifier`); si no, usa una regresión logística
propia en Python puro, que siempre funciona.

```bash
pip install scikit-learn   # opcional
```

## Flujo de trabajo completo

```bash
# 1. Descargar histórico de velas (cripto, API pública de Binance, sin API key)
python3 -m pobot download --symbol BTCUSDT --interval 1m --n-candles 20000 --out data/btc_1m.csv

# 2. Backtest rápido de una estrategia de reglas
python3 -m pobot backtest --csv data/btc_1m.csv --strategy confluence --payout 0.85

# 3. Validación seria: walk-forward out-of-sample (la que realmente importa)
python3 -m pobot walkforward --csv data/btc_1m.csv --strategy confluence \
    --payout 0.85 --train-size 2000 --test-size 500

# 4. (Opcional) Optimizar parámetros — puntuado siempre out-of-sample
python3 -m pobot optimize --csv data/btc_1m.csv --strategy rsi_reversal \
    --grid grid.json --train-size 2000 --test-size 500

# 5. (Opcional) Entrenar el modelo ML sobre los mismos datos
python3 -m pobot train --csv data/btc_1m.csv --horizon 3 --payout 0.85 --out models/btc_1m.json

# 6. Señal puntual sobre la última vela cerrada del CSV
python3 -m pobot signal --csv data/btc_1m.csv --strategy confluence --symbol BTCUSDT

# 7. Señales en vivo (requiere red con salida a Binance)
python3 -m pobot live --symbol BTCUSDT --interval 1m --strategy confluence --telegram
```

`python3 -m pobot list-strategies` lista las estrategias registradas: `rsi_reversal`,
`ema_trend`, `bollinger_bounce`, `macd_momentum`, `pattern_sr`, `confluence` (combina las
anteriores por votación ponderada) y `ml` (modelo entrenado, requiere `--model`).

### Parámetros de una estrategia

Cada estrategia acepta parámetros por JSON vía `--params`, por ejemplo:

```bash
python3 -m pobot backtest --csv data/btc_1m.csv --strategy rsi_reversal \
    --params '{"period": 10, "oversold": 25, "overbought": 75}'
```

## Cómo leer el reporte

```
Operaciones: 812 (wins=430, losses=382, ties=0)
Winrate observado: 52.96%  (IC 95% Wilson: [49.51%, 56.38%])
Winrate de equilibrio (payout considerado): 54.05%
¿Ventaja estadísticamente demostrada?: NO
EV por operación: -0.0093   PnL total: -7.5530
Profit factor: 0.968
```

- **Winrate observado** es el punto estimado; el **IC 95% Wilson** es el rango donde
  probablemente está el winrate real. Si ese rango incluye o queda por debajo del winrate de
  equilibrio, **no hay ventaja demostrada**, aunque el punto estimado se vea "bien".
- El reporte además desglosa por **hora del día (UTC)** y por **expiración (nº de velas)** —
  ahí es donde se responde "en qué momento y con qué tiempo conviene entrar", si es que existe
  tal condición.
- `optimize` aplica una corrección tipo Bonferroni: cuantas más combinaciones de parámetros se
  prueban, más exigente es el umbral de significancia, porque probar muchas combinaciones y
  quedarse con la mejor es en sí mismo una forma de sobreajuste (data dredging).

## Configuración de Telegram

1. Crea un bot con [@BotFather](https://t.me/BotFather) y copia el token.
2. Escríbele al bot una vez y consulta `https://api.telegram.org/bot<TOKEN>/getUpdates` para
   obtener tu `chat_id` (o usa [@userinfobot](https://t.me/userinfobot)).
3. Pon ambos valores en tu `.env` local (nunca en el repositorio):

```
TELEGRAM_BOT_TOKEN=123456:ABC-tu-token
TELEGRAM_CHAT_ID=123456789
```

4. Usa `--telegram` en `signal` o `live`. Si el `.env` no tiene ambos valores, el bot sigue
   funcionando solo por consola, sin fallar.

## Seguridad y credenciales

- Este bot **nunca** pide ni almacena credenciales de Pocket Option (usuario, contraseña, SSID
  de sesión). No hace falta ninguna para generar señales.
- El único secreto que maneja es el token de Telegram, y vive exclusivamente en tu `.env`
  local (excluido por `.gitignore`).
- `BotConfig.save()` nunca serializa la sección de Telegram al JSON de configuración.

## Advertencias honestas

- **Las opciones binarias tienen EV negativo por diseño** para el trader promedio: el payout
  siempre es menor a 100%, así que hace falta una ventaja informacional real y sostenida para
  ganar a largo plazo. Ninguna estrategia de este repo garantiza esa ventaja; el bot mide si
  existe, no la fabrica.
- **Los pares OTC de fin de semana de Pocket Option son precios sintéticos generados por el
  propio bróker**, no cotizaciones de un mercado real. No existe histórico público fiable para
  backtestearlos, y un modelo entrenado con datos de Binance (mercado real) **no transfiere**
  a esos pares. Este proyecto no incluye soporte para backtestear OTC por esa razón.
- Un backtest o walk-forward positivo en el pasado no garantiza resultados futuros. Revalida
  periódicamente con datos nuevos.
- La martingala (`risk.martingale_enabled`) no mejora el EV esperado de una estrategia: solo
  cambia la forma del riesgo hacia "muchas ganancias pequeñas, ruina ocasional grande". Está
  desactivada por defecto y emite una advertencia explícita si se activa.

## Estructura del proyecto

```
pobot/
  types.py            Candle, Direction, Signal, Trade
  edge.py             Matemática de payout/EV/Wilson/Kelly
  config.py           Configuración + carga de .env
  data/               CandleSeries, generador sintético, descargador de Binance
  indicators/         SMA/EMA/RSI/MACD/ADX/Bollinger/patrones de vela (sin look-ahead)
  labeling.py         Etiquetado de dirección por expiración
  features.py         Vector de features por barra, sin look-ahead
  strategies/         Reglas técnicas + confluencia + wrapper de modelo ML
  ml/                 Regresión logística pura, calibración, backend sklearn opcional
  backtest/           Motor de simulación, métricas, walk-forward, optimizador
  risk.py             Gestión de capital (stake fijo/fracción/Kelly, martingala opcional)
  live/               Runner de señales en vivo + notificador (consola/Telegram)
  cli.py              Interfaz de línea de comandos
tests/                Suite de pruebas (unittest, sin dependencias externas)
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

El test más importante es `tests/test_null_edge.py`: verifica que, sobre ruido puro (una serie
sintética sin ninguna estructura predecible), el motor de backtest **nunca** reporte ventaja
estadística. Si ese test falla, hay un bug de fuga de información (look-ahead) o de
contabilidad — no una "buena estrategia" — porque el generador de ruido no tiene, por
construcción, ninguna señal explotable.

`tests/test_features.py` verifica lo contrario en la otra dirección: que ningún feature de la
barra `t` cambie si se altera cualquier vela posterior a `t`.

## Nota sobre el descargador de Binance

`pobot/data/binance.py` usa solo `urllib` (sin dependencias) contra la API pública de Binance.
En algunos entornos de red restringidos el host puede estar bloqueado; en ese caso, prueba con
`--base-url https://data-api.binance.vision`, o ejecuta la descarga desde una máquina con
salida a internet sin restricciones.
