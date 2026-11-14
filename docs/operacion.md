# Operación — qué necesita el modelo en inferencia

> **En una frase:** qué datos hay que darle al modelo ya entrenado, de dónde salen y con
> cuánto retraso llegan.
>
> Qué datos hay que darle al modelo ya entrenado, de dónde salen y qué antigüedad tienen.
> Es la sección que convierte el proyecto en producto. El catálogo de fuentes con sus
> licencias está en [`datos.md`](datos.md).

## 1. Las entradas en operación y su edad

| Entrada | Fuente | Latencia (adquisición → disponible) | Revisita sobre el AOI | **Edad del dato más reciente** |
|---|---|---|---|---|
| Imagen multiespectral | Sentinel-3 OLCI **NRT** L1 | **< 3 h** | ~2 d (~3 d con nubes) | **0-2 días (media ~1 día)** |
| Radar | Sentinel-1 GRD NRT (**v2**) | ~1 h | ~6 d | **0-6 días (media ~3 días)** |
| Línea base pre-fuego | Sentinel-2 SR (compuesto) | ~3 h | 5 d | días o semanas antes del fuego |
| Meteo observada | GFAS | ~12 h | 6-12 h | **~12 h** |
| Meteo observada | ERA5-Land-**T** | 5 d | horario | ~5 d |
| **Pronóstico** | GFS / ECMWF HRES | ~6-12 h | horario | **a futuro** |
| Fuego activo | FIRMS / VIIRS | ~1-3 h | 12 h | **~12 h** |
| Topografía | Copernicus DEM | estático | — | — |
| Combustible | ESA WorldCover | estático | anual | — |
| **Cicatriz conocida** | ⚠️ **CEMS no sirve en operación** | — | — | ver §3 |

**El cuello de botella es Sentinel-3.** Su latencia de publicación es de menos de 3 horas,
así que el problema no es el retraso de la información sino la **revisita**: el producto
resultante es **de cada ~2-3 días, no diario**. Para un producto diario habría que subir
a VIIRS (12 h, 375 m), lo que refuerza la v2 (decisiones.md §4).

**Matices verificados sobre la disponibilidad:**

- Los productos **NRT de OLCI se publican en menos de 3 h** desde la adquisición; los NTC
  en más de 24 h (en la práctica 48-72 h, porque incorporan la órbita precisa y datos
  meteorológicos de análisis en vez de pronóstico).
- **La ventana del archivo NRT es limitada** (del orden de un mes a un año según el nivel
  de producto). Para histórico hay que usar NTC.
- En **mayo 2026 un problema del segmento de tierra (ground segment)** retrasó los NRT de
  S3A/S3B fuera del plazo nominal. Es un aviso: una cadena de producción tiene que
  **monitorizar la latencia**, no asumirla.
- La revisita de OLCI con dos satélites es de ~2 días *ignorando nubes*; la efectiva sobre
  un AOI concreto es de ~3 días.
- **ERA5-Land consolidado va con 2-3 meses de retraso**; la versión de cerca de tiempo real
  es **ERA5-Land-T**, con 5 días de latencia. Para operación, la alternativa real es
  descargar la salida NWP en bruto (GFS/ECMWF) en lugar del producto ERA5.

## 2. El horizonte se cuenta desde la última observación

Esto cambia el significado de la predicción y hay que dejarlo explícito:

```
  Si el último OLCI disponible tiene 1,4 días de antigüedad:
      "predecir t+1d"  really significa  "t+2,4d desde ahora"
      "predecir t+3d"  really significa  "t+4,4d desde ahora"
```

La razón es la misma que en los productos CEMS: hay que usar `observed_time` (cuándo vio
el satélite el evento), no `event_time`. Si no, **se está acreditando tiempo en el que el
fuego ya estaba quemando**.

> **Consecuencia para el entrenamiento:** el horizonte de cada etiqueta debe medirse desde
> la última observación OLCI disponible en ese intervalo, no desde la ronda de CEMS. Si no,
> el modelo aprende a predecir un horizonte que nunca tendrá en operación.

> **La misma regla, aplicada al canal "cicatriz conocida"**, queda resuelta por
> construcción: la cicatriz la produce TMB-S3, que solo ve los frames hasta *f* y por tanto
> no puede contener quemado posterior (dataset.md §7 D1, dataset.md §8).

## 3. La cicatriz conocida: por qué CEMS no sirve en operación

En el entrenamiento, la entrada "cicatriz conocida" es la última delineación de CEMS. Pero
**CEMS no existe como servicio en operación**: hay que activarlo (lo pide una autoridad
autorizada) y tarda de días a semanas en producir productos.

Si se entrena con "cicatriz de CEMS" y en operación se usa "cicatriz de otra cosa", aparece
un **train/test shift** silencioso. Cuatro salidas:

| Opción | Cómo | Veredicto |
|---|---|---|
| (a) Autoregresivo | La predicción del modelo en *t*+1 se usa como entrada en *t*+2 | Funciona, pero **los errores se acumulan** sin ninguna corrección externa |
| (b) Cicatriz de MCD64A1 | Fecha de quema MODIS a 500 m | Simple, pero 500 m contra 300 m y ~6 semanas de latencia |
| (c) Cabeza de delineación propia | El mismo modelo predice también *"¿dónde está quemado ahora?"* | Correcta en principio, pero exige un entrenamiento extra |
| **(d) TMB-S3 congelado** | Un modelo **ya entrenado y publicado** para el mismo sensor, la misma resolución y las mismas etiquetas | **La elegida.** Es la opción (c) sin el coste de entrenar (c) |

## 4. El modelo: un generador de cicatriz externo + una cabeza de propagación

```
   Sentinel-3  (OLCI, 300 m, ~2 dias)
   Sentinel-1  (GRD, radar, ~6 dias) - v2, fuera de la v1 (BL5)
   Sentinel-2  (linea base pre-incendio, 20 m)
   ERA5 / GFS  (viento, temperatura, humedad)
   DEM / WorldCover  (topografia, combustible)
   horizonte h (dias)
         |
         |            +-------------------------------+
         |            |   TMB-S3  (CONGELADO)        |
         +----------->|   "donde esta quemado ahora" |
                      +-------------------------------+
                                    |
                                    v
         +--------------------------------------------------------------+
         |                        ENCODER                               |
         |                  ConvLSTM + SegFormer                        |
         +--------------------------------------------------------------+
                                    |
                                    v
                      +-------------------------------+
                      |  CABEZA: PROPAGACION         |
                      |  "donde ardara en h"          |
                      +-------------------------------+
```

Por qué TMB-S3 y no una cabeza de delineación propia:

1. **El ajuste de dominio es casi perfecto**: mismo sensor (OLCI), misma resolución (300 m),
   mismas 5 bandas (`Oa04, Oa06, Oa08, Oa17, Oa21`), misma geografía (Europa). El *dominio de
   las entradas* es el mismo. **El de las etiquetas no**: TMB-S3 etiqueta con la regla OR (que
   infla el área un 34 %, investigacion.md §2) y solo sobre el perímetro final, no sobre las rondas
   (datos.md §5.1). Ese es el gap real, y por eso hay que medir el F1 de TMB-S3
   contra CEMS sobre nuestras series (decisiones.md §5, P5).
2. **Es la opción (c) de §3 sin el coste de entrenar (c)**: el modelo ya existe, con
   código MIT y checkpoints publicados (F1 65,5-68,2 en su propio test).
3. **Resuelve el train/test shift por completo**: el generador es idéntico en
   entrenamiento y operación. Y como solo ve los frames hasta *f*, no puede filtrar
   quemado futuro.
4. **Permite centrarse en propagación**, que es el problema abierto.

**Lo que se pierde y lo que se gana.** Se pierde la cabeza de delineación propia, y con
ella el preentrenamiento con MCD64A1 como regularizador de esa cabeza. Se gana: un
proyecto mucho más acotado, y una cicatriz de calidad **medible y citable** (el F1 de
TMB-S3 medido sobre el propio test set es una figura del informe).

### El problema serio: TMB-S3 se sabe de memoria 50 de nuestras series

La auditoría (decisiones.md §3) encontró que **62 de las 96 series mediterráneas están en
TMB-S3**, y 50 de ellas en su `train`/`val`: sus pesos las aprendieron. En esas 50 series la
cicatriz de entrada **no es una estimación, es el perímetro CEMS memorizado**, así que el
target `polígono(r+1) − scar(f)` mide "crecimiento entre rondas" en lugar de "crecimiento
después de *f*". En las 46 restantes el target significa lo mismo en train y en test.

Entrenar con las dos a la vez sería darle al modelo **la misma pregunta con dos respuestas
distintas**: en 50 ejemplos el mapa perfecto del charco y en 46 un croquis, examinado con
croquis.

→ **Se excluyen las 50 series memorizadas del dataset (D13).** Quedan 46, y en ellas la
cicatriz es siempre una estimación: train y test hacen la misma pregunta. Las cifras, la
tabla y el corolario ("rehacer el split por año no lo arregla") están en
[`decisiones.md`](decisiones.md) §3.

**La cicatriz de TMB-S3 tiene un sesgo conocido**: sus etiquetas usan la regla OR, que
infla el área un 34%. Como el target se define respecto a `scar_TMB3(f)` (dataset.md §2, D9), ese
sesgo **se cancela parcialmente entre la entrada y la etiqueta**. Aun así, conviene
**cuantificarlo** sobre las propias series: el área de `scar_TMB3(f)` frente al área
fraccional de CEMS.

**Ruido de corrupción: sigue haciendo falta.** La cicatriz de TMB-S3 no es gold (F1 ~66), y
su error es *estructurado*: está en el frente, no en el interior. Durante el
entrenamiento se deforma la cicatriz de entrada: se borra un ~15% de los píxeles quemados al
azar y se dilata o erosiona el borde un píxel. Es una red de seguridad barata que hace al
modelo robusto a una cicatriz imperfecta, que es lo que verá en operación.

**Fine-tune de TMB-S3 (opcional, decisiones.md §5).** Si en algún momento se quiere mejorar la
cicatriz, se pueden afinar los pesos de TMB-S3 con las propias series y medir el delta. Es
un experimento barato y no toca el resto de la cadena de procesamiento.

> **Recomendación: usar TMB-S3 congelado y centrarse en la cabeza de propagación.** El
> proyecto gana foco y pierde la complicación de entrenar un segundo modelo. La dirección
> de mejora ya no es "delinear mejor", sino "predecir la propagación mejor".

## 5. El ciclo operativo

```
  T+0 h    Detección de hotspot (FIRMS / VIIRS / GOES)
           -> se abre un "evento"
  T+3 h    Llega OLCI NRT
           Descarga: OLCI, ERA5/GFAS, GFS forecast, DEM, WorldCover (+ S1 en la v2)
  T+3,5 h  TMB-S3 -> cicatriz actual (congelado)
           Modelo -> mapa de probabilidad de quemado a h (1, 3, 7 días)
  T+2-3 d  Llega el siguiente OLCI -> se repite
           ...
  T+n d    Dos o tres iteraciones sin crecimiento -> se cierra el evento
```

**La cadencia la marca el sensor más lento: OLCI (~2-3 días).**

## 6. Requisitos de infraestructura

| | |
|---|---|
| Copernicus Data Space | Cuenta gratuita obligatoria. OLCI NRT vía STAC |
| Google Earth Engine | Bien para prototipar, **mal para producción**: hay que automatizar con la API de CDSE |
| GFAS / GFS | Descarga directa desde NASA / NOAA, sin cuenta de Earthdata para GFAS |
| Almacenamiento | Nada: solo se descargan las ventanas del AOI, no los archivos completos |
| Monitorización | Hay que vigilar la latencia real de NRT (§1), no asumirla |

El detalle de cuentas por fuente está en [`datos.md`](datos.md) §6.1. CEMS, como cualquier
otra fuente de etiquetas, **nunca está en el camino de inferencia** (datos.md §6.4).

## 7. Qué cambia en el diseño por todo esto

| Cambio | Por qué |
|---|---|
| **Cicatriz de entrada generada por TMB-S3 (congelado)** | Quita CEMS del camino de inferencia sin tren/test shift: en las tres fases la cicatriz sale del mismo modelo (dataset.md §8, §4) |
| **Horizonte anclado en la última observación** | Si no, el modelo se entrena con horizontes que nunca tendrá en operación (§2) |
| **El horizonte *h* es una entrada explícita** | Necesario para derivar el perímetro final "agregando horizontes" y para comparar limpiamente con el baseline B2 (dataset.md §2, D10) |
| **Muestra por frame OLCI**, no por ronda CEMS | Reproduce la cadencia de operación y da más muestras (dataset.md §2) |
| **Preentrenamiento con MCD64A1 como supervisor, no como entrada** | MCD64A1 tiene ~6 semanas de latencia: sería imposible en operación, pero es un supervisor de milhões de etiquetas. Nunca entra en inferencia (dataset.md §8) |
| **El target se mide desde *f*, no desde la ronda anterior** | `polígono(r+1) − scar_TMB3(f)`: medir desde la ronda cuenta quemado ya visible en la imagen (dataset.md §2, D9) |
| **Ruido de corrupción sobre la cicatriz de entrada** | La cicatriz de TMB-S3 no es gold (F1 ~66) y su error está justo en el frente: hay que hacer al modelo robusto a ella (§4) |
| **Se excluyen las 50 series que TMB-S3 se sabe de memoria** | En ellas la cicatriz de entrada es el perímetro CEMS memorizado y el target cambia de significado entre train y test (§4, decisiones.md §3) |
| **Validación cruzada agrupada en vez de split fijo** | Con 46 series, un reparto fijo evalúa 9 y desperdicia 37 (dataset.md §5) |
| **Dos etapas, ambas evaluables (M-A y M-B)** | El ajuste CEMS es una inicialización: reportar los dos modelos convierte el conjunto pequeño en el diseño del experimento (dataset.md §8) |
| **La evaluación es siempre contra CEMS** | CEMS lo dibujaron analistas con otras imágenes; con FireCCI S3.1.1 el modelo podría replicar su clasificador en vez de aprender física (investigacion.md §2, R21) |
| **La entrada de meteo necesita viento de pronóstico**, no solo análisis | ERA5-Land es análisis del pasado; para predecir el futuro hace falta GFS/ECMWF forecast. La literatura lo hace así (FireCast usa viento; WSTS meteo + forecast) |
| **FIRMS/VIIRS como canal adicional** | Es la señal más timely que existe sobre dónde está el fuego ahora mismo. Ruidosa y a 375 m, pero aporta el ancla en tiempo real |

## 8. Regenerar el dataset y lanzar runs largos

> **El dataset no se versiona en Git** (`data/` está en `.gitignore`). Cada workstation
> genera el suyo en local, y **el dataset debe regenerarse siempre después de tocar el
> generador de cicatriz o la construcción de la ventana temporal**.

### Orden obligatorio antes de entrenar

```bash
# 1. Regenerar el dataset (≈20 s con GPU; no requiere re-descargar nada)
uv run --extra rocm python scripts/build_pytorch_dataset.py

# 2. Fijar la referencia de comparación (baselines B0/B1 sobre el dataset nuevo)
uv run --extra rocm python scripts/evaluate_baselines.py

# 3. Entrenar (≈2 s por época y fold → ~4 min para 15 épocas × 5 folds)
nohup uv run --extra rocm python scripts/train_propagation.py \
      --epochs 15 --tag prop_v2 > runs/prop_v2.log 2>&1 &
```

**El paso 1 no es opcional.** La cicatriz de entrada (`x_scar`) y el target están
horneados dentro de los `.npz` de cada muestra: cambiar el generador sin regenerar deja
el dataset viejo en disco y el entrenamiento sigue usando el anterior.

### En AMD (ROCm), el DataLoader es el problema, no la GPU

Medido en la RX 6800 XT (gfx1030, ROCm 6.4) con este dataset:

| DataLoader | train/época | validación |
|---|---|---|
| `num_workers=2` + `pin_memory=True` (lo que había) | 74 s | 12 s |
| `num_workers=2`, sin `pin_memory` | 82 s | 9 s |
| `num_workers=0` | **1,7 s** | **0,07 s** |

Los workers se bifurcan (`fork`) desde un proceso que ya tiene el contexto HIP
inicializado, y esa bifurcación cuesta decenas de segundos. Con `pin_memory` encima,
cada lote reserva memoria fijada con `hipHostMalloc`, que en ROCm es carísimo.

**La GPU nunca fue el problema**: el modelo son 780 k parámetros y son 32 ms por lote.
El script detecta el backend y usa `num_workers=0` en ROCm, `2` en CUDA. Se puede
forzar con `--num-workers N` y `--pin-memory` / `--no-pin-memory`.

Si algún día el dataset crece mucho (miles de muestras), el diagnosis cambia: ahí sí
compensa pagar la carga en paralelo, y conviene medir antes de asumir.

### Runs largos

El entrenamiento completo son ~4 min: lánzalo con `nohup` (o `tmux`) y vigila el log.
Con Ctrl+C se pierde el fold en curso y no queda traza de por qué murió el proceso.

### Cómo leer los números

- **F1 a umbral 0,5** y **F1 en el mejor umbral** se reportan por separado. Un modelo
  probabilístico desbalanceado rara vez opera en 0,5; el barrido de umbrales (0,1–0,7)
  da el punto de operación real y hace la comparación con los baselines binarios justa.
- **El umbral operativo se congela**: al terminar la CV se toma la mediana de los
  umbrales óptimos de cada fold de validación y se guarda en
  `checkpoints/<run>/operational_threshold.json` junto a los pesos. El conjunto de test
  (fold -1, fuera de dominio) se evalúa **después**, con ese umbral ya congelado: es el
  número publicable, porque el test no ha participado ni en la elección del checkpoint
  ni en la del umbral.
- **En inferencia, ese umbral es una constante**: `quemado = probs >= TAU_OPERACION`.
  Si los umbrales óptimos difieren en más de 0,1 entre folds, el entrenamiento avisa de
  que el modelo es inestable y el umbral poco fiable.
- **Para estimar área quemada no hace falta umbral**: el área esperada es la suma de las
  probabilidades. El umbral solo se aplica cuando hace falta una decisión binaria
  (activar una alerta, categorizar un mapa de riesgo).
- **Recall** importa tanto como el F1: un F1 bajo con recall ~0 significa que el modelo
  predice ausencia de quemado, no que no haya aprendido nada.
- **Un target sin quemado se reporta como NaN**, no como F1 = 1,0. Si un informe muestra
  un 100% sin más, suspecta de un dataset vacío antes de congratularte.
- La referencia a superar es **B1 (dilatación radial)** del `evaluate_baselines.py` sobre
  el mismo dataset.
