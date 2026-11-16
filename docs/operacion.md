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

### 6.1 Autenticarse en Earth Engine (para `scripts/download_gee_olci.py`)

El script necesita un proyecto de Google Cloud con la API de Earth Engine activada y
registrado para uso comercial o no comercial. Hay dos formas de autenticarse:

**A) Credenciales personales (lo habitual).** Desde una máquina con navegador:

```bash
earthengine authenticate
earthengine set_project MI_PROYECTO
```

Quedan guardadas en `~/.config/earthengine/credentials` y el script las usa tal cual.

**B) Cuenta de servicio (para máquinas sin navegador: servidores, contenedores, WSL).**
Es la opción que usa quien no tiene GUI. La cuenta de servicio necesita **dos roles** en el
proyecto, y su clave JSON **fuera del repositorio**:

| Rol en la consola | ID | Para qué |
|---|---|---|
| **Earth Engine Resource Admin** | `roles/earthengine.resourceAdmin` | poder usar Earth Engine: crear computations, leer assets |
| **Service Usage Consumer** | `roles/serviceusage.serviceUsageConsumer` | poder *usar* la API de EE en el proyecto (`serviceusage.services.use`) |

Los dos hacen falta, y **los dos son de la cuenta de servicio**, no del usuario: es la
cuenta la que llama a la API.

> Con solo el primer rol, `ee.Initialize()` funciona pero **consultar la colección falla**
> con `Permission 'earthengine.computations.create' denied` o
> `Caller does not have required permission to use project ...`. Son errores distintos para
> el mismo síntoma: el primero es que el proyecto no está registrado, el segundo es que
> falta el rol de Service Usage Consumer.

Los dos errores suenan a permiso pero significan cosas distintas, y es fácil confundirlos:

| Mensaje | Qué falta |
|---|---|
| `Permission 'earthengine.computations.create' denied ... (or it may not exist)` | el **proyecto no está registrado** para Earth Engine → paso de registro web |
| `Caller does not have required permission to use project ... Grant the caller the roles/serviceusage.serviceUsageConsumer role` | falta el **rol de Service Usage Consumer** |

Y un tercer matiz: tras conceder un rol, la IAM de Google **se propaga de forma gradual**, así
que durante unos minutos Earth Engine puede seguir dando errores de permiso que luego se
resuelven solos. `scripts/check_ee_access.py` los distingue y dice qué falta:

```bash
# .env  (ya está en .gitignore)
EARTHENGINE_PROJECT=MI_PROYECTO
EARTHENGINE_SERVICE_ACCOUNT=/home/usuario/.config/earthengine/service-account.json
```

Las variables del entorno real tienen prioridad sobre el `.env`. Si no hay ninguna de las
dos, el script se comporta exactamente como antes: `ee.Initialize()` a secas.

### 6.2 Comprobar el acceso antes de descargar

```bash
python scripts/check_ee_access.py
```

Comprueba, en este orden: si la API de Earth Engine está habilitada, si `ee.Initialize()`
funciona y si la consulta real de la colección OLCI responde (que es lo que exige
`serviceusage.services.use`, el permiso que más falla). Dice qué falta en cada caso y sale
con código 1 si algo no está listo.

> En NixOS, el wheel de `rasterio` (que usa el script para abrir los GeoTIFF) necesita
> `libexpat.so.1`, que no existe en el sistema. `devenv.nix` lo resuelve propagando
> `expat`; ver [`nixos.md`](nixos.md) §5.

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

# 3. Entrenar (≈1 s por época y fold → ~1,5 min para 15 épocas × 5 folds)
nohup uv run --extra rocm python scripts/train_propagation.py \
      --epochs 15 --tag prop_v4 > runs/prop_v4.log 2>&1 &
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

El entrenamiento completo son ~2 min con el dataset vigente (344 muestras): lánzalo con `nohup` (o `tmux`) y vigila el log.
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

## 9. Qué etiqueta se está midiendo (diagnóstico del target)

El dataset guarda **dos etiquetas** por muestra:

| Etiqueta | Definición | Para qué |
|---|---|---|
| **De ensamblaje** (`target`) | `clip(CEMS(r+h) − cicatriz_TMB3(f), 0, 1)` | Es la que entrena el modelo de producción |
| **De referencia** (`target_cems`) | `clip(CEMS(r+h) − CEMS(r_f), 0, 1)` | El crecimiento **real** entre rondas CEMS. Es la verdad de propagación |

No son la misma cosa. La de ensamblaje resta una **fotografía** de TMB-S3, que en el frente
de fuego se queda corta: el resultado mezcla crecimiento nuevo con quemado que ya existía
y que el generador no detectó. Medido sobre las **344 muestras** del dataset vigente:

| | Etiqueta de ensamblaje | Etiqueta de referencia |
|---|---|---|
| Superficie mediana a umbral 0,5 | 56 px | **5 px** (11,1× menos) |
| IoU entre ambas | **0,083** | — |

**Por eso el F1 se multiplica por 3,7**: una etiqueta 11 veces más grande es más fácil de
acertar.

### El dataset vigente y por qué tiene 344 muestras

El dataset se regeneró el 2026-11-15 al anclar la ventana temporal en el frame `f` (ver
`decisiones.md` §1.5, A2 y A3). Antes tenía 390 muestras y ahora tiene **344**: se descartan
**46**, y el motivo está medido, no supuesto.

| Motivo del descarte | Muestras |
|---|---|
| La adquisición de `f` no está en el inventario local | 24 |
| La adquisición de `f` existe pero no es utilizable (nocturna o con >20 % de píxeles sin dato) | 22 |
| **Total** | **46** |

Las 22 del segundo caso **antes se usaban con otra imagen**: el ensamblador tomaba la última
adquisición anterior a `f` y la ponía en su sitio, con lo que la cicatriz y la etiqueta
quedaban ancladas a un instante distinto del que define el horizonte. Descartarlas es lo que
exige el spec.

Las 24 primeras sí son recuperables: la consulta de imágenes descarga ahora por la **extensión
del AOI** y no por el punto del centroide (`decisiones.md` §1.5, A1). Con el inventario
completo, **412 de las 414 muestras del manifiesto** tienen su adquisición de `f` disponible;
las 2 que faltan no tienen adquisición de `f` en ninguna de las dos fuentes.

Composición por fold: 30 (f0), 16 (f1), 108 (f2), 95 (f3), 23 (f4) y **72 fuera de dominio**
(fold -1, el conjunto de test).

### El resultado del diagnóstico

Conjunto de test (fold -1, fuera de dominio), **umbral operativo congelado en 0,20** para
todo lo que se compara (el modelo y los baselines, con la misma agregación por evento y el
mismo subconjunto de muestras evaluables):

| Modelo | F1 contra la etiqueta de ensamblaje | F1 contra la **referencia** (propagación real) |
|---|---|---|
| B0 persistencia | 0,00% | 0,00% |
| B1 dilatación radial | 14,98% | 5,40% |
| Trivial "todo quemado" | 14,25% | 5,83% |
| **Modelo de producción** (`prop_v4`) | **33,73%** | **9,06%** |
| Modelo entrenado con la referencia (`prop_cems_diag_v4`) | 9,16% | 9,16% |

El número de **muestras evaluables** (las que tienen métrica informativa) depende de la
etiqueta:

| Etiqueta | Umbral 0,50 (baselines) | Umbral 0,20 (operativo) |
|---|---|---|
| De ensamblaje | 71/72 | 71/72 |
| De referencia | 48/72 | 49/72 |

Contra la etiqueta de referencia, que tiene 11× menos superficie, 23 de las 72 muestras del
test no tienen quemado en la verdad y por tanto **no son medibles**. En este dataset el
subconjunto evaluable casi no cambia entre el umbral 0,50 y el operativo, pero en el dataset
anterior sí cambiaba, y por eso los números de la tabla se miden todos al umbral operativo: a
0,50 los mismos baselines sacan 12,36% (ensamblaje) y 3,25% (referencia), sobre otro
subconjunto.

Cómo leerlo:

- **El número publicable de propagación es 9,06%**, no 33,73%. Este último mide desempeño
  contra una etiqueta que incluye quemado no detectado.
- **El modelo sí aprende, pero menos de lo que parecía**: contra el crecimiento real saca
  9,06%, **1,55× el trivial de esa misma etiqueta y ese mismo umbral** (5,83%). La vara
  real contra el crecimiento real es **~5,8%**: contra la verdad, B1 (5,40%) y no hacer nada
  (5,83%) siguen siendo prácticamente indistinguibles, y el modelo está por encima de los
  dos, pero no por un orden de magnitud.
- **Entrenar con la etiqueta limpia no mejora**: 9,16% frente a 9,06% contra la misma verdad,
  al mismo umbral (0,20) y sobre el mismo subconjunto (49/72 muestras). Empate, dentro del
  ruido. A su propio umbral operativo (0,10) ese modelo saca 11,27%, pero sobre 70/72
  muestras, o sea otro subconjunto y otro número, y por tanto no comparable. El target de
  referencia tiene 11× menos superficie y es más ruidoso en términos relativos, así que no es
  un juego de suma cero: el problema de propagación es genuinamente difícil.
- **Recall 81% con precisión 26%**: el modelo acierta *dónde* se va a quemar, pero
  sobreestima la anchura del frente. Es el siguiente frente de trabajo.
- **El umbral no es estable**: los óptimos por fold van de 0,10 a 0,40 y el entrenamiento
  avisa. El umbral congelado (mediana, 0,20) es poco fiable; los números de test hay que
  leerlos con esa reserva.

Las cifras de las versiones anteriores del dataset están **sustituidas** por estas. La
comparación con ellas no es válida: el dataset anterior tenía la ventana temporal anclada a
una adquisición distinta de `f`, y sus etiquetas no correspondían a la definición de quemado
nuevo.

### Cómo reproducir el diagnóstico

```bash
# 1. Regenerar el dataset (guarda ya las dos etiquetas)
uv run --extra rocm python scripts/build_pytorch_dataset.py

# 2. Ver los números contra las dos etiquetas y el trivial
#    (por defecto, umbral 0,50)
uv run --extra rocm python scripts/evaluate_baselines.py

# 2b. Los mismos baselines medidos al umbral operativo del run, que es como
#     se comparan con el modelo (mismo umbral, mismas muestras evaluables)
uv run --extra rocm python scripts/evaluate_baselines.py \
      --operational-threshold-from checkpoints/<run>

# 3. Reevaluar un run existente sin reentrenar
uv run --extra rocm python scripts/train_propagation.py --eval-only checkpoints/<run>

# 4. Dataset y entrenamiento con la etiqueta de referencia (diagnóstico)
uv run --extra rocm python scripts/build_pytorch_dataset.py --target-mode cems \
      --samples-out data/processed_target_cems/samples \
      --index-out data/processed_target_cems/dataset_index.json
uv run --extra rocm python scripts/train_propagation.py --epochs 15 \
      --index data/processed_target_cems/dataset_index.json --tag prop_cems_diag_v4
```

### Ojo con comparar etiquetas distintas

**Un F1 de una etiqueta no se compara con un F1 de otra.** Cada etiqueta tiene su escala
mínima (el trivial "todo quemado"), que es la referencia con la que hay que leerlo.
Por eso `evaluate_baselines.py` imprime siempre el trivial al lado.

**Y un F1 a un umbral no se compara con un F1 a otro umbral.** `evaluate_baselines.py`
evalúa por defecto a umbral 0,50 (donde las predicciones binarias de B0/B1 y el trivial son
naturales), mientras que el modelo se evalúa al **umbral operativo congelado** (0,10 en el run
actual). Los dos conjuntos de números se apoyan además en subconjuntos de muestras distintos:
contra la etiqueta de referencia hay 54/78 muestras evaluables a 0,50 y 76/78 a 0,10. Para
comparar modelo y baselines hay que ponerlos al mismo umbral, que es lo que hace
`--operational-threshold-from <dir_de_checkpoints>`: lee el umbral congelado del run y lo
aplica a los baselines, de modo que ambos se miden sobre la misma discretización de la
verdad y las mismas muestras evaluables. La tabla de la sección anterior está medida al
umbral operativo para todos.

### Qué cuenta como evaluable, y qué significan las máscaras

Las dos máscaras del paquete de muestra (`mask` y `mask_cems`) usan la **misma codificación
canónica**: `1` = píxel evaluable, `255` = píxel ignorado (no observado en la ronda). Es la
convención que emite el rasterizador CEMS y la que espera la visualización, donde las zonas
`255` se atenúan en el panel.

Una **muestra es evaluable** cuando su métrica es informativa, es decir cuando la verdad
contiene quemado por encima del umbral de evaluación. Si no lo hay, el F1 es `NaN`: no es un
0, es "no se puede medir". Por eso los informes informan de los tres números:

- **Series evaluables**: cuántas series han podido medirse (el promedio es por evento).
- **Muestras evaluables**: cuántas muestras del conjunto tienen métrica informativa.
- **Total**: cuántas muestras había en el conjunto evaluado.

Un subconjunto con pocas muestras evaluables está inflando su promedio por evento, así que
el recuento es parte del número, no un detalle del informe. Cuando una muestra no tiene
ronda CEMS previa, su validez común es íntegramente `255` y su métrica contra la etiqueta de
referencia no es informativa: cuenta como no evaluable, no como un cero.

Medido sobre los conjuntos del proyecto:

| Conjunto | Etiqueta | Series evaluables | Muestras evaluables (umbral 0,50) |
|---|---|---|---|
| Validación (folds 0-4, 312 muestras) | De ensamblaje | 33/33 | 280/312 |
| Validación (folds 0-4, 312 muestras) | De referencia | 27/33 | 205/312 |
| Test (fold -1, 78 muestras) | De ensamblaje | 8/8 | 78/78 |
| Test (fold -1, 78 muestras) | De referencia | 8/8 | 54/78 |

### Sobre la comparación con TMB-S3 upstream

TMB-S3 upstream evalúa **detección** de quemado: mira imágenes donde el fuego ya está y
etiqueta qué está quemado, con etiquetas derivadas de MCD64A1 y una regla "OR" que infla el
área un 34%. Su F1 de 69,45 / IoU 59,81 es de **su propio test set y sus propias etiquetas**.

Este proyecto evalúa **pronóstico**: predecir qué se quemará en las próximas horas, con la
etiqueta del incremento CEMS. **Son dos preguntas distintas**, así que el 69,45 no es una
meta ni una "brecha a cerrar": no se compara. La vara de este proyecto es **el trivial "todo
quemado"** y **B1**, medidos al mismo umbral operativo que el modelo (16,03% y 15,80%
contra la etiqueta de ensamblaje; 7,46% y 6,88% contra la referencia), porque usan la misma
información disponible que el modelo.

Donde el número de upstream sí es útil es como referencia de la **calidad del generador de
cicatriz**, que es el techo de la entrada `x_scar` y, por tanto, el techo de la etiqueta de
ensamblaje.
