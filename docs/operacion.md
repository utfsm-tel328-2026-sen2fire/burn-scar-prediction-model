# Operación — qué necesita el modelo en inferencia

> **En una frase:** qué datos hay que darle al modelo ya entrenado, de dónde salen y con
> cuánto retraso llegan.
>
> Qué datos hay que darle al modelo ya entrenado, de dónde salen y qué antigüedad tienen.
> Es la sección que convierte el proyecto en producto. El catálogo de fuentes con sus
> licencias está en [`datos.md`](datos.md).

## 1. Las entradas en operación y su edad

**El modelo actual consume solo las entradas marcadas con ✅.** Las demás están previstas
para la v2 (BL5) y hoy no llegan al encoder: son catálogo de fuentes disponibles, no
entradas reales.

| Entrada | Fuente | En el modelo | Latencia (adquisición → disponible) | Revisita sobre el AOI | **Edad del dato más reciente** |
|---|---|:--:|---|---|---|
| Imagen multiespectral | Sentinel-3 OLCI **NRT** L1 | ✅ | **< 3 h** | ~2 d (~3 d con nubes) | **0-2 días (media ~1 día)** |
| Radar | Sentinel-1 GRD NRT | v2 | ~1 h | ~6 d | 0-6 días (media ~3 días) |
| Línea base pre-fuego | Sentinel-2 SR (compuesto) | v2 | ~3 h | 5 d | días o semanas antes del fuego |
| Meteo observada | GFAS | v2 | ~12 h | 6-12 h | ~12 h |
| Meteo observada | ERA5-Land-**T** | v2 | 5 d | horario | ~5 d |
| **Pronóstico** | GFS / ECMWF HRES | v2 | ~6-12 h | horario | a futuro |
| Fuego activo | FIRMS / VIIRS | v2 | ~1-3 h | 12 h | ~12 h |
| Topografía | Copernicus DEM | ✅ | estático | — | — |
| Combustible | ESA WorldCover | ✅ | estático | anual | — |
| **Cicatriz conocida** | **TMB-S3 congelado** (no CEMS) | ✅ | ver §3-4 | — | ver §3 |

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
   Sentinel-3  (OLCI, 300 m, ~2 dias) -------------------+
   DEM / WorldCover  (topografia, combustible) ---------+--> ENCODER
   horizonte h (dias) ---------------------------------+     ConvLSTM + SegFormer
                                                         |
   Sentinel-3  (OLCI, 300 m) -----------------------------+--> TMB-S3 (CONGELADO)
                                                            "donde esta quemado ahora"
                                                                  |
                                                                  v
   Sentinel-1  (GRD, radar, ~6 dias)   - v2, fuera del modelo actual (BL5)
   Sentinel-2  (linea base pre-incendio, 20 m) - v2
   ERA5 / GFS  (viento, temperatura, humedad) - v2
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
TMB-S3 medido sobre el propio test set es una cifra de referencia).

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
           Descarga: OLCI, DEM, WorldCover
           (meteorología y Sentinel-1 quedan para la v2)
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
| **El horizonte *h* es una entrada explícita** | Necesario para derivar el perímetro final "agregando horizontes" y para comparar limpiamente con el baseline BASE-1 (dataset.md §2, D10) |
| **Muestra por frame OLCI**, no por ronda CEMS | Reproduce la cadencia de operación y da más muestras (dataset.md §2) |
| **Preentrenamiento con MCD64A1 como supervisor, no como entrada** | MCD64A1 tiene ~6 semanas de latencia: sería imposible en operación, pero es un supervisor de milhões de etiquetas. Nunca entra en inferencia (dataset.md §8) |
| **El target se mide desde *f*, no desde la ronda anterior** | `polígono(r+1) − scar_TMB3(f)`: medir desde la ronda cuenta quemado ya visible en la imagen (dataset.md §2, D9) |
| **Ruido de corrupción sobre la cicatriz de entrada** | La cicatriz de TMB-S3 no es gold (F1 ~66) y su error está justo en el frente: hay que hacer al modelo robusto a ella (§4) |
| **Se excluyen las 50 series que TMB-S3 se sabe de memoria** | En ellas la cicatriz de entrada es el perímetro CEMS memorizado y el target cambia de significado entre train y test (§4, decisiones.md §3) |
| **Validación cruzada agrupada en vez de split fijo** | Con 46 series, un reparto fijo evalúa 9 y desperdicia 37 (dataset.md §5) |
| **Dos etapas, ambas evaluables (M-A y M-B)** | El ajuste CEMS es una inicialización: reportar los dos modelos convierte el conjunto pequeño en el diseño del experimento (dataset.md §8) |
| **La evaluación es siempre contra CEMS** | CEMS lo dibujaron analistas con otras imágenes; con FireCCI S3.1.1 el modelo podría replicar su clasificador en vez de aprender física (investigacion.md §2, R21) |
| **La entrada de meteo (v2) necesita viento de pronóstico**, no solo análisis | ERA5-Land es análisis del pasado; para predecir el futuro hace falta GFS/ECMWF forecast. La literatura lo hace así (FireCast usa viento; WSTS meteo + forecast). **Queda fuera del alcance actual**: el modelo entrenado no consume meteo (decisiones.md §1.2, P2) |
| **FIRMS/VIIRS como canal adicional (v2)** | Es la señal más timely que existe sobre dónde está el fuego ahora mismo. Ruidosa y a 375 m, pero aporta el ancla en tiempo real |

## 8. Regenerar el dataset y lanzar runs largos

> **El dataset no se versiona en Git** (`data/` está en `.gitignore`). Cada workstation
> genera el suyo en local, y **el dataset debe regenerarse siempre después de tocar el
> generador de cicatriz o la construcción de la ventana temporal**.

### Orden obligatorio antes de entrenar

```bash
# 1. Regenerar el dataset (≈20 s con GPU; no requiere re-descargar nada)
uv run --extra rocm python scripts/build_pytorch_dataset.py

# 2. Fijar la referencia de comparación (baselines BASE-0/BASE-1 sobre el dataset nuevo)
uv run --extra rocm python scripts/evaluate_baselines.py

# 3. Entrenar (≈1 s por época y fold → ~1,5 min para 15 épocas × 5 folds)
nohup uv run --extra rocm python scripts/train_propagation.py \
      --epochs 15 --seed 42 --tag prop_v4_integrity > runs/prop_v4_integrity.log 2>&1 &
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

El entrenamiento completo son ~2 min con el dataset vigente (296 muestras): lánzalo con `nohup` (o `tmux`) y vigila el log.
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
- **Un target sin quemado se reporta como NaN**, no como F1 = 1,0. Si un informe de
  evaluación muestra un 100% sin más, suspecta de un dataset vacío antes de congratularte.
- La referencia a superar es **BASE-1 (dilatación radial)** del `evaluate_baselines.py` sobre
  el mismo dataset.

## 9. Qué etiqueta se está midiendo (diagnóstico del target)

El dataset guarda **dos etiquetas** por muestra:

| Etiqueta | Definición | Para qué |
|---|---|---|
| **De ensamblaje** (`target`) | `clip(CEMS(r+h) − cicatriz_TMB3(f), 0, 1)` | Es la que entrena el modelo de producción |
| **De referencia** (`target_cems`) | `clip(CEMS(r+h) − CEMS(r_f), 0, 1)` | El crecimiento **real** entre rondas CEMS. Es la verdad de propagación |

No son la misma cosa. La de ensamblaje resta una **fotografía** de TMB-S3, que en el frente
de fuego se queda corta: el resultado mezcla crecimiento nuevo con quemado que ya existía
y que el generador no detectó. Medido sobre las **296 muestras** del dataset vigente:

| | Etiqueta de ensamblaje | Etiqueta de referencia |
|---|---|---|
| Superficie mediana a umbral 0,5 | 55,5 px | **1,5 px** (37× menos) |
| IoU media entre ambas | **0,033** | — |

**Por eso el F1 de ensamblaje no puede interpretarse como propagación**: una etiqueta mucho
más grande es más fácil de acertar.

### El dataset vigente y por qué tiene 296 muestras

El dataset canónico se regeneró con identidad de adquisición única, validación
de capas estáticas e inventario de cobertura. El manifiesto contiene **411 muestras
candidatas únicas** tras colapsar tres productos CDSE duplicados.

| Motivo del descarte | Candidatas |
|---|---:|
| `EMSR785_AOI01`: WorldCover completamente vacío en una descarga limpia | 48 |
| Etiquetas CEMS rasterizadas no disponibles | 5 |
| La ronda objetivo no existe en la etiqueta local | 19 |
| Adquisición de `f` ausente o no utilizable | 43 |
| **Total** | **115** |
| **Muestras ensambladas** | **296** |

Todas las 296 entradas tienen `sample_id` y ruta de fichero únicos, repartidas en 40 series.
La serie excluida y el motivo quedan registrados en `inventory.excluded_series` del índice
(`EMSR785_AOI01`, `reason: invalid_static_layers`, `stage: static_layers`). Composición por
fold: 29 (f0), 15 (f1), 108 (f2), 96 (f3), 23 (f4) y **25 fuera de dominio** (fold -1, test).

El soporte de horizontes es explícito y no se filtra: 171 muestras en `(0,1]`, 99 en `(1,3]`,
25 en `(3,7]`, 1 en `(7,14]` y ninguna en `(14,+inf)`. La mediana es 0,903 días y el máximo
observado 7,024. **No hay horizonte máximo en el dataset ni en la evaluación**: `h` se
conserva continuo y las bandas son solo diagnóstico, de modo que ninguna banda filtra el
entrenamiento ni las métricas globales. **Ninguna muestra representa un arranque en frío**:
las 296 tienen cicatriz no vacía y el índice declara `cold_start_represented: false`.

### Protocolo de conteos y procedencia

Todo número que se publica distingue cuatro denominadores, que nunca se colapsan en uno:

| Denominador | Qué cuenta | Dónde vive |
|---|---|---|
| **Entradas** | filas del índice, una por muestra candidata aceptada | `inventory.total_entries` |
| **Muestras únicas** | entradas con `sample_id` distinto | `inventory.total_unique_samples` |
| **Ficheros únicos** | entradas con ruta de fichero distinta | `inventory.total_unique_files` |
| **Series** | valores distintos de `series_id` | `inventory.total_series` |

A esos se añaden los **muestras evaluables** de cada etiqueta, que son un subconjunto
distinto según la verdad contenga quemado o no. La evaluación informa del mismo descriptivo
para el modelo y para los baselines (`evaluated_set`), de modo que dos cifras solo se comparan
si comparten índice, umbral, etiquetas y subconjunto.

La procedencia es obligatoria en cada artefacto: el índice registra la versión del generador,
la huella SHA-256 del manifiesto de entrada y una huella de contenido estable
(`dataset_content_sha256`); `operational_threshold.json` registra la huella del índice, la
semilla, el modo de target, el esquema, el soporte de horizontes y la huella de cada
checkpoint. `scripts/verify_canonical_artifacts.py` contrasta esas sumas de verificación
contra `artefactos/canonicos.json`.

### Resultados canónicos

Conjunto de test (fold -1, fuera de dominio: 25 muestras únicas, 7 series), con modelo y
baselines sobre el **mismo índice, las mismas etiquetas y el mismo subconjunto evaluable**.
El umbral operativo congelado del run canónico es **0,20**; la comparación justa entre
modelos se hace además a un **umbral común de 0,10**, que es donde se comparan todos.

**Contra la etiqueta de ensamblaje** (`target`, incluye quemado no detectado por TMB-S3):

| Modelo | Umbral | F1 | Precisión | Evaluables |
|---|---:|---:|---:|---:|
| BASE-0 persistencia | 0,10 | 0,00% | 0,00% | 24/25 |
| BASE-1 dilatación radial | 0,10 | 15,94% | 19,11% | 24/25 |
| Trivial "todo quemado" | 0,10 | 16,29% | 9,41% | 24/25 |
| **M-A** (`mcd_pretrained`, cero-disparo) | 0,10 | 19,71% | 14,05% | 24/25 |
| **M-B** (`prop_m_b_finetuned`, BCE+Dice) | 0,10 | 30,98% | 29,89% | 24/25 |
| **M-B + Tversky** (`prop_m_b_tversky`) | 0,10 | 32,90% | 39,53% | 24/25 |
| **Modelo de producción** (`prop_v4_integrity`) | 0,10 | **36,46%** | 28,40% | 24/25 |
| **Modelo de producción** (`prop_v4_integrity`) | 0,20 | 39,34% | 33,12% | 24/25 |

**Contra la referencia** (crecimiento real entre rondas CEMS, que es propagación):

| Modelo | Umbral | F1 | Evaluables |
|---|---:|---:|---:|
| BASE-0 persistencia | 0,10 | 0,00% | 23/25 |
| BASE-1 dilatación radial | 0,10 | 7,30% | 23/25 |
| Trivial "todo quemado" | 0,10 | 8,42% | 23/25 |
| **M-A** (`mcd_pretrained`, cero-disparo) | 0,10 | 10,61% | 23/25 |
| **M-B** (`prop_m_b_finetuned`, BCE+Dice) | 0,10 | 11,80% | 23/25 |
| **M-B + Tversky** (`prop_m_b_tversky`) | 0,10 | 10,08% | 23/25 |
| **Modelo de producción** (`prop_v4_integrity`) | 0,10 | **14,68%** | 23/25 |
| **Modelo de producción** (`prop_v4_integrity`) | 0,20 | 10,59% | 23/25 |

> **Qué es canónico y qué no.** Las cifras anteriores son las del run publicado en
> `artefactos/canonicos.json` (`checkpoints/antonio_prop_v4_integrity_20261117_165923`,
> semilla 42). El detalle por run —IoU, recall y el **soporte por banda de horizonte**—
> queda en el JSON de cada run (`runs/<run>/`), que no se versiona; las bandas son
> diagnóstico y ninguna filtra las métricas globales.
>
> **Reproducibilidad entre máquinas.** El protocolo y el **umbral operativo se reproducen**
> (una reentrenada con la misma semilla da el mismo 0,20), pero **los decimales no**: la
> cicatriz de TMB-S3 y el entrenamiento se calculan en GPU y dependen del hardware. En esta
> máquina (AMD ROCm) la reentrenada da 41,93% / 7,92% a 0,20, frente a 39,34% / 10,59% del
> run publicado (NVIDIA). Por eso el `dataset_content_sha256` —que incorpora la cobertura de
> la cicatriz— **tampoco es estable entre máquinas**: es estable entre regeneraciones en la
> misma, que es lo que garantiza.

> **Dónde están los números, para que se puedan comprobar.** Los resultados publicables van
> versionados en [`artefactos/resultados/`](../artefactos/resultados/README.md): los
> **baselines** sobre el dataset canónico (que reproducen exactamente las filas de esta
> sección), las **auditorías de integridad** de los dos índices y una **reproducción local**
> del run con su umbral, sus huellas y su barrido de umbrales. Lo que **no** está
> versionado es el JSON de métricas del run publicado (vive en la máquina de origen); de él
> quedan las sumas de verificación en `artefactos/canonicos.json`.

La CV diagnóstica `prop_cems_diag_v4_integrity` usa su propio dataset, su propia semilla y su
propio umbral congelado (0,10, con dispersión 0,30) y obtiene 9,43% contra ambas etiquetas
sobre 23/25 muestras; al cambiar etiqueta, umbral y subconjunto, **no es comparable
directamente** con las tablas anteriores.

Cómo leerlo:

- **El número publicable de propagación es 14,68%** (umbral común 0,10), no 36,46%. El
  segundo mide desempeño contra una etiqueta que incluye quemado no detectado.
- Contra el crecimiento real, el modelo está por encima de BASE-1 (7,30%) y del trivial
  (8,42%), pero las diferencias siguen siendo modestas y el test contiene solo 7 series y
  25 muestras.
- **El umbral importa más que el modelo**: el mismo `prop_v4_integrity` pasa de 14,68% a
  10,59% de F1 contra la referencia solo por cambiar de 0,10 a 0,20. Publicar una cifra sin
  su umbral es publicar un número sin unidad.
- **El arranque en frío no está representado**: no se publica un desempeño para ese
  escenario y no se fabrica una muestra artificial.

Las cifras de las versiones anteriores del dataset están **sustituidas** por estas. La
comparación con ellas no es válida: el dataset anterior tenía la ventana temporal anclada a
una adquisición distinta de `f`, sus etiquetas no correspondían a la definición de quemado
nuevo y su índice contenía entradas duplicadas. El run `prop_v4` queda marcado como
**histórico y no publicable** (`checkpoints/khaen_prop_v4_20261115_210453/historico.json`) y
las herramientas avisan cuando se le pide evaluar con él.

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

# 3b. Auditar la integridad y el inventario del dataset canónico
uv run --extra rocm python scripts/audit_dataset_integrity.py \
      --index data/processed/dataset_index.json \
      --output runs/dataset_integrity_production.json

# 3c. Barrer umbrales sobre un run sin reentrenar (una sola pasada de inferencia
#     por checkpoint; compara Precision, Recall, F1, IoU y area predicha/real)
uv run --extra rocm python scripts/sweep_thresholds.py \
      --checkpoint-dir checkpoints/<run> --split test --thresholds 0.1 0.2

# 4. Dataset y entrenamiento con la etiqueta de referencia (diagnóstico)
uv run --extra rocm python scripts/build_pytorch_dataset.py --target-mode cems \
      --samples-out data/processed_target_cems/samples \
      --index-out data/processed_target_cems/dataset_index.json
uv run --extra rocm python scripts/train_propagation.py --epochs 15 --seed 42 \
      --index data/processed_target_cems/dataset_index.json --tag prop_cems_diag_v4_integrity
```

### Artefactos canónicos y sus sumas de verificación

`artefactos/canonicos.json` registra la huella de los dos índices de dataset, del umbral
operativo y de cada checkpoint del run canónico. Se comprueba con:

```bash
uv run --extra rocm python scripts/verify_canonical_artifacts.py
```

El script recalcula las sumas y las contrasta con el manifiesto; si algo ha cambiado, lo
informa y devuelve código de error. Para regenerar el manifiesto tras producir nuevos
artefactos: `--write`.

| Artefacto | Huella de contenido (SHA-256) |
|---|---|
| Índice de producción (`data/processed/dataset_index.json`) | `34d562db30f59dc74fc875a6cefd2a79c61c6f474091ebaad92a8c5901faa75c` |
| Índice de referencia (`data/processed_target_cems/dataset_index.json`) | `793121e1ac12ad6062c7c815639c402f1d93d2425a7d0e077beeea77420cc89c` |

La **huella de contenido** no depende de la cabecera temporal del índice: dos regeneraciones
del mismo dataset producen la misma huella aunque el fichero cambie en `generated_at`. La
**huella de fichero** sí cambia con cada regeneración, y por eso es la que permite localizar
el índice exacto con el que se entrenó un run concreto (`dataset_index_sha256` en
`operational_threshold.json`).

> ⚠️ **La huella de contenido es estable en la misma máquina, no entre máquinas.** El
> índice incorpora la cobertura de la cicatriz de TMB-S3, que se calcula en GPU: un píxel
> cuyo valor cae justo en el 0,5 puede caer a un lado o al otro según el hardware, y eso
> cambia el recuento y la huella. En una máquina con el mismo dataset pero otra GPU, la
> huella sale distinta aunque las 296 muestras, los folds y las bandas de horizonte sean
> idénticos. Para comparar dos índices, el criterio es el **inventario**, no el hash.

### Ojo con comparar etiquetas distintas

**Un F1 de una etiqueta no se compara con un F1 de otra.** Cada etiqueta tiene su escala
mínima (el trivial "todo quemado"), que es la referencia con la que hay que leerlo.
Por eso `evaluate_baselines.py` imprime siempre el trivial al lado.

**Y un F1 a un umbral no se compara con un F1 a otro umbral.** `evaluate_baselines.py`
puede evaluar a cualquier umbral, pero para comparar con el modelo hay que usar el
**umbral operativo congelado** (0,20 en el run canónico) o el **umbral común** con el que
se midieron todos los modelos (0,10). El umbral también determina qué
píxeles de la verdad cuentan como quemado y, por tanto, qué muestras son evaluables. La
opción `--operational-threshold-from <dir_de_checkpoints>` aplica a los baselines el mismo
umbral y el mismo subconjunto que el modelo. Las tablas de la sección anterior están
medidas **al umbral común 0,10 para todos los modelos**, con una fila adicional del
modelo de producción a su umbral operativo 0,20.

### Qué cuenta como evaluable, y qué significan las máscaras

Las dos máscaras del paquete de muestra (`mask` y `mask_cems`) usan la **misma codificación
canónica**: `1` = píxel evaluable, `255` = píxel ignorado (no observado en la ronda). Es la
convención que emite el rasterizador CEMS y la que espera la visualización, donde las zonas
`255` se atenúan en el panel.

Una **muestra es evaluable** cuando su métrica es informativa, es decir cuando la verdad
contiene quemado por encima del umbral de evaluación. Si no lo hay, el F1 es `NaN`: no es un
0, es "no se puede medir". Por eso los informes de evaluación informan de los tres números:

- **Series evaluables**: cuántas series han podido medirse (el promedio es por evento).
- **Muestras evaluables**: cuántas muestras del conjunto tienen métrica informativa.
- **Total**: cuántas muestras había en el conjunto evaluado.

Un subconjunto con pocas muestras evaluables está inflando su promedio por evento, así que
el recuento es parte del número, no un detalle del informe de evaluación. Cuando una muestra no tiene
ronda CEMS previa, su validez común es íntegramente `255` y su métrica contra la etiqueta de
referencia no es informativa: cuenta como no evaluable, no como un cero.

Medido sobre el **dataset canónico de 296 muestras** (validación: 29 + 15 + 108 + 96 + 23 =
271; test: 25). Una muestra es evaluable cuando la verdad contiene quemado por encima del
umbral, así que **el recuento depende de la etiqueta y del umbral, no del modelo**:

| Conjunto | Etiqueta | Series evaluables | Evaluables @0,10 | Evaluables @0,20 |
|---|---|---|---:|---:|
| Validación (folds 0-4, 271 muestras, 33 series) | De ensamblaje | 33/33 | 257/271 | 251/271 |
| Validación (folds 0-4, 271 muestras, 33 series) | De referencia | 33/33 | 239/271 | 224/271 |
| Test (fold -1, 25 muestras, 7 series) | De ensamblaje | 7/7 | 24/25 | 24/25 |
| Test (fold -1, 25 muestras, 7 series) | De referencia | 7/7 | 23/25 | 23/25 |

Estos recuentos se derivan **solo de la verdad y del umbral**, así que son reproducibles
con cualquier modelo: si cambian, lo que ha cambiado es el dataset o el umbral, no el
entrenamiento.

### Sobre la comparación con TMB-S3 upstream

TMB-S3 upstream evalúa **detección** de quemado: mira imágenes donde el fuego ya está y
etiqueta qué está quemado, con etiquetas derivadas de MCD64A1 y una regla "OR" que infla el
área un 34%. Su F1 de **65,5-68,2** (IoU 56,7-58,7, media de 3 semillas; solo los 4
checkpoints SegFormer publicados — datos.md §5.1) es de **su propio test set y sus propias
etiquetas**.

Este proyecto evalúa **pronóstico**: predecir qué se quemará en las próximas horas, con la
etiqueta del incremento CEMS. **Son dos preguntas distintas**, así que ese F1 no es una
meta ni una "brecha a cerrar": no se compara. La vara de este proyecto es **el trivial "todo
quemado"** y **BASE-1**, medidos al mismo umbral operativo que el modelo (16,29% y 15,94%
contra la etiqueta de ensamblaje; 8,42% y 7,30% contra la referencia), porque usan la misma
información disponible que el modelo.

Donde el número de upstream sí es útil es como referencia de la **calidad del generador de
cicatriz**, que es el techo de la entrada `x_scar` y, por tanto, el techo de la etiqueta de
ensamblaje.

### Comparativa M-A vs M-B (diagnóstico)

Todos los modelos se evalúan sobre el **mismo test canónico** (fold -1, 25 muestras, 7
series), con el **mismo índice**, las **mismas etiquetas** y el **mismo subconjunto
evaluable**; solo cambia el umbral, que se indica en la tabla.

| Modelo | Umbral | F1 ensamblaje | F1 referencia | Precisión (ensamblaje) |
|---|---:|---:|---:|---:|
| BASE-0 persistencia | 0,10 | 0,00% | 0,00% | 0,00% |
| BASE-1 dilatación radial | 0,10 | 15,94% | 7,30% | 19,11% |
| Trivial "todo quemado" | 0,10 | 16,29% | 8,42% | 9,41% |
| **M-A** (`mcd_pretrained`, cero-disparo sin CEMS) | 0,10 | **19,71%** | **10,61%** | 14,05% |
| **M-B** (`prop_m_b_finetuned`, BCE+Dice) | 0,10 | **30,98%** | **11,80%** | 29,89% |
| **M-B + Tversky** (`prop_m_b_tversky`, α=0,7, β=0,3) | 0,10 | **32,90%** | **10,08%** | **39,53%** |
| **Modelo de producción** (`prop_v4_integrity`) | 0,10 | **36,46%** | **14,68%** | 28,40% |
| **Modelo de producción** (`prop_v4_integrity`) | 0,20 | 39,34% | 10,59% | 33,12% |

Cómo leerlo:

- **M-A es una transferencia real**: sin haber visto CEMS ni una sola etiqueta de propagación,
  alcanza 19,71% contra la etiqueta de ensamblaje y 10,61% contra la referencia, por encima
  de BASE-1 (15,94% / 7,30%) y del trivial (16,29% / 8,42%) en las dos etiquetas.
- **El ajuste fino improves a M-A** en la etiqueta de ensamblaje (30,98% frente a 19,71%),
  pero **no** en la de referencia (11,80% frente a 10,61%): el salto grande del ajuste es
  sobre la etiqueta grande, que es la más fácil.
- **Tversky compra precisión y pierde recall**: sube la precisión al 39,53% y estabiliza el
  umbral entre folds (0,10–0,20), pero baja el F1 de referencia a 10,08%. Es el
  commitment de no-sobreestimar el frente, no de acertarlo más.
- **El modelo de producción sigue dominando** en la etiqueta de referencia (14,68%), que es
  la única que mide propagación real.
- **El umbral cambia el número y no el modelo**: a 0,20 el mismo `prop_v4_integrity` sube a
  39,34% contra la etiqueta de ensamblaje y baja a 10,59% contra la referencia. Por eso el
  umbral operativo se congela y se publica por separado, y por eso **un F1 de una etiqueta
  no se compara con un F1 de otra**.
- **Con 7 series y 25 muestras, ninguna de estas diferencias es estadísticamente
  concluyente.** Se publican como diagnóstico, no como clasificación de modelos.

## 10. Preentrenamiento con MCD64A1 (M-A → M-B)

`MODIS/061/MCD64A1` es el **supervisor** del preentrenamiento: da el día de quema por píxel
a 500 m, con ~6 semanas de retraso, y por eso no puede filtrar información que en operación
no tendrías (dataset.md §8). El flujo completo es:

```bash
# 1. Diagnóstico R19: desfase mediano (BurnDate - observed_time) sobre CEMS
uv run python scripts/audit_mcd64a1_lag.py --limit 12

# 2. Minería de eventos MCD64A1 mediterráneos (2018-2026, DBSCAN 150 km/30 d, excluyendo CEMS)
uv run python scripts/build_mcd64a1_manifest.py --max-events 500

# 3. Descarga incremental y reanudable de parches 64x64 (MCD64A1 + OLCI con nubes reales + DEM/WC)
uv run python scripts/download_gee_mcd64a1.py --manifest data/mcd_manifest.json

# 4. Ensamblar el dataset PyTorch de preentrenamiento (TMB-S3 congelado + máscara QA)
uv run --extra rocm python scripts/build_mcd64a1_dataset.py

# 5. Etapa 1 (M-A): preentrenar con MCD64A1 y evaluar transferencia cero-disparo sobre CEMS
uv run --extra rocm python scripts/pretrain_mcd64a1.py \
      --epochs 10 --output-checkpoint checkpoints/mcd_pretrained.pt

# 6. Etapa 2 (M-B): ajustar los 5 folds de CEMS inicializando desde el checkpoint M-A
uv run --extra rocm python scripts/train_propagation.py \
      --epochs 15 --init-checkpoint checkpoints/mcd_pretrained.pt \
      --tag prop_m_b_finetuned
```

| Paso | Qué garantiza | Dónde está escrito |
|---|---|---|
| 1 | El desfase de detección de MODIS es aceptable (< 1-2 días) o se corrige con un *shift* | `decisiones.md` §2 R19 |
| 2 | Ningún evento de MCD64A1 se solapa en espacio o tiempo con los 96 eventos de CEMS | `dataset.md` §7 D15 |
| 3 | Las nubes y el humo **reales** se conservan en OLCI; solo se filtra el nodata de swath y el nocturno | spec `mcd64a1-pretraining` |
| 4 | Las muestras del preentrenamiento tienen el **mismo esquema de entradas** que las de CEMS (OLCI, DEM, WorldCover, cicatriz y horizonte), con TMB-S3 congelado. No llevan etiqueta de referencia: su target sale de `BurnDate` | `dataset.md` §7 D12, D17 |
| 5 | M-A se puede evaluar en cero-disparo sobre CEMS sin haberlo visto nunca | `dataset.md` §8 |
| 6 | M-B inicializa cada fold desde el checkpoint M-A sin romper el entrenamiento desde cero | `dataset.md` §8 |

> **Dónde está el resultado.** La comparación de este pipeline está en §9
> ("Comparativa M-A vs M-B"). Con los runs actuales, M-A **ya supera a BASE-1 y al trivial**
> sin haber visto CEMS, y el ajuste M-B mejora a M-A sobre la etiqueta de ensamblaje pero
> no sobre la de referencia. El flujo está implementado y es reproducible.
