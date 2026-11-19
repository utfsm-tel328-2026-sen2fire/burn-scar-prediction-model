# Investigación — estado del arte, por qué 300 m y cómo verificar los números

> **En una frase:** por qué se eligió 300 m, qué dicen los benchmarks comparables y cómo se
> pueden volver a comprobar los números de este repositorio.
>
> Material de referencia. Aquí no se decide nada: se justifica. Lo decidido
> está en [`decisiones.md`](decisiones.md) y [`dataset.md`](dataset.md).

## 1. Resumen ejecutivo

Un incendio forestal deja una cicatriz que deja de crecer en cuanto el fuego se apaga.
Lo que crece mientras el fuego está activo es el **frente de propagación**. Por eso el
proyecto no es "delinear la cicatriz" (segmentación, un problema resuelto) sino
**"dónde va a arder el frente en las próximas horas"** (predicción, un problema abierto).

El hallazgo que condiciona todo el diseño: **una imagen satelital no predice propagación**.
El fuego avanza por viento, humedad, temperatura, combustible y topografía; una imagen
solo dice *dónde ya ardió*. Todo benchmark serio del campo (WSTS, TS-SatFire, FireCast,
NextDayWildfireSpread) mete meteorología, topografía y combustible como entradas. Un
modelo puramente imagen→imagen aprende el prior "el fuego sigue donde iba", que funciona
mientras el viento no cambie y falla justo en los casos que importan.

Por eso el proyecto se apoya en **datos abiertos de imagen y de terreno** (Sentinel-3 OLCI,
Copernicus DEM, WorldCover): nada de esto tiene coste ni licencia comercial. **Hoy el
modelo entrenado solo consume imagen y terreno** — la meteorología (ERA5/GFAS, GFS) queda
**fuera del alcance** (decisiones.md §1.2, P2), y la razón de que esté documentada aquí es
precisamente la que da este apartado.

**El enfoque elegido:**

| | |
|---|---|
| Etiquetas | CEMS EMSR: delineaciones de quemado fechadas por ronda de monitorización |
| Entrada **actual** | Sentinel-3 OLCI (5 bandas, revisita ~2 días) + cicatriz de TMB-S3 + Copernicus DEM + ESA WorldCover + horizonte *h* |
| Entrada **prevista (v2)** | + Sentinel-1 (radar, ~6 días) + Sentinel-2 (línea base pre-incendio) + ERA5/GFS (meteo observada y prevista) + FIRMS/VIIRS (focos activos) |
| Salida | Probabilidad de quema por píxel a horizonte variable |
| Resolución | 300 m |
| Región | Mediterráneo / UE |
| Modelo | Encoder ConvLSTM + SegFormer + una cabeza de propagación. La cicatriz la genera **TMB-S3 congelado** (operacion.md §4) |
| En operación | Ciclo de ~2-3 días, marcado por la revisita de Sentinel-3 (operacion.md §1) |

**El problema conocido del dataset**: 96 series con señal de crecimiento es un número bajo
para generalizar (WSTS tiene 607, TS-SatFire 179, TMB-S3 246). Y la auditoría de splits
(decisiones.md §3) reduce el conjunto útil a **46 series limpias**, porque TMB-S3 se sabe de
memoria 50 de ellas (operacion.md §4). De esas 46, el manifiesto conserva 44 y el dataset
publicado **40 series** con 296 muestras. Es el principal frente abierto y la razón de
las dos palancas de ampliación (decisiones.md §4).

**Las tres decisiones que definen el proyecto**, y que se complementan:

1. **La cicatriz la pone TMB-S3, congelado.** Quita CEMS del camino de inferencia sin
   crear un desajuste entre entrenamiento y operación: la cicatriz siempre sale de un
   modelo (operacion.md §3-4).
2. **Se audita qué sabe TMB-S3.** Resultado: 62 de las 96 series están dentro de su
   dataset. Se excluyen las 50 que están en su `train`/`val`, y quedan 46 (decisiones.md §3).
3. **Con tan pocas series, el reparto fijo desperdicia el 80%.** Por eso se usa
   **validación cruzada agrupada**: cada serie se evalúa exactamente una vez
   (dataset.md §5).

---

## 2. Estado del arte

### 2.1 Tres tareas distintas que se confunden

```
  (A) DELINEAR            (B) PROPAGAR             (C) EXTENT FINAL
      "dónde quedó           "hacia dónde va          "hasta dónde
       quemado"               el frente"               llegará"
      NO es predicción      <- ESTE es el objetivo    (B) con h grande
```

- **(A) Delineación**: Clasificar píxeles como quemados/no quemados a partir de la imagen.
  Resuelto desde hace años con umbrales sobre NBR/dNBR, y desde 2016 con redes convolucionales.
  Es una **tarea de detección**, no de predicción. Dado que el enunciado inicial mezclaba
  "cicatriz" con "crecimiento", conviene no confundirlas: el modelo que buscamos es (B).

- **(B) Propagación**: Dado el estado actual del fuego, predecir dónde arderá después.
  Es un problema de dinámica: depende de física, no solo de teledetección.

- **(C) Extent final**: Predecir el perímetro total. Es (B) con horizonte largo, y tiene la
  complicación adicional de que hay que anticipar **cuándo se detiene** el fuego (la cola),
  que no es observable en el momento de la predicción.

**Formulación elegida: (B) multi-horizonte.** Es la que da la señal de entrenamiento más
densa (las rondas de CEMS son etiquetas intermedias, no solo una etiqueta final) y el
perímetro final se deriva agregando horizontes.

### 2.2 La física: por qué la imagen sola no predice

El frente de un incendio se propaga en la dirección del viento, influenciado por:

- **Viento** (velocidad y dirección): el factor dominante en la propagación de superficie.
- **Humedad y temperatura**: determinan la inflamabilidad del combustible.
- **Combustible**: tipo, carga y continuidad (más importante en sotobosque y matorral).
- **Topografía**: pendiente (la pendiente acelera en ascenso) y orientación (exposición solar).
- **Vulnerabilidad**: caminos, líneas de fuego, zonas con continuidad de combustible.

De los cinco, **solo la topografía y el combustible se pueden inferir de una imagen**. El
viento, la humedad y la temperatura **no se ven en ningún satélite**: son datos
meteorológicos. Por eso cualquier modelo que se pretenda predecir propagación los necesita.

### 2.3 Benchmarks y datasets públicos

| Dataset | Tarea | Res. | Eventos | Periodo | Qué aporta |
|---|---|---|---|---|---|
| **WildfireSpreadTS** (WSTS) | Propagación día+1 | 375 m | 607 | US, 2018-2021 | Benchmark de referencia; 23 canales multimodales. **El más comparable a este proyecto** |
| **WSTS+** | Propagación | 375 m | +4 años | US | Extensión de WSTS con más años y geografía |
| **TS-SatFire** | AF + BA + progresión | 375 m | 179 | CONUS, 2017-2021 | VIIRS día+noche; QA manual de etiquetas. 71 GB, CC-BY |
| **Mesogeos** | Peligro + extent final | 1 km | ~9.500 muestras | Mediterráneo, 2006-2022 | Cubo diario armonizado con 30 variables de drivers. **NeurIPS 2023 D&B** |
| **WildFireSpread** | Extent final | 1 km | ~9.500 | Mediterráneo, 2006-2022 | El dataset + modelos de Mesogeos, listos para entrenar |
| **TMB-S3** | Delineación con modelo temporal | 300 m | 246 activaciones CEMS | Europa, 2016-2025 | 246 activaciones CEMS con Sentinel-3 OLCI; **código MIT reutilizable** |
| **Mediterranean DL** | Extent final | var. | Mediterráneo | 2006-2022 | 10 timesteps (−4d a +5d post-ignición), 27 variables, dataset y modelos liberados |
| **FireCCI S311** | Quemado + fecha | 300 m | global | 2019-2024 (ampliado en 2025-10) | Sentinel-3 Synergy + VIIRS, binario. **Solo como ablación**: MCD64A1 es el supervisor principal (más cobertura, fecha diaria, 2000-2026) |
| **FireCCI51** | Quemado + fecha | 250 m | global | 2001-2022 | MODIS. 57 GB en CEDA, también en GEE |
| **MCD64A1** | Quemado + fecha | 500 m | global | 2000-2026 | **Día de quema por píxel**. Libre, reproducible, enorme |
| **MTBS** | Severidad + perímetro | 30 m | ~EE.UU. | 1984-2024 | Solo EE.UU. y snapshot anual; no crecimiento intra-incendio |

### 2.4 Hallazgos de la literatura que condicionan el diseño

1. **El frame pre-incendio es imprescindible.** TMB-S3 (arXiv 2609.34596) encuentra que el
   modelado temporal (ConvLSTM) solo mejora cuando hay una adquisición pre-fire en la
   secuencia; con frames post-fire solos el gain es despreciable. *Los frames post-fire
   comparten la misma firma de quemado, así que no hay señal de cambio que propagar.*
   → **La cadena de procesamiento debe siempre incluir una línea base pre-incendio.**

2. **El etiquetado binario en el borde infla el área.** TMB-S3 reetiqueta a 300 m con la
   regla "quemado si **cualquier** píxel de 10 m dentro lo está", y reporta una inflación
   del **34%** del área quemada (×3 en incendios pequeños). Para una tarea de *crecimiento*
   esto es peor: el error se concentra justo en la banda del frente, que es lo que se
   intenta predecir.
   → **Etiquetado cuantificado** (fracción quemada por píxel) en vez de OR lógico.

3. **Cinco bandas bastan.** TMB-S3 demuestra que un subconjunto de 5 bandas OLCI iguala
   al conjunto completo de 21. → **Reduce el volumen del dataset ×4.**

4. **Separación espacial obligatoria.** Mesogeos y WildFireSpread usan un **split temporal
   por años** (2006-2020 / 2021 / 2022 en ambos), elegido explícitamente para no filtrar
   incendios cercanos en el tiempo.
   Eso mide generalización **en el tiempo**, no **en el espacio**: un incendio de test puede
   caer en la misma comarca que otros de train. TMB-S3 estratifica por cluster geográfico
   (DBSCAN, ε=150 km), aunque sus propios autores avisan de que eventos cercanos de un
   mismo cluster pueden acabar en splits distintos.
   → **Split por cluster geográfico, nunca aleatorio.**

5. **Las imágenes comerciales no están disponibles para el usuario final.** En CEMS, el
   80% de las delineaciones mediterráneas 2023-2026 (390 de 489) se hizo sobre sensores
   comerciales (Legion, SPOT, WorldView, Pleiades, GeoEye, PlanetScope); solo el 18%
   (87) salió de Sentinel-2, y un 2% no declara imagen. Ver decisiones.md §2 (riesgo R2).

6. **Reutilizar un modelo pretrained implica auditar sus splits.** Si TMB-S3 (que entrenó
   con CEMS) se usa como generador de cicatriz, hay que comprobar qué geografía ya vio:
   su `manifest.parquet` permite hacerlo. Sin esa auditoría, el modelo de propagación
   hereda una fuga espacial que no es visible (R18, decisiones.md §3).
   → **Auditoría de splits antes de construir el dataset.** *(Ejecutada: 62 de las 96
   series están en TMB-S3. Ver decisiones.md §3.)*

### 2.5 Arquitecturas de referencia

- **ConvLSTM + SegFormer** (TMB-S3): backbone espacial SegFormer, encoder temporal ConvLSTM.
  Es la opción por defecto: probada, con código MIT, y con checkpoints publicados.
- **UNet3D**: convoluciones 3D sobre la pila temporal. Más simple, peor rendimiento.
- **Vision Transformer** (WildFireSpread):Patch embedding sobre la secuencia temporal.
- **ConvNeXt-UPerNet**: backbone más potente, a cambio de más parámetros.

### 2.6 Qué etiquetas sirven para medir, y cuáles no

Este es un criterio que no suele estar explícito y que decide el diseño entero.

**La pregunta que hay que hacerse de cada fuente de etiquetas:** *¿puedo deducir la
respuesta mirando las imágenes que le doy al modelo?*

- Si **sí**, el modelo puede aprender a **replicar el algoritmo que generó la etiqueta**
  en vez de aprender la física de la propagación. Dará buenos números y no significarán
  nada.
- Si **no**, el modelo tiene que aprender de verdad, y el número mide algo.

| Fuente | ¿Quién la generó? | ¿Se puede deducir desde las entradas OLCI? | Uso |
|---|---|---|---|
| **CEMS** | **Analistas**, sobre imagen comercial (80%) y Sentinel-2 (18%) | **No** — miraron otras imágenes | ✅ **La única válida para evaluar** |
| **MCD64A1** | Algoritmo sobre reflectancia **MODIS** | Parcialmente (NIR) | ✅ Preentrenamiento |
| **FireCCI51** | Algoritmo sobre reflectancia **MODIS** | Parcialmente (NIR) | ✅ Preentrenamiento (250 m, mejor que MCD64A1) |
| **FireCCI S3.1.1** | Algoritmo sobre reflectancia **OLCI + SLSTR** | **Casi directamente** — es el mismo sensor | ⚠️ Solo ablación o tercera opinión |

> **Por qué esto decide que CEMS se queda.** CEMS es la única fuente cuyo proceso no pasa
> por las imágenes del modelo *en el frente*: el Sentinel-2 con el que se trazó el 18% es solo
> línea base pre-incendio en este proyecto, así que no permite reconstruir el borde del
> incendio. Por eso es la que puede usarse de examen, y por eso las otras tres sirven de
> prácticas (R21, D16).

> **Y por qué FireCCI S3.1.1 no sustituye a CEMS.** Sobre el papel es ideal: 300 m, Sentinel-3,
> 2019-2024, con fecha de detección y nivel de confianza por píxel. Pero se generó mirando
> OLCI. Como **ejercicio** es buenísimo (mejor resolución que MCD64A1, con etiqueta blanda);
> como **examen**, mediría la capacidad de copiar un clasificador. Su papel: contrastar
> etiquetas de CEMS (el 80% viene de imagen comercial) y servir de ablación.

---

## 3. Por qué 300 m

### 3.1 La tensión

```
  REVISITA           vs          RESOLUCIÓN
  (datos libres)                 (datos libres)

  10 min   GOES-19/18    2 km      térmico; ignora el borde
  12 h     VIIRS         375 m     ritmo real, pero grueso
  ~2 d     Sentinel-3    300 m     <-- banda útil (300-375 m)
  1,4 d    HLS (L8/9+S2) 30 m      media efectiva global
  3-6 d    S1 / S2       10 m      detalle máximo, ritmo lento
  12 d     NISAR L-band  3-10 m    archivo desde ago-2025: insuficiente
```

**El techo duro:** con datos gratuitos, la frecuencia *sub-diaria* solo existe a 375 m
(VIIRS, 12 h) o más grueso. A 300 m la frecuencia es de ~2 días; a 10 m, de 3-6 días.
La **banda de resolución útil es, por tanto, 300-375 m**: dentro de ella la elección entre
VIIRS (12 h) y Sentinel-3 (300 m) no la marca la frecuencia sino la disponibilidad
(§3): VIIRS no está en GEE.

### 3.2 La revisita de la constelación no es el número de observaciones

Esto es lo que suele pasarse por alto. Un incendio mediterráneo típico dura 4 días:

| Sensor | Revisita | Observaciones en 4 días |
|---|---|---|
| Sentinel-2 | 5 d | **1** |
| Sentinel-1 | 6 d | **1** |
| Sentinel-3 | 2 d | **2-3** |
| VIIRS | 12 h | **8** |

A 10 m **no hay dinámica de crecimiento observable**. El modelo no puede aprender a
propagar con una sola imagen antes y otra después.

> **Regla operativa: la resolución y el horizonte se eligen juntos, no por separado.**

### 3.2bis Cuándo existen datos OLCI (esto acota el proyecto)

No es un detalle histórico: **Sentinel-3 es un sensor joven**, y eso fija la ventana temporal
de todo el proyecto.

| | Fecha | Consecuencia |
|---|---|---|
| S3A lanzado | 16 feb 2016 | |
| **Primeros datos OLCI de S3A** | **26 abr 2016** (ampliados a 4 abr 2016) | **El preentrenamiento no puede empezar antes de 2016** |
| OLCI L1 NRT oficial | 20 oct 2016 | |
| OLCI L1 NTC oficial | 14 dic 2016 | Lo que usaremos para el dataset histórico |
| S3B lanzado | 25 abr 2018 | |
| **Primeros datos OLCI de S3B** | **15 may 2018** | |

> **El límite físico es 2016; la decisión es 2018.** MCD64A1 tiene etiquetas desde 2000, pero
> **sin OLCI no hay entrada**: antes de abril de 2016 no hay muestra posible. Aun así, el
> pool de preentrenamiento **empieza en 2018** (decisiones.md §1.1 B13, decisiones.md §1.3 BL3), porque entre 2016 y
> 2018 solo hay S3A y al entrar S3B cambia el dominio de iluminación (R22). Esto reduce el
> pool a ~8 años y es la razón de que la v2 (o un dataset distinto) sea el sitio donde está
> el margen de ampliación.

> **Un cambio de dominio en mayo 2018.** Hasta entonces solo hay S3A (una geometría de
> observación); desde mayo 2018 hay S3A + S3B (dos geometrías, 140° de separación en fase).
> El preentrenamiento que abarque ambos periodos mezcla dos dominios de iluminación
> (R22). Es un argumento más para **empezar por 2018-2026** y no por 2016: menos dominio
> mezclado y menos descarga.

### 3.3 La evidencia empírica: el incremento típico por ronda

Medido sobre las 24 series con estadísticas completas de área quemada (§4):

| | Valor |
|---|---|
| Área final quemada | mediana **2.601 ha**, p75 12.022 ha, máx 33.013 ha |
| **Incremento por ronda** | min 2,9 ha · p25 62,6 ha · **mediana 491 ha** · p75 1.556 ha · máx 19.082 ha |
| En píxeles de 300 m | p25 **7 px** · **mediana 54,6 px** · máx 2.120 px |
| Incrementos < 1 px (9 ha) | 14% |
| Incrementos < 3 px (27 ha) | 27% |
| Incrementos ≤ 0 (fuego parado) | 8% |

**Lectura.** La distribución del incremento es bimodal:

- **Rondas con fuego activo**: incrementos de cientos a miles de hectáreas → 10-2.000
  píxeles a 300 m → **de sobra observables**. Aquí está el problema real y la señal es
  abundante.
- **Rondas con el fuego ya apagándose**: incrementos de ~3-60 ha → menos de un píxel a
  300 m → **casi no observable**, pero también casi trivial de predecir ("no crezcas").

Es decir: **300 m aguanta bien el problema**, porque el caso difícil tiene señal de sobra
y el caso que no la tiene no aporta dificultad. Ese 14% de rondas estancadas debe
**incluirse en el entrenamiento**, porque el modelo tiene que aprender a decir "para".

### 3.4 Alternativas descartadas

| Opción | Por qué no |
|---|---|
| **10 m (Sentinel-2, 5 d)** | 1-2 observaciones por incendio: no hay dinámica que aprender. Y el SWIR de S2 sí es útil, pero entra como línea base estática, no como serie temporal |
| **VIIRS 375 m / 12 h** | La mejor cadencia posible en abierto (8-16 observaciones por incendio). Descartado solo porque no está en GEE y obliga a procesar swaths L1B a mano. **Es la opción a recuperar en la v2** |
| **1 km (Mesogeos)** | Demasiado grueso para la dinámica de un frente. Sirve como fase 0 y como baseline, no como modelo final |
| **NisAR L-band** | Archivo desde agosto 2025: insuficiente para entrenar. Relevante a partir de 2028 |

### 3.5 Matiz: por qué no 30 m con HLS

HLS (Landsat 8/9 + Sentinel-2 armonizados) da ~1,4 días de revisita media global a 30 m,
lo que suena mejor que 300 m. Pero la media global no es la revisita de un AOI concreto:
la cadencia real por evento es de 2-3 días y muy desigual, con huecos por nube. A 300 m
con Sentinel-3 la cadencia es más regular y la geometría del dato es más simple. **La
decisión es revisable** si en la fase 1 se ve que la cadencia, y no la resolución, es el
cuello de botella.

> ⚠️ **HLS no es rápido.** La revisita de 1,4 días es una media global; el dato **tarda
> 2-3 días en estar disponible** (HLS v2.0), y la versión de baja latencia (HLS-LL, <6 h)
> no estará hasta 2027. Si la cadencia resultara ser el cuello de botella, HLS a 30 m no es
> la respuesta: es más frecuente pero más lento de recibir que un OLCI NRT.

---

## 4. Cómo verificar los números

Todos los datos del censo ([`dataset.md`](dataset.md) §3) y de los incrementos por ronda
(§3.3) se obtuvieron de la API pública de CEMS el 2026-11-06.
Re-verificado el **2026-11-08**: 268 activaciones en total (2023: 55, 2024: 69, 2025: 69,
2026: 75), 131 de incendio y 107 mediterráneas con la lista de 14 países de dataset.md §3 — sin
cambios. El método, para poder repetirse:

```python
import requests

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"

# 1) Listado de activaciones (paginado, sin autenticación)
activations = []
url = f"{API}/public-activations-info/?limit=500"
while url:
    data = requests.get(url).json()
    activations += data["results"]
    url = data.get("next")

# 2) Detalle de una activación (solo funciona desde EMSR656, 2023-03-23)
detail = requests.get(f"{API}/public-activations/?code=EMSR922").json()["results"][0]

# 3) Delineaciones de un AOI, con su sensor y su área quemada en hectáreas
for aoi in detail["aois"]:
    for product in aoi["products"]:
        if product["type"] != "DEL":
            continue
        monitor = product["monitoringNumber"]          # 0 = inicial, 1, 2, ... = rondas
        sensor = [i["sensorName"] for i in product["images"]]
        when = product["images"][0]["acquisitionTime"] if product["images"] else None
        area_ha = product["stats"]["Burnt area"]["None"]["affected"]
```

**Limitaciones del método:**

- La API **limita el ritmo** (HTTP 429). Hay que ir con concurrency bajo y backoff; una
  pasada completa de las 107 activaciones mediterráneas tarda varios minutos.
- El fetch de las 107 se completó al 100%, pero el contenido de `stats` solo está presente
  en algunos productos (24 series con área en todas sus rondas), lo que explica que el
  §3 se base en una submuestra de 24.
- El conteo de rondas por el HTML de la página legacy y por la API difieren en uno: la API
  llama `monitoringNumber` 0 al producto inicial, así que *rondas* = número de productos DEL
  y *transiciones* = rondas − 1.

Los detalles de cada fuente (endpoints, IDs en GEE, licencias) están en
[`datos.md`](datos.md).

### 4.1 Cómo verificar la auditoría de splits (decisiones.md §3)

El `manifest.parquet` de TMB-S3 se lee sin instalar nada, a través del servidor de
datasets de Hugging Face:

```python
import requests

BASE = "https://datasets-server.huggingface.co"
DS = "links-ads/tmb-s3"

# 1) Todas las filas del manifest (770 cajas, 246 activaciones)
rows, offset = [], 0
while True:
    r = requests.get(f"{BASE}/rows?dataset={DS}&config=manifest&split=train"
                     f"&offset={offset}&length=100").json()
    rows += [x["row"] for x in r["rows"]]
    if len(r["rows"]) < 100:
        break
    offset += 100

# Por cada fila: bbox_id ("EMSR168-1-0"), split, geometry_wkt (POLYGON en lon/lat)
tmb = {}
for row in rows:
    code = row["bbox_id"].split("-")[0]
    tmb.setdefault(code, {"split": row["split"], "boxes": []})["boxes"].append(row["geometry_wkt"])

# 2) Cruce con nuestras series: el código EMSR es la clave primaria
#    (96 series mediterráneas -> 62 con código presente en TMB-S3)
# 3) Distancia al centro del evento más cercano (haversine sobre el centroide)
```

**Limitaciones del método:**

- El manifest se sirve en `config=manifest`, `split=train`: **las 770 filas están en el
  mismo split de la API**, pero la columna `split` de cada fila es la que cuenta
  (517 `train` / 135 `val` / 118 `test`).
- La geometría es la **caja** (tile de ~25×25 km), no el perímetro del incendio. Para
  medir proximidad se usa el centro de la caja, lo que introduce un error de hasta ~17 km.
  La clasificación por código (62/96) **no depende** de este error.
- Se comprobó que **ningún código aparece en más de un split** de TMB-S3: la asignación
  es consistente a nivel de activación, no de caja.
