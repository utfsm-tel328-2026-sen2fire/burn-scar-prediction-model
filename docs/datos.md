# Catálogo de fuentes de datos

> Dónde está cada dato, con qué identificador exacto, cómo se accede y bajo qué licencia.
>
> **Verificado el 2026-11-06.** Los catálogos de CEMS y de Google Earth Engine cambian;
> el método para volver a comprobarlos está en [`investigacion.md`](investigacion.md) §4.

---

## 1. Tabla maestra

| Fuente | Rol | Identificador | Res. | Cadencia | **Latencia** | Cobertura | Coste |
|---|---|---|---|---|---|---|---|
| **CEMS Rapid Mapping** | **Etiquetas** | `EMSR###` | 10 m vector | por ronda | días-semanas (requiere activación) | global (sesgo hacia Europa) | Gratis |
| **Sentinel-3 OLCI** | Imagen principal | `COPERNICUS/S3/OLCI` | 300 m | ~2 días | **< 3 h** (NRT) / > 24 h (NTC) | global | Gratis |
| **Sentinel-2 SR** | Línea base pre-incendio | `COPERNICUS/S2_SR_HARMONIZED` | 10/20 m | 5 días | ~3 h | global | Gratis |
| **Sentinel-1 GRD** | Radar (penumbra) | `COPERNICUS/S1_GRD` | 10 m | ~6 días | ~1 h (NRT) | global | Gratis |
| **ERA5-Land** | Viento, T, humedad | `ECMWF/ERA5_LAND/HOURLY` | 11 km | horario | 5 d (`-T`) / 2-3 meses (consolidado) | global | Gratis (registro) |
| **ERA5 diario** | Agregados diarios | `ECMWF/ERA5/DAILY` | 27 km | diario | ídem | global | Gratis (registro) |
| **GFS / ECMWF HRES** | **Pronóstico** (viento futuro) | NOAA / ECMWF | 25-31 km | horario | ~6-12 h | global | Gratis |
| **GFAS** | Meteo near-real-time | NASA LAADS | 0,1° | 6-12 h | ~12 h | global | Gratis |
| **FIRMS / VIIRS** | Fuego activo + FRP | NASA FIRMS | 375 m | 12 h | ~1-3 h | global | Gratis |
| **Copernicus DEM** | Topografía | `COPERNICUS/DEM/GLO30` | 30 m | estático | — | global | Gratis |
| **ESA WorldCover** | Combustible / cobertura | `ESA/WorldCover/v200` | 10 m | anual | — | global | Gratis |
| **MCD64A1** | **Supervisor** del preentrenamiento | `MODIS/061/MCD64A1` | 500 m | mensual | ~6 semanas | global | Gratis (login) |
| **FireCCI51** | Supervisor de preentrenamiento a 250 m | `ESA/CCI/FireCCI/5_1` | 250 m | mensual | ~6 semanas | global | Gratis |
| **FireCCI S3.1.1** | **Tercera opinión** / ablación (300 m, Sentinel-3) | CEDA `10.5285/d441079fc...` | 300 m | mensual | ~6 semanas | global | Gratis |
| **TMB-S3** | **Generador de cicatriz** (congelado) + código listo | HF `links-ads/tmb-s3` | 300 m | diario | — | Europa | Gratis (CC BY 4.0 / MIT) |
| **WildFireSpread** | Dataset + modelos listos | HF + Kaggle | 1 km | diario | — | Mediterráneo | Gratis |
| **Mesogeos** | Cubo de drivers | Zenodo `10.5281/zenodo.7473331` | 1 km | diario | — | Mediterráneo | Gratis |
| **WildfireSpreadTS** | Benchmark | NeurIPS 2023 D&B | 375 m | diario | — | US | Gratis |
| **TS-SatFire** | Benchmark | HF `GT-Fire-Research-and-Innovation/ts-satfire` | 375 m | 12 h | — | CONUS | Gratis (CC BY 4.0) |

**No están en Google Earth Engine** (si hacen falta, hay que ir por STAC o LAADS):
HLS, VIIRS, Sentinel-3 SLSTR, FIRMS.

---

## 2. CEMS — Copernicus Emergency Management Service

La fuente de etiquetas. Y la única que da una **serie temporal de la cicatriz**.

### 2.1 Qué es

Cada vez que hay un disaster relevante, la UE activa el servicio. Un equipo de analistas
mapea el evento desde imágenes de satélite y publica:

- **Delineación** (`DEL`): el perímetro de lo quemado / inundado / deslizado.
- **Grading** (`GRA`): el daño sobre activos (edificios, carreteras).
- **First estimate** (`FEP`): una estimación rápida de baja resolución.

Y luego vuelve a mapear el mismo evento días después → **rondas de monitorización**
(`DEL_MONIT01`, `DEL_MONIT02`…). Cada ronda es una nueva observación de la cicatriz, con
su fecha. Encadenarlas da la curva de crecimiento.

### 2.2 Acceso por API (2023 en adelante)

```python
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"

# Listado completo, paginado, sin autenticación
url = f"{API}/public-activations-info/?limit=500"
while url:
    data = requests.get(url).json()
    activations += data["results"]
    url = data.get("next")

# Detalle de una activación (AOIs, productos, capas, imágenes)
detail = requests.get(f"{API}/public-activations/?code=EMSR922").json()["results"][0]
```

**Censo medido (2026-11-06):**

| | |
|---|---|
| Activaciones en la API | 268 (2023: 55, 2024: 69, 2025: 69, 2026: 75) |
| De ellas, incendio | **131** |
| De ellas, incendio y mediterráneas | **107** (14 países) |
| Productos de delineación (Med) | 489 |
| Series con ≥1 delineación | 130 |
| Series con ≥2 delineaciones | **96** |
| Transiciones de crecimiento | **359** |

> Re-verificado el 2026-11-08 contra la API: 268 activaciones (2023: 55, 2024: 69, 2025: 69,
> 2026: 75), 131 de incendio, 107 mediterráneas (14 países). Sin cambios.

**Límite duro:** el endpoint de detalle **devuelve 403 para todo código anterior a
EMSR656** (23/03/2023). Verificado con EMSR001, 100, 200, 300, 400, 450, 500, 550, 600,
610, 620, 630, 640, 650 y 655.

**Límite de ritmo:** la API responde **HTTP 429** si vas rápido. Una pasada completa de las
107 activaciones mediterráneas necesita concurrency bajo y backoff de varios segundos.

### 2.3 Acceso a 2012-2022 (la vía legacy)

> ⚠️ **No hay API para este periodo, pero los datos sí existen y son públicos.**

La página de cada activación no tiene el corte de 2023 y lista los productos con su fecha
de entrega:

```
https://mapping.emergency.copernicus.eu/activations/EMSR300/
```

Los paquetes vectoriales están en un bucket S3 con **patrón predecible**:

```
https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/
   {EMSR}/{EMSR}_{AOI2d}{NOMBRE}_{PRODUCTO}_v{n}_vector.zip
```

Ejemplo real (incendio de Ática, julio 2018, `EMSR300`):

```
EMSR300_01GERANEIA_01DELINEATION_MAP_v1_vector.zip
EMSR300_01GERANEIA_01DELINEATION_MONIT01_v1_vector.zip
EMSR300_01GERANEIA_01DELINEATION_MONIT02_v1_vector.zip
EMSR300_01GERANEIA_01DELINEATION_MONIT03_v1_vector.zip
EMSR300_01GERANEIA_01DELINEATION_MONIT04_v1_vector.zip
EMSR300_01GERANEIA_02GRADING_MAP_v2_vector.zip
```

Cada zip son ~3 MB (verificado: 3.094.039 bytes) con los shapefiles de la capa.

**Censo pre-2023 (recorriendo EMSR001-655):** 207 activaciones de incendio en total, 166
mediterráneas, **66 series con ≥2 delineaciones** y **131 transiciones**.

> Nota: el HTML de las activaciones antiguas usa "Delineation Map" / "Delineation MonitNN",
> mientras la API llama `monitoringNumber` 0 al producto inicial. *Rondas* = nº de productos
> DEL; *transiciones* = rondas − 1.

### 2.4 Qué trae cada producto de detalle

```jsonc
{
  "type": "DEL",
  "monitoringNumber": 3,              // 0 = inicial, 1..n = rondas
  "images": [
    { "sensorName": "Sentinel-2", "sensorType": "optical",
      "resolutionClass": "VHR2", "acquisitionTime": "2026-09-11T09:20:00" }
  ],
  "stats": {
    "Burnt area": { "None": { "unit": "ha", "affected": 114.0 } },   // área quemada ACUMULADA
    "Land use": { "Forests ": { "unit": "ha", "total": 1357.5, "affected": 45.4 } }
  },
  "layers": [
    { "name": "EMSR929_AOI01_DEL_MONIT03_..._ORTHO_cog.tif", "format": "cog" },
    { "name": "EMSR929_AOI01_DEL_MONIT03_notAnalysedA_v1_VT", "format": "vt",
      "json": "https://rapidmapping-viewer.s3.eu-west-1.amazonaws.com/..." },
    { "name": "EMSR929_AOI01_DEL_MONIT03_observedEventA_v1_VT", "format": "vt", "json": "..." },
    { "name": "EMSR929_AOI01_DEL_MONIT03_observedEventP_v1_VT", "format": "vt", "json": "..." }
  ]
}
```

**Las tres capas de vector que importan** (capas `vt`, cada una con su URL `json` a GeoJSON):

| Capa | Significado |
|---|---|
| `observedEventA` | **Polígono del evento**: el área quemada. Es la etiqueta |
| `notAnalysedA` | Dónde el analista miró y decidió que no era quemado. Base del negativo |
| `observedEventP` | Polígonos de quemado de tipo "traza" (línea), para fuegos estrechos |

> ⚠️ **La capa `notAnalysedA` es el "no quemado" correcto.** Fuera de ella no hay dato:
> no es "no quemado", es "nadie miró". Confundir las dos es la fuga silenciosa más
> peligroso: el modelo aprende a predecir en zonas donde nadie miró y se le penaliza por
> hacerlo mal.

### 2.5 Atajo: la curva de crecimiento sin geometría

`stats["Burnt area"]["None"]["affected"]` da el **área quemada acumulada en hectáreas por
ronda**. Con eso se obtiene la curva de crecimiento (y el sensor de cada ronda, vía
`images[].sensorName`) **sin descargar un solo shapefile**.

Ejemplo real (`EMSR929`, Trapezitsa, Grecia, septiembre 2026):

```
  fecha        sensor       ha
  2026-09-04   Sentinel-2    52,2   (primera estimación, FEP)
  2026-09-07   Legion        56,7
  2026-09-09   Legion        95,6
  2026-09-10   Legion       109,6
  2026-09-11   Sentinel-2   114,0
  2026-09-12   Sentinel-2   115,2
  2026-09-14   Sentinel-2   115,2
```

Sirve para filtrar series rápido (descartar las de una sola ronda) y para saber de
antemano con qué sensor se trazó cada etiqueta.

> ⚠️ **El sensor de cada ronda es metadato crítico, no un detalle.** En la curva de arriba, el
> paso de Sentinel-2 (10 m) a Legion (50 cm) ocurre entre las dos primeras rondas: parte del
> "crecimiento" medido ahí es **diferencia de detección**, no propagación. Guardar siempre
> el sensor de cada ronda y usarlo para **estratificar la evaluación** (riesgo R14 de
> `decisiones.md` §2). La regla que hay que cumplir al construir las muestras es: el target
> se ancla en la ronda siguiente (`r+1`) y la cicatriz de entrada la produce TMB-S3, no una
> ronda de CEMS (dataset.md §7 D1, D9).

> **FEP fuera.** El primer producto de cada serie (`FEP`, first estimate) es una estimación
> de baja resolución: no entra en la serie de etiquetas (dataset.md §7 D5).

### 2.6 Otras piezas de CEMS

| Recurso | Para qué |
|---|---|
| **EMSN** (Risk & Recovery) | Modalidad post-evento con delineación + **grading de severidad**. Otra fuente de etiquetas (no de crecimiento) |
| **EFFIS / GWIS** | European Forest Fire Information System: datos de incendios para Europa |
| **Zenodo 5789336** | `10.5281/zenodo.5789336` — 218 activaciones de incendio 2012-2020 rasterizadas a 30 m por año/temporada + un índice GeoJSON de 0,1 MB. **Útil para enumerar incendios antiguos**, no para crecimiento |
| **HDX** | `data.humdata.org` tiene paquetes por activación, pero **solo de 2026** (EMSR915-935) |

### 2.7 La librería `copernicus-ems`

```bash
pip install copernicus-ems
```

| Módulo / comando | Qué hace |
|---|---|
| `RapidMapping.activations(country=, categories=, since=, until=)` | Lista activaciones |
| `.products(code)` | Productos de una activación (con `observed_time` y máscara del AOI) |
| `.observed_events(code, event_types=, notations=)` | Polígonos como GeoJSON + `observed_time` + máscara de observación |
| `.ml.rasterize(events, resolution_m=)` | Raster **tri-estado**: `1`=quemado, `0`=no quemado *dentro del AOI*, `255`=ignorar |
| `.ml.evaluate(prediction, labels)` | Métricas **enmascaradas**: CSI, POD, FAR (IoU se deriva de CSI), solo dentro de lo observado |
| `.geo.summary(events)` | km² quemado y km² observado, por producto |
| `copernicus-ems dataset -o DIR --size 256` | Recortes para entrenamiento, con sidecar de procedencia y split por área |
| `copernicus-ems scenes CODE --aoi N --optical` | Adquisiciones Sentinel-1/2 **abiertas** más cercanas a un AOI, vía STAC de Copernicus Data Space |
| `copernicus-ems pairs CODE --aoi N` | VV/VH de Sentinel-1 sobre la rejilla de etiquetas (requiere credenciales CDSE) |
| CLI | `copernicus-ems list / fetch / export` |

Está construida para inundaciones: las constantes de tipo de incendio (`event_type`,
notations) están en su roadmap, pendientes de verificar con activaciones reales.

**Estado en GitHub**: `agent-engineering-studio/copernicus-ems`, licencia **Apache-2.0**,
tipada (mypy strict), `httpx` como única dependencia, extras opcionales `[geo]`, `[ml]`,
`[cdse]`, `[mcp]`.

> ⚠️ **Dos 403 distintos, no confundirlos.** (a) El endpoint de **detalle de activación**
> devuelve 403 para todo código anterior a EMSR656 (§2.2). (b) Los **paquetes de producto**
> (`{code}_products.zip`, ~100 MB) devuelven 403 para algunas activaciones concretas
> (p. ej. EMSR664). Las capas vectoriales en GeoJSON siguen siendo accesibles en ambos
> casos: es lo que hace la librería.

> **Recomendación: forkearla en vez de escribir el cliente desde cero.** Es exactamente el
> 80% del trabajo de la ingesta de etiquetas.

---

## 3. Imágenes

### 3.1 Sentinel-3 OLCI — el backbone

```
  ee.ImageCollection("COPERNICUS/S3/OLCI")
  Copia: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI
```

| | |
|---|---|
| Resolución | 300 m (21 bandas) |
| Revisita | ~2 días |
| Cobertura | 2016-10-18 → presente |
| Producto | **OLCI EFR, radiancia *top-of-atmosphere*** (no reflectancia de superficie) |
| Bandas | Oa01 (400 nm) … Oa21 (1020 nm) |

**Dos trampas importantes:**

1. **No hay SWIR.** Las 21 bandas cubren 400-1029 nm: visible + NIR. **No se puede calcular
   NBR ni dNBR con OLCI.** Para el cambio de quemado hay que usar el índice NIR
   (Oa21 a 1020 nm / Oa17 a 865 nm) y visible: el carbón es muy oscuro en NIR, así que la
   caída de NIR es una señal fuerte. El subconjunto de 5 bandas validado por TMB-S3 es
   `Oa21, Oa17, Oa08, Oa06, Oa04`.

2. **Es radiancia TOA, no reflectancia de superficie.** Como todos los frames pasan a la
   misma hora solar, **el ratio entre frames cancela la iluminación**: trabaja con ratios,
   no con reflectancia absoluta.

**TMB-S3 no incluye las banderas de calidad de OLCI** en la versión pública de GEE; si
hace falta máscara de nube fina, hay que ir a los datos de origen (Copernicus Data Space).

### 3.2 Sentinel-2 SR — línea base pre-incendio

```
  ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
```

Resolución 10/20/60 m, revisita 5 días, desde 2017-03. **Incluye la banda `SCL`**
(classificación de escena) → máscara de nubes y sombras fiable, que OLCI no trae.

Rol en este proyecto: **compuesto pre-incendio** (2-3 acquisition sin nubes → NBR, NDWI,
línea base de combustible) y, en fase 2, severidad vía dNBR. Es la única fuente con SWIR
y por tanto la única que permite NBR/dNBR.

> En GEE el asset es `COPERNICUS/S2_SR_HARMONIZED`. La versión "harmonized" corrige la
> deriva radiométrica entre S2A/S2B/S2C, necesaria si se mezclan satélites.

### 3.3 Sentinel-1 GRD — radar

```
  ee.ImageCollection("COPERNICUS/S1_GRD")
```

Resolución 10 m, revisita ~6 días, desde 2014-10. **Ve a través de nubes, humo y noche.**

Rol: es el **complemento essential** de OLCI, por dos razones:

1. **Radar ve bajo el dosel.** CEMS marca como quemado lo que ardió bajo árboles o en
   hierba fina; eso puede no producir cambio espectral en óptico pero sí en radar. Es la
   respuesta directa al riesgo R2.
2. **No le afecta el humo**, que es justamente lo que tapa las imágenes durante un incendio.

**Ajustes obligatorios en GEE:**

```python
.applyOrbitCorrection()   # corrección de fase orbital
.normalize()              # eliminación de ruido térmico (obligatorio desde 2021 con S1C/D)
```

Polarizaciones útiles: **VV** y **VH**. En Sentinel-1 el cociente VV/VH baja
notablemente tras un incendio.

### 3.4 Por qué no VIIRS (y cuándo recuperarlo)

VIIRS (Suomi-NPP, NOAA-20, NOAA-21) ofrece bandas de imagen a 375 m con un periodo de 12 h, y es
la mejor cadencia posible en datos abiertos: **8-16 observaciones por incendio** frente a
las 2-3 de Sentinel-3.

Descartado en la v1 porque **no está en GEE**: obliga a descargar swath L1B de LAADS
(`VNP02IMG`/`VNP03IMG`) y a reproyectarlos. Es la primera opción a recuperar si la
cadencia, y no la resolución, resulta ser el cuello de botella.

---

## 4. Variables de entorno

### 4.1 ERA5 — el viento

```
  ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")   # 11 km, horario
  ee.ImageCollection("ECMWF/ERA5/DAILY")          # 27 km, agregado diario
```

**Variables a usar:** componentes del viento `u` y `v` (m/s), temperatura, humedad
específica, precipitación.

**Dos versiones que hay que distinguir:**

| Producto | Latencia | Uso |
|---|---|---|
| `ERA5-Land` (consolidado) | 2-3 meses | Entrenamiento (histórico) |
| `ERA5-Land-T` (cerca de tiempo real) | 5 días | Lo más reciente que hay en el CDS |
| **Salida NWP en bruto** (GFS / ECMWF HRES) | **~6-12 h** | **Operación** |

> Para una predicción operativa, el producto ERA5 no sirve: incluso su versión de cerca de
> tiempo real va con 5 días de retraso, y además **el análisis describe el pasado**. Lo que
> hace falta para anticipar el frente es el **viento previsto**, que se obtiene del propio
> NWP (GFS o ECMWF HRES) en bruto, no del producto ERA5 reanalizado. Por eso la entrada
> meteorológica del modelo debería llevar **dos bloques**: observación (contexto: humedad,
> temperatura, sequedad del combustible) y **pronóstico** (qué viento sopla en las próximas
> 24-72 h, que es lo que realmente dirige la propagación).

> ⚠️ **Trampa de convención.** ERA5 publica el viento como vector *hacia dónde* sopla.
> La "dirección del viento" meteorológica es *de dónde* viene. Hay que invertir el ángulo
> (y el Bearing) para no apuntar el frente en la dirección contraria. Es un error fácil de
> cometer y muy difícil de detectar en las métricas.

> ⚠️ **Trampa de resolución.** A 11 km, un viento de 50 km/h puede estar mal representado
> en un incendio de 20 km de extensión. Usar las celdas-grid más próximas al centroide
> del AOI, no un promedio de toda la escena.

**Alternativa sin registro:** NASA **GFAS** (Global Fire Assimilation System) en
LAADS, diario y sin cuenta. Fuente de la doble verificación meteorológica y la opción
near-real-time sin clave de CDS.

### 4.2 Copernicus DEM — topografía

```
  ee.ImageCollection("COPERNICUS/DEM/GLO30")   # 30 m
```

Derivar pendiente y orientación con `ee.Terrain.slope()` y `ee.Terrain.aspect()`.

### 4.3 ESA WorldCover — combustible

```
  ee.ImageCollection("ESA/WorldCover/v200")   # 10 m, anual, 2021
```

Clase de cobertura del suelo. Define el tipo de combustible (bosque, matorral, hierba,
agricultura) → controla tanto la inflamabilidad como la velocidad de propagación.

### 4.4 MCD64A1 — fecha de quema (MODIS)

```
  ee.ImageCollection("MODIS/061/MCD64A1")   # 500 m
```

Aunque se publica como producto mensual, el payload es **día de quema por píxel**:

| Capa | Contenido |
|---|---|
| `BurnDate` | Día del año en que quemó (1-366); 0 = no quemado, −1 = sin dato, −2 = agua |
| `Uncertainty` | Incertidumbre en días |
| `FirstDay` / `LastDay` | Ventana de detección fiable del mes |
| `QA` | Bit 2 = periodo de mapeo acortado (nube); bits 5-7 = "observaciones demasiado espaciadas" (descartar) |

**Por qué importa:**

1. **Es 100% reproducible con datos abiertos.** Contrasta directamente al 82% de las
   etiquetas de CEMS que vienen de imagen comercial (riesgo R2).
2. **Permite deriva una curva de crecimiento diaria a 500 m, global, desde 2000.** Es el
   dataset de preentrenamiento natural: millones de transiciones para aprender la física
   de la propagación, y luego afinar con las 46 series limpias de CEMS.

**Su papel en este proyecto: supervisor del preentrenamiento, nunca entrada.** Es
precisamente por tener ~6 semanas de retraso por lo que es un buen supervisor: **no puede
filtrar información que en producción no tendrías**. Pero por la misma razón **no puede
entrar en el modelo como canal de entrada** (dataset.md §8).

**Trampas:**

- **Deriva orbital.** Terra dejó el mantenimiento de órbita en 2020 (de 10:30 a ~9:00) y
  Aqua se desplazó de 13:30 a ~15:50 (hasta 2026). La detección pre-2020 y post-2020 **no
  es homogénea**: no mezclar años sin marcar el dominio.
- Latencia de ~6 semanas tras el cierre del mes. Sirve para entrenar, no para predecir en vivo.
- El "día de quema" es el día en que el algoritmo **detectó** el cambio, no el día en que
  ardió. La capa `Uncertainty` cuantifica ese ruido. Es un desfase de R19: hay que medirlo
  contra `observed_time` de CEMS sobre nuestros eventos antes de fiarse del preentrenamiento.
- **Fin de vida de MODIS: el reloj se acorta.** Suomi NPP cesa sus datos el **2026-11-01** y
  Terra/MODIS tiene la fecha de apagado anunciada. El sucesor es **VNP64A1** (NOAA), a
  **500 m** (no 375 m), desde marzo de 2012, con peor exactitud temporal que MCD64A1
  (38% de detección el mismo día frente a 46%). **MCD64A1 tiene años contados**: el
  preentrenamiento a largo plazo debería pensar en VNP64A1 o en la combinación de ambos.

---

## 5. Datasets listos para usar

### 5.1 TMB-S3 — el andamiaje de código

| | |
|---|---|
| Dataset | HF `links-ads/tmb-s3` · 246 activaciones CEMS europeas (2016-2025) · 770 cajas de ~25×25 km |
| Estructura | Zarr por caja: `s3olci` (T, 21, H, W) + `firemask` (10 m) + `validity` (10 m) + `landcover` (10 m) |
| Splits | 146 / 50 / 50 eventos (train/val/test), **estratificados geográficamente con DBSCAN (150 km)** |
| Código | `github.com/links-ads/tmb-s3` · **MIT** · `uv`, Python ≥3.12, PyTorch CUDA 12.8, W&B |
| Modelos | HF `links-ads/tmb-s3-models` — 4 SegFormer (2D early-fusion y ConvLSTM, 21 y 5 bandas) |
| Resultados | F1 65,5-68,2 · IoU 56,7-58,7 (media de 3 semillas; solo los 4 checkpoints SegFormer publicados) |
| Licencia | Dataset CC BY 4.0 |

**Lo que aporta**: el esqueleto completo (dataset zarr, datamodule Lightning con collation
de longitud variable, transforms con early-fusion, ConvLSTM sobre SegFormer, inferencia con
ventana deslizante 64×64 stride 32, BCE ponderada) y un **protocolo de split ya resuelto**,
con un matiz: separa por cluster geográfico (DBSCAN, ε=150 km) y estratifica por cluster y
tamaño de evento, pero **sus propios autores listan como limitación la fuga espacial
residual** ("nearby events from the same geographic cluster can fall in different splits").

**Lo que hay que corregir al reutilizarlo:**

1. **La regla de reetiquetado.** Usa "quemado si **cualquier** píxel de 10 m dentro lo
   está", lo que infla el área un 34% (21,0% de píxeles quemados bajo la regla OR frente a
   15,7% de área fraccional real). Para propagación, usar **fracción de área**.
2. **La selección de checkpoints**: el paper selecciona por **mejor F1 de validación**
   ("the checkpoint with the best validation F1 is retained"), pero **no documenta qué
   semilla de las 3 se publica**. Reentrenar y seleccionar por validación.
3. **El dataset no trae las geometrías multi-ronda**, solo el perímetro final → hay que
   reingerir CEMS para las etiquetas de crecimiento.

**Su papel en este proyecto: generador de cicatriz congelado.** Es la pieza que produce el
canal `scar` de la entrada en las tres fases (preentrenamiento, ajuste e inferencia). Por
eso sus sesgos se heredan, y por eso hay que medirlos sobre nuestras propias series
(operacion.md §4).

**La auditoría de splits (paso 0, bloqueante).** El dataset incluye un
`manifest.parquet` con las columnas `split` (`train`/`val`/`test`) y `bbox_id` con formato
`EMSR168-1-0` (código de activación, índice de AOI, ronda). Cruzándolo con nuestros 96
eventos mediterráneos se puede saber **qué incendios y qué geografía ya vio TMB-S3**, y por
tanto construir un split en el que test y val contengan **solo eventos que TMB-S3 no ha
visto** (decisiones.md §3, R18). El mismo cruce sirve para excluir esos eventos del
pool de preentrenamiento con MCD64A1 (R20).

### 5.2 WildFireSpread / Mesogeos

| | |
|---|---|
| Dataset | HF + Kaggle · ~9.500 muestras · parches 64×64 km a 1 km/px · 10 días (5 antes + 5 después de la ignición) · 27 variables · netCDF · 11,1 GB zip |
| Modelos | UNet2D, UNet3D, Vision Transformer, ConvLSTM (checkpoints en HF) |
| Cubo | `mesogeos_cube.zarr` · 1 km × 1 km × 1 día · 2006-04-01 → 2022-09-29 · 31 países |
| Variables | `burned_areas`, `ignition_points` (EFFIS) · ERA5-Land · MODIS (LST, NDVI, LAI) · JRC EDO · WorldPop · C3S land cover · EU-DEM |
| Licencia | Dataset **CC BY 4.0** · Zenodo `10.5281/zenodo.7473331` (concept DOI, apunta a la última versión; la v1.2 concreta es `8036851`) · código Apache-2.0 |

**Aviso importante:** el `burned_areas` de Mesogeos viene de **perímetros finales de EFFIS**,
no de rondas de monitorización. Soporta la tarea "área final", **no** propagación.

**Segundo aviso:** lo aleatorio en este dataset es el **offset del parche** (cada incendio se
recorta con un desplazamiento aleatorio, para que la quemadura no quede siempre en el
centro), **no el split**. El split es **temporal por años** (2006-2020 train / 2021 val /
2022 test), elegido explícitamente para no filtrar incendios cercanos en el tiempo. Eso mide
generalización *en el tiempo*, no *en el espacio*: un incendio de 2022 puede estar en la
misma comarca que otros de 2019 que ya estaban en train. Para medir generalización a
regiones no vistas hay que re-separar por cluster geográfico.

### 5.3 WildfireSpreadTS (WSTS) y TS-SatFire

| | WSTS | TS-SatFire |
|---|---|---|
| Eventos | 607 (US, 2018-2021); WSTS+ añade 4 años | 179 (CONUS, 2017-2021) |
| Resolución | 375 m | 375/750 m |
| Entradas | 23 canales multimodales (combustible, topografía, clima) | 27 canales (VIIRS + SRTM + GRIDMET + GFS) |
| Tarea | propagación día+1 | AF + BA + progresión |
| Acceso | NeurIPS 2023 D&B | HF `GT-Fire-Research-and-Innovation/ts-satfire` (CC BY 4.0) · Kaggle |

Ambos sirven como **referencia de métricas** contra las que comparar, y como fuente de
código para el tratamiento de imbalance y el protocolo de evaluación.

### 5.4 FireCCI — la alternativa con Sentinel-3 (y su trampa)

FireCCI produce mapas globales de área quemada. Las dos versiones que interesan:

| | **FireCCIS311** (Sentinel-3) | **FireCCI51** (MODIS) |
|---|---|---|
| Sensor de entrada | **Sentinel-3 OLCI + SLSTR (Synergy)** + VIIRS térmico | MODIS NIR + focos activos de Terra/Aqua |
| Resolución | **300 m** (0,002777°) | **250 m** (0,00224573°) |
| Periodo | **2019-2024** (ampliado en 2025-10) | 2001-2022 (en GEE, hasta 2020-12) |
| Capas | **JD** (día juliano de la primera detección) · **CL** (confianza 0-100) · LC (cobertura C3S v2.1.1) | **JD** · **CL** · LC |
| Acceso | CEDA, 75 GB, GeoTIFF mensuales, 6 tiles continentales, OPeNDAP · DOI `10.5285/d441079fc77f49fabeb41330612b252f` · **no está en GEE** | **GEE**: `ESA/CCI/FireCCI/5_1` · CEDA (57 GB) · DOI `10.5285/58f00d8814064b79a0c49662ad3af537` |
| Licencia | Open access, cualquier uso | Open access, cualquier uso |

**Qué aporta cada una:**

- **FireCCI51** es MCD64A1 a **250 m** en vez de 500 m, con **etiqueta blanda** (la capa CL
  es la probabilidad de que el píxel esté realmente quemado). Bueno como supervisor de
  preentrenamiento. Se queda en 2020-2022, así que no cubre la parte reciente.
- **FireCCIS311** tiene la **misma resolución y el mismo sensor** que la entrada del
  modelo, y además capa de confianza. Sobre el papel es el mejor supervisor posible.

**La trampa: la etiqueta es reconstruible desde la entrada.** FireCCIS311 se generó
mirando la reflectancia de **OLCI**, que es exactamente lo que el modelo recibe. Un
modelo entrenado contra FireCCIS311 puede limitarse a **replicar el clasificador de
FireCCI** en vez de aprender física de propagación.

> **Por eso FireCCIS311 no sustituye a CEMS como conjunto de evaluación**
> (`investigacion.md` §2, R21). CEMS lo dibujaron analistas mirando imagen comercial (80 %)
> y Sentinel-2 (18 %): es la única fuente cuyo proceso **no pasa por las imágenes del modelo
> en el frente** — el Sentinel-2 que usa el modelo es solo línea base pre-incendio.
>
> **Jerarquía de independencia de etiquetas:**
>
> | Fuente | ¿Generada con…? | ¿Recuperable desde OLCI? | Uso |
> |---|---|---|---|
> | **CEMS** | analistas, imagen comercial + S2 | **No** (el S2 del modelo es solo línea base pre-incendio) | **evaluación** |
> | MCD64A1 / FireCCI51 | reflectancia **MODIS** | parcialmente | preentrenamiento |
> | FireCCIS311 | reflectancia **OLCI** | **casi directamente** | ablación / tercera opinión |

**El papel de FireCCIS311 en este proyecto:** (a) **tercera opinión** para contrastar las
etiquetas de CEMS —el 80% viene de imagen comercial—, y (b) ablación de supervisor de
preentrenamiento a 300 m.

**Trampas comunes a las dos:**

- La fecha de detección **no es la fecha de quemado**: con nubes puede ser varios días
  posterior al final del fuego (R19). Igual que MCD64A1.
- El quemado es **binario** ("se asume el píxel completo quemado"), no fraccional. La capa
  CL sirve de peso, pero no de área fraccional.
- El producto **se reinicia cada mes**: `JD` es el día dentro del mes. Para una serie
  temporal hay que concatenar los meses, y un píxel quemado en junio y otra vez en julio
  aparece como quemado en julio.

---

## 6. Acceso en tiempo real (operación)

Cómo se accede a cada dato cuando el objetivo es **predecir durante el incendio**, no
entrenar. El razonamiento completo está en [`operacion.md`](operacion.md).

### 6.1 Requisitos de cuenta

| Fuente | Registro | Coste |
|---|---|---|
| Copernicus Data Space (S1/S2/S3) | Obligatorio | Gratis |
| NASA Earthdata (MCD64A1, VIIRS) | Obligatorio (EARTHDATA login) | Gratis |
| NASA LAADS (GFAS, MODIS) | Obligatorio | Gratis |
| CDS de Copernicus (ERA5) | Obligatorio | Gratis |
| NOAA (GFS) | No | Gratis |
| Google Earth Engine | Obligatorio | Gratis (con cuotas) |
| CEMS | No para leer; **activación solo por autoridad autorizada** | Gratis |

### 6.2 NRT frente a NTC

Los productos Sentinel se publican en dos regímenes:

| Régimen | Latencia | Para qué |
|---|---|---|
| **NRT** (Near Real Time) | **< 3 h** para OLCI L1/L2 | Operación: predicción durante el incendio |
| **STC** (Short Time Critical) | < 48 h | Compromiso intermedio |
| **NTC** (Non Time Critical) | > 24 h (en la práctica 48-72 h) | Estudio: incorpora órbita precisa y meteo de análisis en vez de pronóstico |

> La diferencia técnica entre NRT y NTC, en Sentinel-3, está en los **valores meteorológicos
> asociados**: NRT usa **pronóstico** ECMWF y NTC usa **análisis**. Para un modelo que
> predice propagación, la reflectancia de nivel 1 es idéntica; lo que cambia es la
> disponibilidad.

**Advertencia operativa:** la ventana del archivo NRT es **limitada** (del orden de un mes
a un año según el nivel de producto). Para recuperar histórico hay que usar NTC.

**Aviso de mayo 2026:** un problema de ground segment hizo que los productos NRT y STC de
S3A y S3B se publicaran fuera de la timeliness nominal (OLCI se recuperó el 21/05/2026).
**Un pipeline operacional tiene que monitorizar la latencia real, no asumirla.**

### 6.3 Qué se descarga en cada iteración

En cada ciclo del evento (cada ~2-3 días, operacion.md §5):

```
  Sentinel-3 OLCI  ->  ventana temporal sobre el AOI, 5 bandas
  Sentinel-1 GRD   ->  ventana temporal sobre el AOI, VV + VH
  Sentinel-2 SR    ->  1-2 frames pre-incendio (composición en memoria)
  ERA5 / GFAS      ->  u, v, T, humedad, precipitación (diario)
  GFS / ECMWF HRES ->  viento y humedad PREVISTOS a 24-72 h
  FIRMS / VIIRS    ->  focos activos + FRP de las últimas 48 h
  DEM, WorldCover  ->  estáticos, cacheados por AOI
```

Solo ventanas: **nunca se descarga el archivo completo de un satélite.** El coste por
iteración es de decenas de MB.

### 6.4 La fuente que no puede estar en el camino de inferencia

**CEMS no puede estar en el camino de inferencia.** Es un servicio de emergencia que
requiere que una autoridad autorizada lo active, y sus productos tardan de días a semanas.
Es una fuente de **etiquetas** y de **ground truth**, nunca de entrada en tiempo real.

De ahí la necesidad de un **generador de cicatriz propio** (ver `operacion.md` §3-4): la
"cicatriz conocida" tiene que salir de un modelo, no de un analista. En este proyecto ese
generador es **TMB-S3 congelado** (§5.1), que usa el mismo sensor, la misma resolución y
las mismas etiquetas que CEMS.

**Y una regla que no se puede violar:** MCD64A1 y CEMS **nunca** están en el camino de
inferencia. MCD64A1 llega con ~6 semanas de retraso y CEMS requiere una activación de
emergencia. Solo aparecen en el entrenamiento (MCD64A1 como supervisor del
preentrenamiento, CEMS como etiqueta del ajuste y como métrica).

### 6.5 Cadencia y producto final

| | |
|---|---|
| Revisor más lento | Sentinel-3 OLCI, ~2-3 días |
| **Cadencia del producto** | **cada ~2-3 días**, no diario |
| Para un producto diario | VIIRS (12 h, 375 m) → la v2 (decisiones.md §4) |

---

## 7. Créditos y atribución

### 7.1 CEMS

> © Union Europea. Copernicus Emergency Management Service (CEMS).
> Reutilización autorizada **indicando la fuente**. Activaciones EMSR individuales con sus
> propios términos (ver la ficha de cada activación en
> `data.jrc.ec.europa.eu`, DOI `10.2905/JRC.…`).

### 7.2 Sentinel / Copernicus

> Contains modified Copernicus data (year). Sentinel-2 OLCI: © ESA.
> Processed by the user; the products are distributed under the
> **Copernicus Data Terms and Conditions**.

### 7.3 Módulos de la Copernicus Data Space Ecosystem

Copernicus Sentinel data (S1/S2/S3), Sentinel cooperated by ESA, processed by CDSE.
Los productos de terceros (WorldCover, DEM, etc.) tienen su propia licencia.

### 7.4 ERA5 / ECMWF

> Copernicus Climate Change Service (C3S) / ERA5: © Copernicus Climate Change Service
> (C3S) / ECMWF. Licensed under the **Copernicus licence**.

Requiere registro en el CDS (gratuto) para acceso programático.

### 7.5 ESA WorldCover

> © ESA WorldCover project 2021 / Contains modified ESA WorldCover data.
> **CC BY 4.0**.

### 7.6 Copernicus DEM

> Produced by the European Union and the Copernicus Programme.
> EU-DEM adaptation: © European Union.

### 7.7 NASA MODIS / MCD64A1

> MCD64A1: © NASA LP DAAC / MODIS Land Science Team. **Datos abiertos de NASA.**
> Atribución recomendada (no obligatoria salvo por la política de citación de la NASA).

### 7.8 Datasets de terceros

| Dataset | Licencia |
|---|---|
| TMB-S3 (dataset / código) | CC BY 4.0 / MIT |
| TS-SatFire | CC BY 4.0 |
| Mesogeos | Zenodo, Apache-2.0 (código) |
| MTBS | Dominio público (USGS) |
| Zenodo 5789336 | Zenodo, con atribución |

> **Nota práctica:** la licencia de **ERA5** (Copernicus) es la única con cláusula de uso
> no comercial explícita. Para un proyecto universitario es suficiente, pero conviene
> revisarlo si el trabajo se difunde fuera del ámbito académico o se wanna publicar
> resultados derivados.

---

## 8. Referencias

**Artículos y repos**

- *Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean* — NeurIPS 2023 D&B. Zenodo `10.5281/zenodo.7473331`
- *WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction* — NeurIPS 2023 D&B. `proceedings.neurips.cc`
- *TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset for Wildfire Detection and Prediction* — Scientific Data (2025). `nature.com/articles/s41597-025-06271-3`
- *Temporal Modelling for Burn Scars on Sentinel-3* (TMB-S3) — arXiv 2609.34596. `github.com/links-ads/tmb-s3`
- *Wildfire spread forecasting with Deep Learning* — arXiv 2505.17556 (Mediterráneo 2006-2022, dataset y modelos liberados)
- *High-Resolution Semantic Segmentation of Woodland Fires Using RAUNet and Time Series of Sentinel-2* — Remote Sensing 15(5):1342
- *Near Real-Time Wildfire Progression Monitoring with Sentinel-1 SAR Time Series and Deep Learning* — Remote Sensing (2020)
- *Deep Learning for Global Wildfire Forecasting* — arXiv 2211.00534
- *Copernicus EMS fire activations delimitations (2012-2020) rasterised at 30 m* — Zenodo `10.5281/zenodo.5789336`

**Documentación de producto**

- Sentinel-2 Mission Status, ESA (2026) — estado de la constelación S2A/B/C/D
- Sentinel-1 Facts and figures, ESA — https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Facts_and_figures
- Sentinel-3 OLCI User Guide — EUMETSAT
- NISAR Data User Guide — Alaska Satellite Facility (ASF)
- MCD64A1 User Guide v6 — LP DAAC
- copernicus-ems — `github.com/agent-engineering-studio/copernicus-ems`
