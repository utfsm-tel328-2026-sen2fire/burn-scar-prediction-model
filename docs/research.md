# Investigación y decisiones de diseño

> Documento de trabajo. Recoge el estado del arte, los datos disponibles, las decisiones
> tomadas y sus alternativas descartadas.
>
> **Datos verificados el 2026-11-06.** Los catálogos de CEMS y de Google Earth Engine
> cambian; el método para volver a comprobarlos está en §10.

---

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

Por eso el modelo usa **imágenes abiertas + variables meteorológicas y de terreno abiertas**
(ERA5/GFAS, Copernicus DEM, WorldCover). Nada de esto tiene coste ni licencia comercial.

**El enfoque elegido:**

| | |
|---|---|
| Etiquetas | CEMS EMSR: delineaciones de quemado fechadas por ronda de monitorización |
| Entrada | Sentinel-3 OLCI (revisita ~2 días) + Sentinel-1 (radar, ~6 días) + Sentinel-2 (línea base pre-incendio) + ERA5 + GFS + DEM + WorldCover |
| Salida | Probabilidad de quema por píxel a horizonte variable |
| Resolución | 300 m |
| Región | Mediterráneo / UE |
| Modelo | Encoder ConvLSTM + SegFormer + una cabeza de propagación. La cicatriz la genera **TMB-S3 congelado** (§6.4) |
| En operación | Ciclo de ~2-3 días, marcado por la revisita de Sentinel-3 (§6.1) |

**El problema conocido del dataset**: 96 series con señal de crecimiento es un número bajo
para generalizar (WSTS tiene 607, TS-SatFire 179, TMB-S3 246). Y la auditoría de splits
(§9.4.1) reduce el conjunto útil a **46 series**, porque TMB-S3 se sabe de memoria 50 de
ellas (§6.4). Es el principal frente abierto y la razón de las dos palancas de ampliación
(§9.1).

**Las tres decisiones que definen el proyecto**, y que se互补an:

1. **La cicatriz la pone TMB-S3, congelado.** Quita CEMS del camino de inferencia sin
   crear un desajuste entre entrenamiento y operación: la cicatriz siempre sale de un
   modelo (§6.3-6.4).
2. **Se audita qué sabe TMB-S3.** Resultado: 62 de las 96 series están dentro de su
   dataset. Se excluyen las 50 que están en su `train`/`val`, y quedan 46 (§9.4.1).
3. **Con 46 series, el reparto fijo desperdicia el 80%.** Por eso se usa **validación
   cruzada agrupada**: las 46 series se evalúan todas, cada una una vez (§5.4).

---

## 2. Glosario

Siglas y términos que aparecen en el documento, explicados.

| Término | Significado |
|---|---|
| **AOI** (*Area of Interest*) | Recinto geográfico que CEMS delimita para mapear un evento |
| **Ronda de monitorización** (*MONIT01, MONIT02…*) | Nuevo mapeo del mismo incendio días después. Es lo que da la serie temporal |
| **Delineación** (*DEL*) | Producto vectorial de CEMS con el perímetro de lo quemado |
| **Revisita** | Cada cuánto pasa el satélite por el mismo punto |
| **SAR / radar** | Sensor de ondas de radio. **Ve a través de nubes y humo** |
| **Óptico** | Sensor de luz. Muy detallado, pero nubes y humo lo tapan |
| **NIR / SWIR** | Infrarrojo cercano / infrarrojo corto. El bosque rebota mucho NIR; el suelo quemado, no |
| **NBR** | `(NIR − SWIR) / (NIR + SWIR)`. El índice clásico de quemado |
| **dNBR** | Diferencia de NBR antes y después del fuego. Da la **severidad** |
| **TOA** | *Top-of-atmosphere*. Reflectancia/radiancia medida en el sensor, sin corregir a superficie |
| **OR / rules of thumb** | Regla práctica empírica |
| **Ground truth (GT)** | La "verdad" contra la que se mide el modelo |
| **Split** | Cómo se reparte el dataset en train / validación / test |
| **Fuga espacial** | Que el mismo incendio (o su vecino) aparezca en train y test. Infla los resultados |
| **U-Net / SegFormer / ConvLSTM / ViT** | Arquitecturas de red neuronal. U-Net y SegFormer = extractoras espaciales; ConvLSTM = añade memoria temporal; ViT = transformer |
| **IoU / F1** | Métricas de acierto, 0-100 |
| **BCE ponderada** | Pérdida de entrenamiento que compensa el desbalance de clases |
| **zarr / netCDF / GeoTIFF** | Formatos de almacenamiento |
| **STAC** | Catálogo estándar para buscar escenas satelitales por metadatos |
| **UTM** | Sistema de proyección por zonas, en metros |

---

## 3. Estado del arte

### 3.1 Tres tareas distintas que se confunden

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
  complication adicional de que hay que anticipar **cuándo se detiene** el fuego (la cola),
  que no es observable en el momento de la predicción.

**Formulación elegida: (B) multi-horizonte.** Es la que da la señal de entrenamiento más
densa (las rondas de CEMS son etiquetas intermedias, no solo una etiqueta final) y el
perímetro final se deriva agregando horizontes.

### 3.2 La física: por qué la imagen sola no predice

El frente de un incendio se propaga en la dirección del viento, influenced por:

- **Viento** (velocidad y dirección): el factor dominante en la propagación de superficie.
- **Humedad y temperatura**: determinan la inflamabilidad del combustible.
- **Combustible**: tipo, carga y continuidad (más importante en sotobosque y matorral).
- **Topografía**: pendiente (la pendiente acelera en ascenso) y orientación (exposición solar).
- **Vulnerabilidad**: caminos, líneas de fuego, zonas con continuidad de combustible.

De los cinco, **solo la topografía y el combustible se pueden inferir de una imagen**. El
viento, la humedad y la temperatura **no se ven en ningún satélite**: son datos
meteorológicos. Por eso cualquier modelo que se pretenda predecir propagación los necesita.

### 3.3 Benchmarks y datasets públicos

| Dataset | Tarea | Res. | Eventos | Periodo | Qué aporta |
|---|---|---|---|---|---|
| **WildfireSpreadTS** (WSTS) | Propagación día+1 | 375 m | 607 | US, 2018-2021 | Benchmark de referencia; 23 canales multimodales. **El más comparable a este proyecto** |
| **WSTS+** | Propagación | 375 m | +4 años | US | Extensión de WSTS con más años y geografía |
| **TS-SatFire** | AF + BA + progresión | 375 m | 179 | CONUS, 2017-2021 | VIIRS día+noche; QA manual de etiquetas. 71 GB, CC-BY |
| **Mesogeos** | Peligro + extent final | 1 km | ~9.500 muestras | Mediterráneo, 2006-2022 | Cubo diario armonizado con 30 variables de drivers. **NeurIPS 2023 D&B** |
| **WildFireSpread** | Extent final | 1 km | ~9.500 | Mediterráneo, 2006-2022 | El dataset + modelos de Mesogeos, listos para entrenar |
| **TMB-S3** | Delineación con modelo temporal | 300 m | 246 activaciones CEMS | Europa, 2016-2025 | 246 activaciones CEMS con Sentinel-3 OLCI; **código MIT reutilizable** |
| **Mediterranean DL** | Extent final | var. | Mediterráneo | 2006-2022 | 10 timesteps (−4d a +5d post-ignición), 27 variables, dataset y modelos liberados |
| **FireCCI S311** | Quemado + fecha | 300 m | global | 2019-2024 | Sentinel-3 Synergy + VIIRS, binario. **Solo como ablación**: MCD64A1 es el supervisor principal (más cobertura, fecha diaria, 2000-2026) |
| **FireCCI51** | Quemado + fecha | 250 m | global | 2001-2022 | MODIS. 60 GB en CEDA, también en GEE |
| **MCD64A1** | Quemado + fecha | 500 m | global | 2000-2026 | **Día de quema por píxel**. Libre, reproducible, enorme |
| **MTBS** | Severidad + perímetro | 30 m | ~EE.UU. | 1984-2024 | Solo EE.UU. y snapshot anual; no crecimiento intra-incendio |

### 3.4 Hallazgos de la literatura que condicionan el diseño

1. **El frame pre-incendio es imprescindible.** TMB-S3 (arXiv 2609.34596) encuentra que el
   modelado temporal (ConvLSTM) solo mejora cuando hay una adquisición pre-fire en la
   secuencia; con frames post-fire solos el gain es despreciable. *Los frames post-fire
   comparten la misma firma de quemado, así que no hay señal de cambio que propagar.*
   → **El pipeline debe siempre incluir una línea base pre-incendio.**

2. **El etiquetado binario en el borde infla el área.** TMB-S3 reetiqueta a 300 m con la
   regla "quemado si **cualquier** píxel de 10 m dentro lo está", y reporta una inflación
   del **34%** del área quemada (×3 en incendios pequeños). Para una tarea de *crecimiento*
   esto es peor: el error se concentra justo en la banda del frente, que es lo que se
   intenta predecir.
   → **Etiquetado cuantificado** (fracción quemada por píxel) en vez de OR lógico.

3. **Cinco bandas bastan.** TMB-S3 demuestra que un subconjunto de 5 bandas OLCI iguala
   al conjunto completo de 21. → **Reduce el volumen del dataset ×4.**

4. **Separación espacial obligatoria.** Mesogeos y WildFireSpread usan un **split temporal
   por años** (2006-2019 / 2020 / 2021-2022 en Mesogeos; 2006-2020 / 2021 / 2022 en
   WildFireSpread), elegido explícitamente para no filtrar incendios cercanos en el tiempo.
   Eso mide generalización **en el tiempo**, no **en el espacio**: un incendio de test puede
   caer en la misma comarca que otros de train. TMB-S3 estratifica por cluster geográfico
   (DBSCAN, ε=150 km), aunque sus propios autores avisan de que eventos cercanos de un
   mismo cluster pueden acabar en splits distintos.
   → **Split por cluster geográfico, nunca aleatorio.**

5. **Las imágenes comerciales no están disponibles para el usuario final.** En CEMS, el
   80% de las delineaciones mediterráneas 2023-2026 (390 de 489) se hizo sobre sensores
   comerciales (Legion, SPOT, WorldView, Pleiades, GeoEye, PlanetScope); solo el 18%
   (87) salió de Sentinel-2, y un 2% no declara imagen. Ver §7 (riesgo R2).

6. **Reutilizar un modelo pretrained implica auditar sus splits.** Si TMB-S3 (que entrenó
   con CEMS) se usa como generador de cicatriz, hay que comprobar qué geografía ya vio:
   su `manifest.parquet` permite hacerlo. Sin esa auditoría, el modelo de propagación
   hereda una fuga espacial que no es visible (R18, §9.4.1).
   → **Auditoría de splits antes de construir el dataset.** *(Ejecutada: 62 de las 96
   series están en TMB-S3. Ver §9.4.1.)*

### 3.5 Arquitecturas de referencia

- **ConvLSTM + SegFormer** (TMB-S3): backbone espacial SegFormer, encoder temporal ConvLSTM.
  Es la opción por defecto: probada, con código MIT, y con checkpoints publicados.
- **UNet3D**: convoluciones 3D sobre la pila temporal. Más simple, peor rendimiento.
- **Vision Transformer** (WildFireSpread):Patch embedding sobre la secuencia temporal.
- **ConvNeXt-UPerNet**: backbone más potente, a cambio de más parámetros.

### 3.6 Qué etiquetas sirven para medir, y cuáles no

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
> por las imágenes del modelo. Por eso es la que puede usarse de examen, y por eso las otras
> tres sirven de prácticas (R21, D16).

> **Y por qué FireCCI S3.1.1 no sustituye a CEMS.** Sobre el papel es ideal: 300 m, Sentinel-3,
> 2019-2024, con fecha de detección y nivel de confianza por píxel. Pero se generó mirando
> OLCI. Como **ejercicio** es buenísimo (mejor resolución que MCD64A1, con etiqueta blanda);
> como **examen**, mediría la capacidad de copiar un clasificador. Su papel: contrastar
> etiquetas de CEMS (el 80% viene de imagen comercial) y servir de ablación.

---

## 4. Por qué 300 m

### 4.1 La tensión

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
(§4.4): VIIRS no está en GEE.

### 4.2 La revisita de la constelación no es el número de observaciones

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

### 4.2bis Cuándo existen datos OLCI (esto acota el proyecto)

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

> **El pool de preentrenamiento es 2016 → 2026, no 2000 → 2026.** MCD64A1 tiene etiquetas
> desde 2000, pero **sin OLCI no hay entrada**, así que antes de 2016 no hay muestra
> posible. Esto reduce el pool a ~10 años y es la razón de que la v2 (o un dataset
> distinto) sea el sitio donde está el margen de ampliación.

> **Un cambio de dominio en mayo 2018.** Hasta entonces solo hay S3A (una geometría de
> observación); desde mayo 2018 hay S3A + S3B (dos geometrías, 140° de separación en fase).
> El preentrenamiento que abarque ambos periodos mezcla dos dominios de iluminación
> (R22). Es un argumento más para **empezar por 2018-2026** y no por 2016: menos dominio
> mezclado y menos descarga.

### 4.3 La evidencia empírica: el incremento típico por ronda

Medido sobre las 24 series con estadísticas completas de área quemada (§10):

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

### 4.4 Alternativas descartadas

| Opción | Por qué no |
|---|---|
| **10 m (Sentinel-2, 5 d)** | 1-2 observaciones por incendio: no hay dinámica que aprender. Y el SWIR de S2 sí es útil, pero entra como línea base estática, no como serie temporal |
| **VIIRS 375 m / 12 h** | La mejor cadencia posible en abierto (8-16 observaciones por incendio). Descartado solo porque no está en GEE y obliga a procesar swaths L1B a mano. **Es la opción a recuperar en la v2** |
| **1 km (Mesogeos)** | Demasiado grueso para la dinámica de un frente. Sirve como fase 0 y como baseline, no como modelo final |
| **NisAR L-band** | Archivo desde agosto 2025: insuficiente para entrenar. Relevante a partir de 2028 |

### 4.5 Matiz: por qué no 30 m con HLS

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

## 5. Diseño del dataset

### 5.1 Composición

Una **serie** = un (incendio, AOI) con delineaciones fechadas. Un archivo por serie.

El dataset tiene **tres ejes temporales** y una regla que los une:

- **Eje T (frames OLCI)**: una adquisición cada ~2 días. Es el eje de las entradas.
- **Eje T' (rondas CEMS)**: delineaciones en fechas irregulares. Es el eje de las etiquetas.
- **Eje h (horizonte)**: cuántos días hasta el objetivo. Es una **entrada** del modelo.

> **Regla de unión: una muestra por frame OLCI.** Para cada frame *f*:
> entrada = todos los frames hasta *f* + la cicatriz en *f* (producida por TMB-S3);
> target = lo nuevo quemado entre *f* y *f* + *h*.
> Es la formulación que **reproduce la operación** (se actualiza cada ~2 días) y da hasta
> ~551 candidatas (frame, serie) en vez de 359 transiciones.

```
  EMSR922__AOI01.npz

  ENTRADAS  (eje temporal T = frames OLCI)
  ------------------------------------------------------------------
   OLCI     (T,  5, H, W)  uint16   reflectancia TOA, 5 bandas, ~2 días
   S1       (T,  2, H, W)  float16  VV, VH (radar), ~6 días
   ERA5     (T,  6, H, W)  float16  viento u/v, Tmax, humedad, precipitación
   scar     (T,  H,   W)   float16  quemado en f, según TMB-S3 (fracción quemada)
   scar_ok  (T,)           bool     ¿la predicción de TMB-S3 en f es utilizable? (nubes, fallo)
   estáticas(4,  H, W)     float32  elevación, pendiente, orientación, cobertura
   horizonte (T,)          float32  h en días (entrada explícita del modelo)

  GT        (eje temporal T' = rondas CEMS)
  ------------------------------------------------------------------
   quemado  (T', H, W)      float16  fracción quemada por ronda CEMS [0..1]
   valido   (T', H, W)      uint8    255 = "nadie miró aquí"

  TARGET    (eje temporal T = frames, alineado con las muestras)
  ------------------------------------------------------------------
   delta    (T,  H, W)      float16  lo NUEVO quemado en (f, f+h]        <- TARGET
   target_ok(T,)            bool     ¿hay objetivo? (False antes de la 1ª ronda y después de la última)

  METADATOS
  ------------------------------------------------------------------
   código EMSR, AOI, fecha y sensor de cada ronda, URLs de origen
```

El GT vive en el eje de rondas y el target en el de frames; cada frame apunta a la ronda
siguiente (`r+1`) mediante los metadatos, y de ahí sale el horizonte real
`h = observed_time(r+1) − t_frame`. Los frames sin ronda objetivo (antes de la primera o
después de la última) se marcan con `target_ok = False` y no entran en el entrenamiento.

**La cicatriz de entrada la produce TMB-S3, no CEMS** (§5.7). Esto tiene tres
consecuencias:

- **Causalidad resuelta por construcción.** TMB-S3 solo ve los frames hasta *f*, así que
  su salida no puede contener quemado posterior. Ya no hace falta ninguna regla de
  `observed_time` para el canal de cicatriz: el problema desaparece.
- **La cicatriz va fresca.** Ya no está desfasada respecto al último frame, que era el
  problema de la regla causal (§7 R15).
- **CEMS pasa a ser exclusivamente la etiqueta**: el perímetro de la ronda siguiente. Ya
  no hace falta rasterizar la ronda anterior ni usarlo como referencia.

**Definición del target (decisión B3).** El target mide **lo que se quema después de *f***,
que es lo único accionable y lo único que no se ve ya en la entrada:

```
   target(f, h) = estado_quemado(f + h) − estado_quemado(f)
```

con el **mismo punto de referencia en las dos fases**:

| Fase | `estado_quemado(f)` | `estado_quemado(f + h)` | Target |
|---|---|---|---|
| Preentrenamiento (MCD64A1) | — (implícito) | píxeles con `BurnDate ∈ (f, f+h]` | exacto |
| Ajuste (CEMS) | `scar_TMB3(f)` | `polígono(r+1)` | `polígono(r+1) − scar_TMB3(f)` |

> ⚠️ **La alternativa descartada** era `polígono(r+1) − polígono(r)`, es decir, medir
> desde la ronda anterior. Eso cuenta **quemado que ya se ve en la imagen** (lo que ardió
> entre la ronda *r* y el frame *f*), que con la separación típica de CEMS (~3 días) y OLCI
> (~2 días) es **del orden de la mitad del incremento típico**: el modelo estaría
> entrenándose para predecir parte del pasado. Además, la semántica no coincidiría con el
> preentrenamiento, que mide exactamente "quemado posterior a *f*". Con la definición
> elegida, el error de TMB-S3 se cancela parcialmente entre la entrada y la etiqueta: un
> píxel que TMB-S3 da por quemado aparece en ambas y no cuenta como nuevo.

**Tres detalles finos del target, que hay que fijar para que la implementación sea
reproducible:**

1. **Puede salir negativo.** Si TMB-S3 marca como quemado un píxel que CEMS no considera
   quemado en `r+1`, la resta da negativo. Se **recorta a 0** (`clip(·, 0, 1)`) y el
   recuento de píxeles negativos se reporta: es una medida directa de la discordancia
   entre TMB-S3 y CEMS (mismo número que el F1 de la cicatriz, §6.4).
2. **El frame *f* tiene que caer dentro del intervalo de la ronda.** Si *f* es posterior a
   `observed_time(r+1)`, la etiqueta ya sería "quemado del pasado" y no valdría. En la
   práctica se exige `t_frame ≤ observed_time(r+1)` y, para que el target sea atribuible a
   la propagación, `observed_time(r) ≤ t_frame`.
3. **La referencia cambia de naturaleza entre fases.** En el preentrenamiento el estado
   quemado en *f* es "implícito" (MCD64A1 solo da la fecha de quema, no un área
   acumulada), y por eso el target es exactamente `BurnDate ∈ (f, f+h]`. En el ajuste es
   explícito (`scar_TMB3(f)`). La semántica común —"quemado **posterior** a *f*"— se
   mantiene; lo que cambia es la fuente, y por eso el ajuste a 300 m fraccional es el que
   recalibra la diferencia.

**El horizonte es una entrada explícita.** El modelo recibe *h* en días. Tres razones:

1. **Es lo que exige el producto**: para derivar el perímetro final "agregando horizontes"
   hay que poder consultar el modelo a un *h* concreto.
2. **CEMS no puede darte esa capacidad**: sus rondas son irregulares, así que sola te da
   los horizontes que el analista eligió. MCD64A1 sí: en el preentrenamiento se muestrean
   *h* de una rejilla amplia `{1, 2, 3, 5, 7, 10, 14}` días, lo que da 7 muestras por frame.
3. **Hace comparables los baselines**: B2 (advección por viento) se define naturalmente
   para un *h* dado, así que la comparación es directa.

Los horizontes reales de CEMS pueden quedar fuera de la rejilla (una ronda muy tardía);
en ese caso se acota *h* al rango entrenado y se reporta cuántos samples caen fuera.

**Arranque en frío.** Los primeros frames de un incendio —antes de que TMB-S3 detecte
nada— tienen `scar` **todo a cero**. Se **incluyen** en el entrenamiento: el modelo tiene
que saber arrancar sin cicatriz previa, que es exactamente lo que pasa en el primer ciclo
de un incendio real (§7 R16). El flag `scar_ok` marca los frames en los que la predicción
de TMB-S3 no es utilizable (frame completamente nublado, fallo de inferencia); esos se
excluyen o se tratan aparte.

### 5.2 El censo medido (Mediterráneo, 2023-2026)

| Magnitud | Valor |
|---|---|
| Activaciones de incendio en la API (global) | 131 |
| Activaciones mediterráneas | 107 (14 países: Albania, Bosnia y Herz., Bulgaria, Croacia, Francia, Grecia, Italia, Montenegro, Portugal, Serbia, Eslovenia, España, Siria, Túnez) |
| Series con ≥1 delineación | 130 |
| **Series con ≥2 delineaciones (señal de crecimiento)** | **96** |
| **Transiciones de crecimiento** | **359** |
| Productos de delineación | 489 |
| Área de AOI | min 5 km² · p25 161 · **mediana 370** · p75 949 · máx 10.034 km² |
| Superficie total | 92.210 km² |
| Píxeles a 300 m | **1.024.559** en total; mediana 4.107 por serie |
| Frames por serie | mediana 4 · media 5,7 · máx 26 |
| Observaciones (serie, frame) | 551 |
| Distribución de rondas por serie | 2→33, 3→25, 4→8, 5→6, 6→2, 7→5, 8→1, 9→6, 10→1, 11→4, 13→1, 16→1, 17→1, 18→1, 25→1 |

> **Nota sobre el conteo de frames.** El recuento "frames por serie" (mediana 4) incluye el
> producto **FEP** (primera estimación) de cada serie: por eso la mediana de *rondas* es 3 y
> la de *frames* es 4, y la media 4,74 vs 5,74. **El FEP se excluye de la serie de etiquetas**
> (D5, §5.6): es una estimación de baja resolución y meterla importaría un borde grueso.
> Excluido el FEP, la mediana de rondas por serie es 3.

Añadiendo el periodo pre-2023 (2012-2022, accesible por otra vía, §7 R1) el total
mediterráneo sube a **~162 series y ~490 transiciones**. ⚠️ Con la auditoría de splits
(§9.4.1) muchas de esas series también caen dentro del `train`/`val` de TMB-S3, así que el
saldo neto tras filtrar es **menor** que +66: hay que repetir la auditoría sobre el
periodo pre-2023 antes de contar con esas series.

### 5.3 Volumen

Con 300 m y un AOI mediano de 370 km² (4.107 píxeles):

| | Por serie | 96 series | **46 series (las que quedan)** |
|---|---|---|---|
| OLCI (T=10, 5 bandas, uint16) | ~400 KB | ~38 MB | ~18 MB |
| S1 + ERA5 + estáticas + cicatriz conocida | ~755 KB | ~72 MB | ~35 MB |
| Etiquetas + máscaras de validez | ~60 KB | ~6 MB | ~3 MB |
| **Total** | **~1,2 MB** | **~115 MB** | **~55 MB** |

El peso real lo llevan unos pocos AOIs enormes (EMSR837 tiene 10.034 km², 111.489 píxeles).

> **El dataset es pequeño en bytes y eso es una ventaja**, no un problema: cabe en un
> portátil, se entrena en minutos en una GPU de consumo y se puede iterar rápido.
### 5.4 Por qué "pequeño en bytes" no significa "pequeño para aprender"

Hay que separar tres magnitudes, que se confunden:

```
  BYTES      ~115 MB       -> irrelevante, es una ventaja
  PÍXELES    1,02 M/serie-frame  -> suficiente para aprender por píxel
  EVENTOS    96             -> ESTE es el límite real
```

Todos los parches de un mismo incendio son el mismo incendio. Para generalizar, y para que
el test set signifique algo, lo que cuenta es el número de **eventos independientes**.

Comparación: WSTS 607 eventos, TS-SatFire 179, TMB-S3 246. **96 está por debajo del
 estándar de la literatura.** Ver §9.

#### El límite real son 46, y además el reparto fijo los desperdicia

Dos cosas se suman, y conviene no confundirlas:

**1. La auditoría de splits deja 46 series utilizables** (§9.4.1): 50 de las 96 están en el
`train`/`val` de TMB-S3, que se las sabe de memoria, y su cicatriz de entrada significa
otra cosa. Quedan 46 limpias (35 activaciones).

**2. Un reparto fijo 60/20/20 sobre 46 series desperdicia el 80%.** Repartir 46 en tres
bolsillos deja 9 series como test y **37 series que nunca se evalúan**: sirven para
entrenar y no aportan ninguna medida. Con tan pocos eventos, cada serie de test vale más
que una serie de train.

→ **Se sustituye el split fijo por validación cruzada agrupada** (D14): las 46 series se
parten en **5 folds por cluster geográfico**; en cada fold entrenan ~37 y se evalúan ~9;
al final **las 46 se han evaluado exactamente una vez**.

| | Reparto fijo 60/20/20 | **CV agrupada, 5 folds** |
|---|---|---|
| Series evaluadas | 9 | **46** (cada una una vez) |
| Entrenamientos | 1 | 5 |
| Series de train por fold | 28 | ~37 |
| Precisión de la estimación | bootstrap sobre 9 | bootstrap sobre 46 |

El coste es entrenar 5 veces. El beneficio es que **el conjunto pequeño de CEMS deja de
ser una limitación y pasa a ser el diseño del experimento**: cada serie aporta su propia
medida y el bootstrap (§8) se hace sobre 46 eventos en vez de 9.
### 5.5 El recorte: una tile fija de 64×64 px (19,2 km)

**El tamaño de la ventana no se elige: lo impone el modelo.** TMB-S3 infiere con una
ventana deslizante de **64×64 px con stride 32**, y sus tiles son de **84×83 px** a 300 m
(~25 × 25 km) — verificado en el `.zarr` del dataset (`s3olci` con shape `(T, 21, 84, 83)`).

A 300 m, **64 px = 19,2 km**. Ese es el lado de la ventana que ve el encoder, y es lo que
hay que recortar para que la entrada coincida con lo que el modelo preentrenado espera.

```
  TILE DE TMB-S3      84 × 83 px  =  25,0 × 24,9 km
  VENTANA DE INFERENCIA 64 × 64 px  =  19,2 × 19,2 km   (con 2 posiciones por eje, stride 32)
  LO QUE DESCARGAMOS  64 × 64 px  =  19,2 × 19,2 km   centrada en la quemadura
```

> **Decisión de diseño (D17): la ventana es 64×64 px a 300 m, centrada en el centro de la
> quemadura, no un recorte de buffer variable.** Es lo que hace que la cicatriz de TMB-S3
> y la entrada del modelo de propagación sean **el mismo patch en las dos redes**.

**El motivo principal del recorte es el coste, no la densidad.** El AOI mediano (4.107
píxeles) cabe de sobra en 64×64, pero EMSR837 tiene 111.489 píxeles: la ventana lo recorta
a 4.096 (**×27**).

El efecto sobre la densidad sí es real, pero mucho menor de lo que parece, porque el AOI
mediano (~19 km de lado) tiene **exactamente el tamaño de la ventana**. Con la quemadura
mediana (~5,1 km de lado equivalente):

| buffer | lado de la ventana | píxeles | densidad | ganancia |
|---|---|---|---|---|
| 0 km | 5,1 km | 289 | 100% | ×14 |
| 2 km | 9,1 km | 921 | 31% | ×4,5 |
| 5 km | 15,1 km | 2.533 | 11% | ×1,6 |
| **10 km (≈ el borde de la tile de 19,2 km)** | **19,2 km** | **4.096** | **7%** | **×1,0** |
| 20 km | 39,2 km | 17.069 | 1,7% | ×0,24 (peor) |

La fila de 10 km es la que manda: es el buffer **implícito** de una tile de 19,2 km
centrada en una quemadura de 5,1 km. Es decir, **no hay elección**: o se recorta a 19,2 km
y se pierde el 93% del tile en "no quemado", o se sube la resolución (no es una opción) o
se reduce el campo de visión del modelo (lo que invalidaría el preentrenamiento de TMB-S3).

> **Conclusión.** El argumento de "×4 de densidad con buffer de 2 km" era correcto como
> cálculo de densidad, pero **no era aplicable**: una ventana de 9,1 km (30 px) no es una
> entrada válida para un encoder entrenado con ventanas de 64 px. La densidad real es del
> 7%, que es baja pero asumible: el frente nuevo son 10-100 px sobre 4.096, y la pérdida
> ponderada + el muestreo alrededor del frente (§7 R6) lo compensan. En los AOIs grandes
> (EMSR837 y compañía) el recorte sí es de un orden de magnitud.

Esto no añade eventos, pero hace el entrenamiento mucho más eficiente: el total del dataset
baja de ~115 MB (96 series) a ~55 MB (46 series), y la descarga de OLCI —el cuello de
botella real— baja en la misma proporción.
### 5.6 Reglas del dataset

Resumen operativo de lo que hay que respetar al construir las muestras. Es la
especificación que se implementa y la que se revisa al auditar una etiqueta.

| # | Regla | Por qué |
|---|---|---|
| **D1** | **La cicatriz la produce TMB-S3** (no CEMS), en las tres fases | Es un modelo: no requiere activación de emergencia, no puede filtrar el futuro y es idéntico en entrenamiento y operación (§5.7) |
| **D2** | **Cuantificado, no binario**: la etiqueta CEMS es la fracción de área quemada del píxel | La regla OR ("cualquier píxel de 10 m") infla el área un 34% (TMB-S3) |
| **D3** | **Máscara de validez por ronda**: `255` = "nadie miró aquí". El target se evalúa en `valido(r+1)` | Fuera de la máscara no hay dato, no hay "no quemado" (R4, R10) |
| **D4** | **Cambio de sensor**: la transición se conserva, pero se guarda el sensor de cada ronda y la evaluación se estratifica | Parte del "crecimiento" entre rondas es diferencia de detección, no propagación (R14) |
| **D5** | **FEP excluido** de la serie de etiquetas | Es una primera estimación de baja resolución; meterla mete un borde grueso |
| **D6** | **Arranque en frío incluido**: los frames en los que TMB-S3 aún no detecta nada (`scar` todo a cero) entran en el entrenamiento | El primer ciclo de un incendio real no tiene cicatriz (R16) |
| **D7** | **Sensor y `observed_time` de cada ronda**, siempre, en metadatos y en evaluación | Permite estratificar (R14) y auditar cualquier muestra |
| **D8** | **Split por cluster geográfico**, nunca aleatorio | Fuga espacial (R5) |
| **D9** | **El target se mide desde *f*, no desde la ronda anterior** | `polígono(r+1) − scar_TMB3(f)`, no `polígono(r+1) − polígono(r)`. Medir desde la ronda cuenta quemado ya visible (§5.1) |
| **D10** | **El horizonte *h* es una entrada del modelo** | Sin él no se puede derivar el perímetro final "agregando horizontes", ni comparar limpiamente con B2 (§5.1) |
| **D11** | **El preentrenamiento excluye los eventos que se solapan en espacio y tiempo con test y val** | Si no, la afirmación de generalización espacial no es verificable (R20) |
| **D12** | **Etiquetas de preentrenamiento a 300 m por *nearest*** | La arquitectura y la resolución no cambian entre fases; solo el target del preentrenamiento sale en bloques de 500 m |
| **D13** | **Del dataset CEMS se excluyen los 50 eventos en el `train`/`val` de TMB-S3** | En ellos la cicatriz de entrada es el perímetro CEMS memorizado, y el target significa "crecimiento entre rondas" en vez de "crecimiento después de *f*". Quedan 46 series (§9.4.1, R18) |
| **D14** | **Validación cruzada agrupada (5 folds por cluster), no split fijo** | Un reparto 60/20/20 sobre 46 series evalúa 9 y desperdicia 37. La CV evalúa **las 46**, cada una una vez (§5.4) |
| **D15** | **Del pool de preentrenamiento con MCD64A1 se excluyen los 96 eventos de CEMS, enteros** | Así el preentrenamiento y toda la evaluación CEMS son disjuntos **por construcción**. Coste: 96 eventos de miles → irrelevante (R20) |
| **D16** | **La evaluación se hace siempre contra CEMS**, nunca contra MCD64A1 ni FireCCI | CEMS lo dibujaron analistas mirando imágenes **distintas** de las del modelo. Con FireCCI S3.1.1 el modelo podría limitarse a replicar su clasificador (R21) |
| **D17** | **Ventana fija de 64×64 px a 300 m (19,2 km), centrada en la quemadura** | Es la ventana de inferencia de TMB-S3 (stride 32 sobre tiles de 84×83 px). La cicatriz y la entrada del modelo de propagación tienen que ser **el mismo patch** (§5.5) |
| **D18** | **La secuencia de entrada incluye siempre un frame pre-incendio** | TMB-S3 demuestra que el modelado temporal solo mejora con una adquisición pre-fire (§3.4, hallazgo 1). Sin ella, el modelo solo ve "el fuego sigue donde iba" |

### 5.7 Arquitectura de tres fuentes

El proyecto usa **tres fuentes de datos, cada una en un rol distinto**. La regla que las
gobierna: **cambia la supervisión, nunca las entradas**.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  ENTRADAS (idénticas en las tres fases)                         │
  │    frames OLCI (5 bandas)  ·  S1 (VV, VH)  ·  ERA5 + GFS        │
  │    DEM + WorldCover  ·  cicatriz TMB-S3  ·  horizonte h          │
  └──────────────────────────────────────────────────────────────────┘
         ▲                ▲                ▲
         │                │                │
    generación        supervisión       evaluación
         │                │                │
    TMB-S3            MCD64A1            CEMS
    (congelado)       (preentrenam.)     (ajuste + métrica)
    300 m             500 m → 300 m      300 m
```

| | Preentrenamiento | Ajuste | Inferencia |
|---|---|---|---|
| Frames OLCI | ✓ | ✓ | ✓ |
| **Cicatriz (TMB-S3)** | ✓ | ✓ | ✓ |
| Meteo, DEM, WorldCover | ✓ | ✓ | ✓ |
| Horizonte *h* | rejilla `{1,2,3,5,7,10,14}` d | el real de CEMS | el que se consulte |
| **Target** | `BurnDate ∈ (f, f+h]` (MCD64A1) | `polígono(r+1) − scar_TMB3(f)` (CEMS) | — |
| **Resolución del target** | 500 m → 300 m (*nearest*) | 300 m fraccional | — |
| **Eventos** | MCD64A1, Mediterráneo, **excluidos los 96 de CEMS** (D15) | 46 series limpias (D13) | — |

**Por qué esto resuelve el train/test shift.** El R11 original era: "se entrena con
delineaciones de CEMS pero en operación no hay CEMS". Con TMB-S3 como generador de cicatriz,
el mecanismo es **idéntico** en las tres fases: la cicatriz siempre sale de un modelo. Y como
el modelo solo ve los frames hasta *f*, no puede filtrar quemado futuro.

**Por qué MCD64A1 es un buen supervisor y no una entrada.** MCD64A1 es **inútil en
operación**: va con ~6 semanas de retraso y para un incendio que arde ahora ni siquiera
cubre el mes en curso. Precisamente por eso es un supervisor ideal: **no puede filtrar
información que en producción no tendrías**. Y es 100% reproducible con datos abiertos.

**Por qué el preentrenamiento es mediterráneo primero.** La física que se aprende (cómo
un frente avanza con viento, combustible y pendiente) se transfiere entre regímenes
parecidos, y no entre todos. Empezar por el Mediterráneo da la señal más relevante con
mucho menos esfuerzo de descarga de OLCI (que es el cuello de botella real, no la CPU).

#### El ajuste con CEMS es opcional, y esa opcionalidad es un resultado

La estructura completa tiene **dos etapas de aprendizaje encadenadas**, y se pueden
evaluar por separado:

```
  ETAPA 1  Preentrenamiento   MCD64A1  (millones de eventos, dataset completo)
            ↓  inicializa el encoder
  ETAPA 2  Ajuste             CEMS     (46 series, 5 folds agrupados)
            ↓
  EVALUACIÓN                   CEMS     (las 46 series, cada una evaluada una vez)
```

Como el ajuste es **una inicialización**, se pueden reportar **los dos modelos**:

| Modelo | Qué es | Qué demuestra |
|---|---|---|
| **M-A** | Solo preentrenamiento (MCD64A1). **Nunca ha visto CEMS** | ¿Cuánto se aprende solo con etiquetas MODIS? Transferencia pura |
| **M-B** | Preentrenamiento + ajuste con CEMS | ¿Cuánto aporta el ajuste sobre un encoder ya entrenado? |

> **Por qué esto no es "un trámite más".** La comparación M-A vs M-B sobre **el mismo
> preentrenamiento y el mismo protocolo** convierte el conjunto pequeño de CEMS de una
> limitación en el diseño del experimento. Y da un **plan de entrega con salida de
> emergencia**: si el ajuste no da tiempo, M-A ya es un resultado completo y defendible
> ("un modelo entrenado solo con etiquetas MODIS predice el crecimiento real de CEMS").

**Lo que el preentrenamiento NO puede cambiar.** El target del preentrenamiento sale en
bloques de 500 m (MCD64A1 es binario a 500 m), así que la tarea es más tosca que la final.
El ajuste con CEMS a 300 m y fraccional recalibra esa diferencia.

**El umbral espectral no está en el camino principal.** Un umbral simple sobre NBR o sobre
la caída de reflectancia en NIR podría generar una cicatriz "de emergencia" sin modelo.
Se queda como **ablación** (y como red de seguridad manual si TMB-S3 fallara), no como
generador principal: es más barato de explicar, pero su error no se puede cuantificar
contra CEMS con la misma solidez, y no tiene la cobertura temporal de TMB-S3.

---

## 6. Qué necesita el modelo en inferencia

Qué datos hay que darle al modelo ya entrenado para que haga una predicción, de dónde
salen y **qué antigüedad tienen**. Esta sección es la que convierte el proyecto de
"experimento" en producto, y es la que fija tres decisiones de diseño (§6.7).

### 6.1 Las entradas en operación y su edad

| Entrada | Fuente | Latencia (adquisición → disponible) | Revisita sobre el AOI | **Edad del dato más reciente** |
|---|---|---|---|---|
| Imagen multiespectral | Sentinel-3 OLCI **NRT** L1 | **< 3 h** | ~2 d (~3 d con nubes) | **0-2 días (media ~1 día)** |
| Radar | Sentinel-1 GRD NRT | ~1 h | ~6 d | **0-6 días (media ~3 días)** |
| Línea base pre-fuego | Sentinel-2 SR (compuesto) | ~3 h | 5 d | días o semanas antes del fuego |
| Meteo observada | GFAS | ~12 h | 6-12 h | **~12 h** |
| Meteo observada | ERA5-Land-**T** | 5 d | horario | ~5 d |
| **Pronóstico** | GFS / ECMWF HRES | ~6-12 h | horario | **a futuro** |
| Fuego activo | FIRMS / VIIRS | ~1-3 h | 12 h | **~12 h** |
| Topografía | Copernicus DEM | estático | — | — |
| Combustible | ESA WorldCover | estático | anual | — |
| **Cicatriz conocida** | ⚠️ **CEMS no sirve en operación** | — | — | ver §6.3 |

**El cuello de botella es Sentinel-3.** Su latencia de publicación es de menos de 3 horas,
así que el problema no es el retraso de la información sino la **revisita**: el producto
resultante es **de cada ~2-3 días, no diario**. Para un producto diario habría que subir
a VIIRS (12 h, 375 m), lo que refuerza la v2 (§9.1).

**Matices verificados sobre la disponibilidad:**

- Los productos **NRT de OLCI se publican en menos de 3 h** desde la adquisición; los NTC
  en más de 24 h (en la práctica 48-72 h, porque incorporate órbita precisa y datos
  meteorológicos de análisis en vez de pronóstico).
- **La ventana del archivo NRT es limitada** (del orden de un mes a un año según el nivel
  de producto). Para histórico hay que usar NTC.
- En **mayo 2026 un problema de ground segment** retards los NRT de S3A/S3B fuera de la
  timeliness nominal. Es un aviso: un pipeline operacional tiene que **monitorizar la
  latencia**, no asumirla.
- La revisita de OLCI con dos satélites es de ~2 días *ignorando nubes*; la efectiva sobre
  un AOI concreto es de ~3 días.
- **ERA5-Land consolidado va con 2-3 meses de retraso**; la versión de cerca de tiempo real
  es **ERA5-Land-T**, con 5 días de latencia. Para operación, la alternativa real es
  descargar la salida NWP en bruto (GFS/ECMWF) en lugar del producto ERA5.

### 6.2 El horizonte se cuenta desde la última observación

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
> no puede contener quemado posterior (§5.6 D1, §5.7).

### 6.3 La cicatriz conocida: por qué CEMS no sirve en operación

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

### 6.4 El modelo: un generador de cicatriz externo + una cabeza de propagación

```
   Sentinel-3  (OLCI, 300 m, ~2 dias)
   Sentinel-1  (GRD, radar, ~6 dias)
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

1. **El ajuste es perfecto**: mismo sensor (OLCI), misma resolución (300 m), mismas 5
   bandas (`Oa04, Oa06, Oa08, Oa17, Oa21`), mismas etiquetas (CEMS), misma geografía
   (Europa). El dominio gap es aproximadamente cero.
2. **Es la opción (c) de §6.3 sin el coste de entrenar (c)**: el modelo ya existe, con
   código MIT y checkpoints publicados (F1 65,5-68,2 en su propio test).
3. **Resuelve el train/test shift por completo**: el generador es idéntico en
   entrenamiento y operación. Y como solo ve los frames hasta *f*, no puede filtrar
   quemado futuro.
4. **Permite centrarse en propagación**, que es el problema abierto.

**Lo que se pierde y lo que se gana.** Se pierde la cabeza de delineación propia, y con
ella el preentrenamiento con MCD64A1 como regularizador de esa cabeza. Se gana: un
proyecto mucho más acotado, y una cicatriz de calidad **medible y citable** (el F1 de
TMB-S3 medido sobre el propio test set es una figura del informe).

#### El problema serio: TMB-S3 se sabe de memoria 50 de nuestros incendios

La auditoría (§9.4.1) encontró que **62 de las 96 series mediterráneas están en TMB-S3**,
y 50 de ellas en su `train`/`val` (los pesos las aprendieron; 12 están en su `test` y sus
pesos no las vieron).

Esto **no** significa que la respuesta esté en la entrada. Significa algo peor y más
sutil: **la cicatriz de entrada cambia de significado**.

| Serie | Qué es `scar_TMB3(f)` | Qué mide el target `polígono(r+1) − scar(f)` |
|---|---|---|
| **Memorizada** (50) | el perímetro CEMS de la ronda *r*, reconstruido casi exacto | crecimiento **entre rondas** |
| **No memorizada** (46) | una **estimación** con F1 ≈ 66 | crecimiento **después de *f*** |

Entrenar con las dos a la vez es darle al modelo **la misma pregunta con dos respuestas
distintas**. La analogía: en 50 ejemplos le das un mapa perfecto del charco y en 46 un
croquis a pulso; luego lo examinas con croquis. Aprende "el mapa siempre es perfecto" y
en el examen real falla, no porque no sepa propagar sino porque los ejemplos no se
parecían al examen.

→ **Se excluyen las 50 series memorizadas del dataset (D13).** Quedan 46, y en ellas la
cicatriz es siempre una estimación: train y test hacen la misma pregunta.

> **Corolario: rehacer el split por año no lo arregla.** Da igual cómo repartamos
> nuestras series: si el código está en el train de TMB-S3, TMB-S3 se lo sabe de memoria.
> La única solución es **sacar el evento del dataset**, no recolocarlo.

**La cicatriz de TMB-S3 tiene un sesgo conocido**: sus etiquetas usan la regla OR, que
infla el área un 34%. Como el target se define respecto a `scar_TMB3(f)` (§5.1, D9), ese
sesgo **se cancela parcialmente entre la entrada y la etiqueta**. Aun así, conviene
**cuantificarlo** sobre las propias series: el área de `scar_TMB3(f)` frente al área
fraccional de CEMS.

**Ruido de corrupción: sigue haciendo falta.** La cicatriz de TMB-S3 no es gold (F1 ~66), y
su error es *estructurado*: está en el frente, no en el interior. Durante el
entrenamiento se deforma la cicatriz de entrada: se borra un ~15% de los píxeles quemados al
azar y se dilata o erosiona el borde un píxel. Es una red de seguridad barata que hace al
modelo robusto a una cicatriz imperfecta, que es lo que verá en operación.

**Fine-tune de TMB-S3 (opcional, §9.5).** Si en algún momento se quiere mejorar la
cicatriz, se pueden afinar los pesos de TMB-S3 con las propias series y medir el delta. Es
un experimento barato y no toca el resto del pipeline.

> **Recomendación: usar TMB-S3 congelado y centrarse en la cabeza de propagación.** El
> proyecto gana foco y pierde la complicación de entrenar un segundo modelo. La dirección
> de mejora ya no es "delinear mejor", sino "predecir la propagación mejor".

### 6.5 El ciclo operativo

```
  T+0 h    Detección de hotspot (FIRMS / VIIRS / GOES)
           -> se abre un "evento"
  T+3 h    Llega OLCI NRT
           Descarga: OLCI, S1, ERA5/GFAS, GFS forecast, DEM, WorldCover
  T+3,5 h  TMB-S3 -> cicatriz actual (congelado)
           Modelo -> mapa de probabilidad de quemado a h (1, 3, 7 días)
  T+2-3 d  Llega el siguiente OLCI -> se repite
           ...
  T+n d    Dos o tres iteraciones sin crecimiento -> se cierra el evento
```

**La cadencia la marca el sensor más lento: OLCI (~2-3 días).**

### 6.6 Requisitos de infraestructura

| | |
|---|---|
| Copernicus Data Space | Cuenta gratuita obligatoria. OLCI NRT vía STAC |
| Google Earth Engine | Bien para prototipar, **mal para producción**: hay que automatizar con la API de CDSE |
| GFAS / GFS | Descarga directa desde NASA / NOAA, sin cuenta de Earthdata para GFAS |
| Almacenamiento | Nada: solo se descargan las ventanas del AOI, no los archivos completos |
| Monitorización | Hay que vigilar la latencia real de NRT (§6.1), no asumirla |
| CEMS | Solo para etiquetas. **Nunca en el camino de inferencia** |

### 6.7 Qué cambia en el diseño por todo esto

| Cambio | Por qué |
|---|---|
| **Cicatriz de entrada generada por TMB-S3 (congelado)** | Quita CEMS del camino de inferencia sin tren/test shift: en las tres fases la cicatriz sale del mismo modelo (§5.7, §6.4) |
| **Horizonte anclado en la última observación** | Si no, el modelo se entrena con horizontes que nunca tendrá en operación (§6.2) |
| **El horizonte *h* es una entrada explícita** | Necesario para derivar el perímetro final "agregando horizontes" y para comparar limpiamente con el baseline B2 (§5.1, D10) |
| **Muestra por frame OLCI**, no por ronda CEMS | Reproduce la cadencia de operación y da más muestras (§5.1) |
| **Preentrenamiento con MCD64A1 como supervisor, no como entrada** | MCD64A1 tiene ~6 semanas de latencia: sería imposible en operación, pero es un supervisor de milhões de etiquetas. Nunca entra en inferencia (§5.7) |
| **El target se mide desde *f*, no desde la ronda anterior** | `polígono(r+1) − scar_TMB3(f)`: medir desde la ronda cuenta quemado ya visible en la imagen (§5.1, D9) |
| **Ruido de corrupción sobre la cicatriz de entrada** | La cicatriz de TMB-S3 no es gold (F1 ~66) y su error está justo en el frente: hay que hacer al modelo robusto a ella (§6.4) |
| **Se excluyen las 50 series que TMB-S3 se sabe de memoria** | En ellas la cicatriz de entrada es el perímetro CEMS memorizado y el target cambia de significado entre train y test (§6.4, §9.4.1) |
| **Validación cruzada agrupada en vez de split fijo** | Con 46 series, un reparto fijo evalúa 9 y desperdicia 37 (§5.4) |
| **Dos etapas, ambas evaluables (M-A y M-B)** | El ajuste CEMS es una inicialización: reportar los dos modelos convierte el conjunto pequeño en el diseño del experimento (§5.7) |
| **La evaluación es siempre contra CEMS** | CEMS lo dibujaron analistas con otras imágenes; con FireCCI S3.1.1 el modelo podría replicar su clasificador en vez de aprender física (§3.6, R21) |
| **El input de meteo necesita viento de pronóstico**, no solo análisis | ERA5-Land es análisis del pasado; para predecir el futuro hace falta GFS/ECMWF forecast. La literatura lo hace así (FireCast usa viento; WSTS meteo + forecast) |
| **FIRMS/VIIRS como canal adicional** | Es la señal más timely que existe sobre dónde está el fuego ahora mismo. Ruidosa y a 375 m, pero aporta el ancla en tiempo real |

---

## 7. Riesgos y trampas

| # | Riesgo | Impacto | Mitigación |
|---|---|---|---|
| **R1** | **CEMS pre-2023 no está en la API.** El endpoint de detalle corta duro en EMSR656 (marzo 2023) y devuelve 403 para todo lo anterior. | Alto: sin eso, el dataset se queda en 96 series | La **página web** de cada activación es pública y no tiene ese corte, y los vectores están en un bucket S3 con URL predecible. Parser propio, ~1 día de trabajo |
| **R2** | **El 80% de las etiquetas se trazó sobre imagen comercial** (390 de 489 delineaciones). Solo 87 (18%) salieron de Sentinel-2 | Alto: el modelo podría aprender a imitar un límite visible solo a 30-50 cm | Etiquetado cuantificado; entrada radar (S1) complementaria; contraste contra MCD64A1 (reproducible al 100%) |
| **R3** | **Humo y nubes durante el incendio** | Alto: S2 y OLCI pierden señal justo cuando más importa | Sentinel-1 (radar) como ancla; el humo también es una *señal* (indica frente activo), no solo un estorbo |
| **R4** | **Ruido en el borde del frente** | Medio | Etiquetado fraccional, no OR; máscara de validez por ronda; pérdida ponderada |
| **R5** | **Fuga espacial** | Alto: invalida los resultados | Folds por cluster geográfico (DBSCAN 150 km, patrón de TMB-S3) + CV agrupada (D14) + verificación de que ningún fold tiene incendios vecinos del otro |
| **R6** | **Desbalance de clases**: el frente nuevo es ~2-7% de los píxeles | Medio | BCE ponderada / focal; muestreo alrededor del frente |
| **R7** | **Deriva orbital de MODIS**: Terra pasó de 10:30 a ~9:00 y Aqua de 13:30 a ~15:50 (sin mantenimiento de órbita desde 2020) | Medio | Marcar pre/post-2020 como dominios distintos; no usar MCD64A1 para etiquetas de años recientes |
| **R8** | **La cicatriz no es propagación** | — | Resuelto por diseño: multi-horizonte con el frente activo |
| **R9** | **Horizonte irregular**: las rondas de CEMS no están espaciadas uniformemente | Medio | Horizonte variable como entrada, o cabezas agrupadas en corto/medio/largo. No prometer "t+1d" fijo |
| **R10** | **El AOI cambia entre rondas**: la máscara de "nadie miró" es la de cada producto | Medio | Usar la máscara de la ronda correspondiente, no un AOI único |
| **R11** | **Train/test shift en la "cicatriz conocida"**: se entrena con delineaciones de CEMS pero en operación no hay CEMS | Alto | **Resuelto**: TMB-S3 congelado genera la cicatriz en las tres fases (§5.7, §6.4). El shift residual (la cicatriz nunca es gold) se cubre con ruido de corrupción y con medir el F1 de TMB-S3 sobre nuestro test (§7 R17) |
| **R12** | **El horizonte se ancla en `event_time` en vez de en la última observación** | Alto | Anclar siempre en `observed_time` / en el último frame OLCI disponible (§6.2). Verificarlo en la definición del dataset |
| **R13** | **La latencia NRT no está garantizada**: incidents de ground segment (mayo 2026) y ventana de archivo NRT limitada | Medio | Product pipeline que monitorice la latencia real y que degrade con elegancia a NTC; no asumir timings |
| **R14** | **El cambio de sensor entre rondas infla el "crecimiento" medido**: al pasar de una imagen de 10 m a una de 50 cm el analista "ve" un borde que antes no se podía resolver. Parte del incremento no es propagación, es diferencia de detección | Alto: el ruido se concentra en la banda del frente, que es justo lo que se predice | Medir el % de transiciones con sensor distinto; estratificar la evaluación por ese criterio y evaluar la métrica principal en el subconjunto "limpio" (mismo sensor o GSD comparable). Guardar el sensor de cada ronda (D4, D7) |
| **R15** | **Fuga temporal en el canal "cicatriz conocida"**: si la entrada de cicatriz proviene de una ronda o de un producto posterior al último frame, el modelo recibe parte de la respuesta | Alto: infla las métricas de forma silenciosa | **Resuelto por diseño**: la cicatriz la produce TMB-S3, que solo ve los frames hasta *f* (D1, §5.7). Comprobación: el timestamp máximo de la entrada nunca supera `t_frame` |
| **R16** | **Arranque en frío**: el primer ciclo de un incendio no tiene cicatriz, una entrada que el modelo nunca vio si todas las muestras de entrenamiento tienen fuego ya detectado | Medio: en operación, el primer ciclo es el más visible para el usuario | Incluir en el entrenamiento los frames con `scar` todo a cero (D6, §5.1). `scar_ok` marca aparte los frames en los que TMB-S3 no pudo producir una predicción válida |
| **R17** | **La cicatriz de entrada no es gold**: la de TMB-S3 tiene F1 ~66 y su error está justo en el frente, que es lo que se predice | Medio | Ruido de corrupción sobre la cicatriz de entrada (borrado aleatorio ~15% + dilatación/erosión de 1 px) y cuantificación del error de TMB-S3 contra CEMS (§6.4) |
| **R18** | **Colisión con los splits de TMB-S3**: TMB-S3 se entrenó con CEMS y con split geográfico por clusters (DBSCAN 150 km). **Medido**: 62 de las 96 series mediterráneas están en TMB-S3 (38 en `train`, 12 en `val`, 12 en `test`); la distancia mediana al evento de TMB-S3 más cercano es **16,7 km** | Alto: en las 50 series del `train`/`val` la cicatriz de entrada es el perímetro CEMS **memorizado**, así que el target cambia de significado entre train y test | **Resuelto**: D13 excluye esas 50 series (quedan 46). Rehacer el split por año **no** sirve: la colisión es con el split de TMB-S3, no con el nuestro (§9.4.1) |
| **R19** | **Desfase de detección de MCD64A1**: la fecha de quemado MODIS llega hasta 6 semanas tarde, así que un píxel quemado el día *f* puede quedar registrado en una fecha posterior. El preentrenamiento "ve" el futuro con retraso | Medio: el modelo aprende un ritmo de propagación ligeramente retrasado | Medir el desfase mediano de `BurnDate` frente a `observed_time` de CEMS sobre nuestros eventos. Se acepta si es < 1-2 días, se corrige con un shift si es mayor |
| **R20** | **Contaminación test ↔ preentrenamiento**: si el preentrenamiento con MCD64A1 incluye un incendio que luego está en test, el modelo ya ha visto esa geografía y esas fechas | Alto: contradice la crítica a Mesogeos y WildFireSpread (R5) | **Resuelto de forma simple**: D15 excluye del pool de MCD64A1 **los 96 eventos de CEMS enteros**, no solo los de test y val. Coste: 96 eventos de miles → irrelevante. Así el preentrenamiento y la evaluación son disjuntos por construcción, y no hace falta rehacer el pool en cada fold de la CV |
| **R21** | **La etiqueta puede ser reconstruible desde la entrada**: si las etiquetas las generó un algoritmo que miró las mismas imágenes que el modelo, el modelo puede aprender a **replicar ese algoritmo** en vez de aprender física de propagación | Alto: el modelo da buenos números y no significan nada | **Jerarquía de independencia de etiquetas** (§3.6): CEMS (analistas, imagen comercial + S2) es **independiente** de OLCI → es la única válida para evaluar (D16). MCD64A1 y FireCCI51 (MODIS) son parcialmente independientes → valen para preentrenar. FireCCI S3.1.1 se generó **con OLCI** → solo ablación o tercera opinión, nunca el examen |
| **R22** | **Cambio de dominio en mayo 2018**, cuando entra S3B: antes solo hay S3A (una geometría de observación), después hay dos (140° de separación en fase) | Medio: el preentrenamiento mezcla dos dominios de iluminación; y el periodo 2016-2018 tiene menos observaciones por evento | **Empezar el preentrenamiento en 2018** (S3A+S3B, dominio homogéneo) y no antes. Si se quiere usar 2016-2018, marcar el dominio explícitamente, como ya se hace con la deriva orbital de MODIS (R7) |

---

## 8. Log de decisiones

Cada decisión, la alternativa descartada y el porqué. Es la sección que hace defendible
el proyecto en una evaluación.

| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| Entrada con variables meteorológicas | Solo imágenes | Sin viento el modelo solo puede extrapolar espacialmente; falla cuando el viento cambia |
| Salida multi-horizonte | Predecir solo el perímetro final | La etiqueta por ronda da señal mucho más densa; el perímetro final es un caso particular |
| Región mediterránea | Global / EE.UU. | Global: etiquetas débiles para propagación (sin rondas). EE.UU.: ecosistema de benchmarks más rico pero régimen distinto y más fácil de replicar que de superar |
| 300 m | 10 m / 30 m / 1 km / 375 m VIIRS | 10 m no tiene dinámica; 1 km es demasiado grueso; VIIRS exigiría procesar swaths L1B |
| CEMS como fuente de etiquetas | MCD64A1 / FireCCI | CEMS es lo único que da una **serie temporal de la cicatriz**. MCD64A1 sí se usará, como contraste y preentrenamiento |
| Etiquetado cuantificado | Binario (OR lógico) | El binario en el borde es ruido irreducible; la fracción lo convierte en información |
| Horizonte variable | t+1d / t+3d / t+7d fijos | No existen etiquetas para horizontes fijos: las rondas son irregulares |
| Predecir el incremento | Predecir el estado completo | La parte ya conocida domina la pérdida y no aporta señal |
| Entradas: 5 bandas OLCI | 21 bandas | Demostrado que 5 igualan a 21 (TMB-S3); ×4 menos volumen |
| GEE como motor de imagen | STAC + rasterio directo | GEE da imágenes, clima, terreno y cobertura con 5-15 líneas cada uno. Ahorra ~2 días |
| `copernicus-ems` para CEMS | Cliente HTTP a mano | Ya resuelve activations, products, máscaras de observación, rasterización tri-estado y métricas enmascaradas |
| Split por cluster geográfico | Split temporal por año (Mesogeos) o aleatorio | El split por año de Mesogeos mide generalización temporal, no espacial: un test de 2022 puede caer junto a un train de 2019. Con 96 eventos, el aleatorio filtra el mismo incendio entre train y test |
| Baselines antes de la red | Ir directo a la arquitectura | Si no supera la advección por viento, no hay modelo |
| **Modelo multi-tarea** (delineación + propagación) | Una sola cabeza de propagación, con la cicatriz de CEMS como entrada | CEMS no existe en operación; sin la cabeza de delineación hay train/test shift y el pipeline no es operacional (§6.3). **Superado**: TMB-S3 congelado hace el papel de la cabeza de delineación sin entrenar una |
| **Horizonte anclado en la última observación** | Anclar en `event_time` o en la ronda de CEMS | Se acreditaría tiempo en el que el fuego ya estaba quemando; el modelo aprendería horizontes que no tendrá en operación (§6.2) |
| **Meteo de pronóstico** (GFS/ECMWF) además del análisis (ERA5) | Solo ERA5-Land | El análisis describe el pasado; para anticipar el frente hace falta el viento futuro (§6.7) |
| **FIRMS/VIIRS como canal de entrada** | Omitirlo, por ser grueso y ruidoso | Es la señal más timely sobre dónde está el fuego ahora mismo; aporta el ancla en tiempo real |
| **Una muestra por frame OLCI** | Una muestra por ronda CEMS / rejilla diaria | Reproduce la cadencia de operación (~551 candidatas frente a 359 transiciones) y hace que el arranque en frío aparezca solo (§5.1) |
| **Cicatriz de entrada causal** (`observed_time ≤ t_frame`) | Usar la última ronda, sea cuando sea | Una ronda posterior filtra quemado futuro al modelo y hace las métricas optimistas sin que se note (§5.6 D1, R15). **Superado** por la decisión "TMB-S3 genera la cicatriz" |
| **Sensor por ronda, evaluación estratificada** | Filtrar las transiciones con sensor distinto | Filtrar encoge un dataset que ya es pequeño; estratificar no pierde nada y el contraste es un resultado propio (§7 R14) |
| **Entrenamiento en dos fases** (delineación → propagación) + ruido de corrupción | Solo cicatriz gold, o solo ruido aleatorio | Cierra el hueco entre la entrada perfecta del entrenamiento y la imperfecta de la operación (§5.6, R17). **Superado** por "TMB-S3 genera la cicatriz": ya no hay dos fases, solo un modelo externo congelado + ruido |
| **FEP excluido** de la serie de etiquetas | Incluirlo como una ronda más | Es una primera estimación de baja resolución: meterla mete un borde grueso en las etiquetas (§5.6 D5) |
| **Métrica principal por evento**, con agregadas como segundo resultado | Solo promedio por píxel | Con 46 series evaluadas, el promedio por píxel da demasiado peso a los AOIs grandes (§8, protocolo de evaluación) |
| **TMB-S3 congelado como generador de cicatriz** | Cabeza de delineación propia; cicatriz de MCD64A1; cicatriz de CEMS; autorregresión | Mismo sensor, resolución, bandas y etiquetas que CEMS; código MIT y checkpoints publicados; elimina el train/test shift porque el mecanismo es idéntico en las tres fases (§5.7, §6.4) |
| **MCD64A1 solo como supervisor de preentrenamiento** | También como fuente de etiquetas de ajuste; como entrada del modelo | En operación no existe (6 semanas de retraso): como entrada rompería el pipeline. Como supervisor es ideal: millones de etiquetas baratas y 100% reproducible (§5.7) |
| **Target = `polígono(r+1) − scar_TMB3(f)`** | `polígono(r+1) − polígono(r)` | La alternativa cuenta quemado ya visible en la imagen (≈ la mitad del incremento típico con 3 días de separación CEMS y 2 de OLCI) y no coincide con la semántica del preentrenamiento (§5.1, D9) |
| **El horizonte *h* es una entrada explícita** | `h` implícito (solo el que toque a cada ronda); `h` fijo | Sin *h* no se puede derivar el perímetro final "agregando horizontes" ni comparar limpiamente con el baseline B2. CEMS no puede enseñar horizontes: sus rondas son irregulares (§5.1, D10) |
| **Rejilla de horizontes `{1,2,3,5,7,10,14}` días en preentrenamiento** | Muestrear `h` de la distribución empírica de CEMS | Una rejilla fija da cobertura(systemática) de los horizontes, y el modelo aprende a responder a cualquier *h*. Con CEMS sola el modelo solo conoce los horizontes que el analista eligió (§5.1) |
| **Preentrenamiento sin eventos que solapen test/val** | Preentrenar con todo MCD64A1 | MCD64A1 cubre el Mediterráneo de forma exhaustiva: casi todos los incendios de test estarían dentro. Sin la exclusión, la afirmación de generalización espacial no es verificable (§7 R20, D11) |
| **Etiquetas de preentrenamiento reescaladas con *nearest*** | Reescalado bilineal; modelo a 500 m | La arquitectura y la resolución se mantienen a 300 m en las tres fases. *Nearest* no inventa quemado donde no lo hay; el ajuste con CEMS a 300 m fraccional recalibra el bloque tosco de 500 m (§5.6 D12) |
| **Preentrenamiento mediterráneo primero** | Global; o empezar por otra región | La física se transfiere entre regímenes parecidos. Además, el cuello de botella real es la descarga de OLCI, no la CPU (§5.7) |
| **Excluir las 50 series que TMB-S3 se sabe de memoria** | Dejarlas y re-split por año; o el criterio de 150 km | En ellas la cicatriz de entrada es el perímetro CEMS memorizado y el target significa "crecimiento entre rondas" en vez de "después de *f*". El criterio de 150 km deja 2 series: inviable. Medido en §9.4.1 |
| **Validación cruzada agrupada (5 folds)** en vez de split fijo | Reparto 60/20/20 sobre las 46 | El reparto fijo evalúa 9 series y desperdicia 37. La CV evalúa las 46, cada una una vez, y el bootstrap se hace sobre 46 eventos (§5.4) |
| **Excluir los 96 eventos de CEMS del pool de MCD64A1** | Excluir solo los de test y val | Con CV, el pool cambiaría en cada fold. Excluir los 96 de una vez lo hace trivialmente correcto y cuesta nada (§5.6 D15) |
| **CEMS como único conjunto de evaluación** | FireCCI S3.1.1 (300 m, Sentinel-3) o MCD64A1 | FireCCI S3.1.1 se generó mirando OLCI: el modelo podría replicar su clasificador en vez de aprender física. CEMS lo dibujaron analistas con otras imágenes (§3.6, R21) |
| **Dos etapas encadenadas, ambas evaluables (M-A y M-B)** | Solo ajuste CEMS; o solo preentrenamiento | El ajuste sobre un preentrenamiento es una inicialización, así que se pueden reportar los dos modelos. Convierte el conjunto pequeño de CEMS en el diseño del experimento y da salida de emergencia si falta tiempo (§5.7) |

### Baselines (el termómetro del proyecto)

| # | Baseline | Idea | Por qué importa |
|---|---|---|---|
| **B0** | Persistencia | El fuego no crece | Suelo trivial |
| **B1** | Dilatación radial | El frente avanza isotropemente | Geometría pura |
| **B2** | **Advección por viento** | El frente avanza en la dirección del viento | **Física sin ML** |

> **Criterio de aceptación: el modelo debe superar B2 en F1 de área recién quemada.**
> Si una red no bate "mueve el frente según el viento", no está aprendiendo propagación:
> está aprendiendo a rellenar ruido.

### Protocolo de evaluación

Con 46 series evaluadas una vez cada una, *cómo* se promedia importa tanto como la métrica.
Fijado de antemano, para que no se pueda ajustar después:

| Elemento | Decisión |
|---|---|
| **Reparto** | **CV agrupada, 5 folds por cluster geográfico** (D14). Cada serie se evalúa exactamente una vez |
| **Métrica principal** | F1 del área **nuevamente quemada**, **promedio por evento** (un evento = una serie; dentro de la serie, promedio de sus muestras) |
| **Incertidumbre** | Bandas de confianza **bootstrap** sobre las **46** series evaluadas (1.000 remuestreos) |
| **Métrica secundaria** | F1 agregado por píxel, para poder comparar con WSTS y TS-SatFire |
| **Binarización** | Umbral **0,5 fijado de antemano**, sin calibrar (como TMB-S3) |
| **Máscara** | Solo píxeles válidos: `valido(r) ∩ valido(r+1)` (§5.6 D3) |
| **Estratificación obligatoria** | Los resultados se reportan **separados** entre transiciones con el mismo sensor y transiciones que lo cambian (§7 R14). La métrica principal se da en el subconjunto "limpio", con el número completo como segundo resultado |
| **Estrato TMB-S3** | Las 12 series que están en el `test` de TMB-S3 se reportan **aparte**: sus pesos no las entrenaron, así que miden si TMB-S3 generaliza de verdad (§9.4.1) |
| **Comparación** | Todos los baselines (B0, B1, B2) se evalúan **con el mismo protocolo y los mismos folds** que el modelo, y se reportan los dos modelos (M-A sin ajuste, M-B con ajuste) |

> **Por qué el promedio por evento y no por píxel.** Agregar por píxel hace que un solo AOI
> enorme (EMSR837 tiene 111.489 píxeles, 16 veces la mediana) domine la métrica. Con 46
> eventos evaluados, además, cualquier diferencia pequeña está dentro del ruido: sin bandas
> de confianza, "el modelo bate a B2" no es una afirmación verificable.

---

## 9. Preguntas abiertas y siguientes pasos

### 9.1 Cómo ampliar el dataset (el frente abierto principal)

| Palanca | Qué gana | Esfuerzo |
|---|---|---|
| **Scraper pre-2023** (2012-2022) | +66 series, +131 transiciones → **~162 series** | ~1 día |
| **Preentrenamiento con MCD64A1** (2016-2026, mediterráneo primero) | Millones de etiquetas baratas para el preentrenamiento; después ajuste con CEMS. **No es una entrada** (§5.7). ⚠️ El pool arranca en **2016**, no en 2000: OLCI no existe antes (§4.1) | 2-3 días |
| **FireCCI51 como supervisor a 250 m** (`ESA/CCI/FireCCI/5_1`, en GEE) | Preentrenamiento a **250 m** en vez de 500 m, con **nivel de confianza por píxel** (etiqueta blanda). Acaba en 2020-2022 | 1 día |
| **FireCCI S3.1.1** (300 m, Sentinel-3, 2019-2024) | Preentrenamiento a 300 m con fecha y confianza. **Solo como ablación o tercera opinión** (§3.6): se generó con OLCI, no sirve para evaluar (R21) | 2-3 días (CEDA, no está en GEE) |
| **Recorte a una tile fija de 64×64** | No añade eventos; recorta el coste por parche (hasta ×27 en AOIs grandes). **No es una elección**: lo impone la ventana de inferencia de TMB-S3 (§5.5, D17) | 2 horas |
| **VIIRS en vez de OLCI** | Cadencia de 12 h en lugar de 2 días | 2-3 días |
| **EMSN (Risk & Recovery)** | Severidad como tarea adicional (multi-task) | 1 día |

### 9.2 Preguntas sin resolver

- ¿El modelo generaliza a otros regímenes? (boreal, templado, australiano…)
- ¿Cuánta pérdida se puede atribuir a la discrepancia entre Sentinel-3 y las delineaciones
  de alta resolución? → el **experimento de divergencia** (§9.3) lo cuantifica.
- ¿Compensa más tener Sentinel-2 en la serie temporal que más eventos? La ablación de
  TMB-S3 sugiere que el pre-fire importa más que la frecuencia.
- ¿El horizonte debe ser entrada continua (un escalar en días) o categorizada (corto /
  medio / largo)? **Decidido: continua y explícita** (§5.1, D10). Queda por decidir si
  además se **codifica posicionalmente** (embeddings por día, como en los modelos de
  video) en lugar de concatenarse como un número normalizado.
- ¿Qué pasa si TMB-S3 deja de estar disponible o sus checkpoints cambian? La cicatriz
  generada debería **guardarse en disco** junto al dataset, para que los experimentos sean
  reproducibles.

### 9.3 Experimento de divergencia (1 día, resuelve la mayor incógnita)

```
  Para 50-100 series de CEMS:
    1. Calcular el quemado desde Sentinel-3 solo (caída de NIR + visible)
    2. Comparar con el polígono CEMS, píxel a píxel a 300 m

    Resultado:
      acuerdo > 90%  ->  Sentinel-3 y CEMS son compatibles. Etiquetado
                        cuantificado y adelante.
      acuerdo < 70%  ->  la discrepancia ES el ruido. S1 + MCD64A1 pasan a
                        ser obligatorios, y el dato "% de divergencia" se
                        convierte en un resultado del propio trabajo.
```

Es también una cifra muy valuable para el informe: "la divergencia entre delineación de
alta resolución y observación de 300 m es del X%".

### 9.4 Escalera de implementación

```
  PASO 0   Auditoría de splits (SOLO metadatos, sin descargar imagen)  <- HECHO
           Cruzar manifest.parquet de TMB-S3 con nuestros eventos
           -> 46 series limpias + CV agrupada de 5 folds
  FASE 0   Mesogeos 1 km + WildFireSpread      -> pipeline y baseline (tarea extent)
  FASE 1a  Preentrenamiento con MCD64A1 (supervisor, Mediterráneo)
           -> modelo M-A (que nunca ha visto CEMS)
  FASE 1b  Ajuste con CEMS (46 series, 5 folds)  -> modelo M-B
           + cicatriz de entrada: TMB-S3 congelado, en las tres fases
  FASE 2   S1/S2 10-20 m                        -> severidad y refinado de borde
```

> **FASE 1a y FASE 1b son independientes**: si se acaba el tiempo, FASE 1a ya da un
> resultado defendible (M-A), y FASE 1b solo lo mejora. El informe reporta los dos
> (§5.7).

### 9.4.1 Paso 0: auditoría de splits — **ejecutada** (resultados reales)

TMB-S3 se entrenó con CEMS y usó **split por cluster geográfico (DBSCAN, 150 km)**. Su
`manifest.parquet` (dataset de Hugging Face `links-ads/tmb-s3`) tiene las columnas
`bbox_id` (`EMSR168-1-0` = activación, AOI, ronda), `split`, `event_time`,
`delineation_time_pre/post` y `geometry_wkt` (el polígono de la caja, en lon/lat).

**La auditoría se ejecutó** (2026-11-07): 770 cajas, 246 activaciones, splits 517 / 135 /
118. Cruzada con las 107 activaciones mediterráneas de CEMS (71 con señal de crecimiento,
**96 series**, el mismo número que el censo de §5.2).

| Resultado | Valor |
|---|---|
| Series mediterráneas | 96 |
| **Series cuyo código EMSR aparece en TMB-S3** | **62 (65%)** |
| ├─ en el **train** de TMB-S3 (los pesos los aprendieron) | **38** |
| ├─ en el **val** de TMB-S3 (se usó para elegir checkpoint) | **12** |
| └─ en el **test** de TMB-S3 (los pesos **no** los vieron) | **12** |
| Distancia mediana al incendio de TMB-S3 más cercano | **16,7 km** (p75 47 km · p90 70 km) |
| Series a ≥150 km de cualquier evento de TMB-S3 | 6 (de ellas, **2** mediterráneas) |

**Decisión que sale de la auditoría: fuera el `train` y el `val` de TMB-S3.** Se excluyen
50 series (38 + 12). Quedan **46 series limpias**, que se reparten así:

| Subconjunto | Series | Para qué sirve |
|---|---|---|
| TMB-S3 **nunca** las vio | 34 | Conjunto principal de evaluación |
| TMB-S3 las tiene en su **test** (los pesos no las entrenaron) | 12 | **Estrato de control**: mide si TMB-S3 generaliza de verdad |

> **El criterio de 150 km se descarta como criterio de split.** Aplicado estrictamente deja
> **2 series mediterráneas** (EMSR914 a1 y a2, Albania): inviable como conjunto de
> evaluación. Se conserva como **dato del informe** (la geografía mediterránea de CEMS y
> la de TMB-S3 están solapadas), no como regla.

> **Rehacer el split por año no arregla nada.** La colisión es con el *split de TMB-S3*, no
> con el nuestro: da igual cómo repartamos nuestras series, si un código está en el train
> de TMB-S3, TMB-S3 se lo sabe de memoria. Por eso se elimina el evento, no se recoloca.

**Por qué la exclusión es necesaria y no un precaution excessive.** El problema no es que
TMB-S3 "haya visto la geografía": es que **en los 50 eventos memorizados la cicatriz de
entrada es el perímetro CENS de la ronda anterior, y en los 46 no memorizados es una
estimación con F1 ≈ 66**. Consecuencia: **el target significa cosas distintas**:

| Evento | `scar_TMB3(f)` es… | El target `polígono(r+1) − scar(f)` mide… |
|---|---|---|
| Memorizado (50) | el perímetro CEMS de la ronda *r* | crecimiento **entre rondas** |
| No memorizado (46) | una estimación del quemado en *f* | crecimiento **después de *f*** |

Si se entrena con los dos a la vez, el modelo recibe la misma pregunta con dos respuestas
distintas. La analogía: en 50 ejemplos le das un **mapa perfecto** del charco y en 46 un
**croquis**, y luego lo examinas con croquis. Aprende "el mapa siempre es perfecto".

Este paso va **antes de cualquier descarga de OLCI** porque decide qué se descarga. El
mismo script implementa D13, D14 y D15 (§5.6): excluye los 50 del dataset CEMS, construye
la CV agrupada y marca los 96 eventos que deben quedar fuera del pool de MCD64A1.

Dentro de la Fase 1 hay **dos comprobaciones que van antes de entrenar**, porque cambian el
dataset y son baratas: se hacen sobre los metadatos, sin descargar imagen.

| Comprobación | Qué mide | Por qué va primero |
|---|---|---|
| **% de transiciones con sensor distinto** | Cuánto del "crecimiento" medido es en realidad diferencia de detección | Decide si hay que filtrar transiciones y dónde se evalúa (§7 R14) |
| **Auditoría de causalidad** | Que ninguna muestra use una ronda o un producto con `observed_time > t_frame` | Si falla, **todas** las métricas están infladas y no hay nada que salvar (§7 R15) |

Si algo va mal, es mucho más barato descubrirlo antes de construir el dataset que después
de entrenar y evaluar.

### 9.5 Preguntas abiertas nuevas (nacidas de §5.6 y §5.7)

- **¿Cuánto se pierde al usar la cicatriz de TMB-S3 en vez de la de CEMS?** Se responde
  midiendo el F1 de TMB-S3 contra las delineaciones de CEMS sobre las 46 series limpias.
  Se reportan **dos cifras**: la de las 46 (más muestra, sesgada a lo conocido) y la de las
  **12 que están en el `test` de TMB-S3** (muestra pequeña pero **generalización real**:
  sus pesos no las entrenaron). Si el F1 es alto (>80%), el margen es pequeño; si es bajo,
  el modelo puede estar heredando ese error y hay que considerar el fine-tune (§6.4).
- **¿B2 (advección por viento) aguanta el mismo?** El baseline hay que evaluarlo **con el
  mismo protocolo estratificado** que el modelo (§8). Si B2 mejora en el subconjunto
  "limpio" más que el modelo, el ruido de las etiquetas estaba inflateando al modelo.
- **¿La divergencia Sentinel-3 ↔ CEMS (§9.3) es del mismo tamaño que la divergencia entre
  sensores?** Si sí, buena parte de R14 es en realidad el mismo problema de §9.3 y se
  atacan juntos.
- **¿El ajuste con CEMS (M-B) mejora sobre el solo preentrenamiento (M-A)?** Es la
  comparación central del proyecto (§5.7). Si el ajuste no aporta, se entrega M-A y el
  proyecto queda más simple. Si M-A ya bate B2, es un resultado en sí mismo.
- **¿El horizonte se comporta de forma monótona?** Si predecir a 7 días da menos quemado
  que predecir a 3, hay un problema de calibración del modelo con *h*, no de propagación.
- **¿FireCCI S3.1.1 está de acuerdo con CEMS?** Como tercera opinión: si la divergencia
  entre FireCCI y CEMS es pequeña, refuerza la confianza en las etiquetas de CEMS; si es
  grande, es el dato de ruido de etiquetas que había (§3.6, R21).

---

## 10. Cómo verificar los números de este documento

Todos los datos del §5.2 y §4.3 se obtuvieron de la API pública de CEMS el 2026-11-06.
El método, para poder repetirse:

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
  §4.3 se base en una submuestra de 24.
- El conteo de rondas por el HTML de la página legacy y por la API difieren en uno: la API
  llama `monitoringNumber` 0 al producto inicial, así que *rondas* = número de productos DEL
  y *transiciones* = rondas − 1.

Los detalles de cada fuente (endpoints, IDs en GEE, licencias) están en
[`data-sources.md`](data-sources.md).

### 10.1 Cómo verificar la auditoría de splits (§9.4.1)

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

---

## 11. Decisiones ya tomadas y lo que queda pendiente

### 11.1 Decisiones cerradas (y dónde están escritas)

| # | Decisión | Dónde |
|---|---|---|
| B1 | El canal de cicatriz lo genera **TMB-S3 congelado** en las tres fases. El umbral espectral queda como ablación | §5.7, §6.4 |
| B2 | El dataset CEMS **excluye los 50 eventos que TMB-S3 se sabe de memoria** (38 de su `train` + 12 de su `val`). Quedan **46 series** (34 que nunca vio + 12 de su `test`). Auditoría **ejecutada** | §7 R18, §9.4.1 |
| B3 | Target de ajuste = **`polígono(r+1) − scar_TMB3(f)`**, no `− polígono(r)` | §5.1, D9 |
| B4 | **MCD64A1 es el supervisor principal** del preentrenamiento. FireCCI S3.1.1 queda como ablación / tercera opinión (no como evaluación) | §5.7, §3.6 |
| B5 | Preentrenamiento **mediterráneo primero** | §5.7 |
| B6 | El preentrenamiento **excluye los 96 eventos de CEMS enteros** (no solo test/val) | §5.6 D15, §7 R20 |
| B7 | El horizonte *h* es una **entrada explícita**; rejilla `{1,2,3,5,7,10,14}` días en preentrenamiento | §5.1, D10 |
| B8 | Etiquetas de preentrenamiento a 300 m por ***nearest*** | §5.6 D12 |
| B9 | **Validación cruzada agrupada (5 folds por cluster)** en vez de split fijo: las 46 series se evalúan todas, cada una una vez | §5.4, §5.6 D14 |
| B10 | **Dos etapas encadenadas y ambas evaluables**: M-A (solo preentrenamiento) y M-B (preentrenamiento + ajuste CEMS). El ajuste es opcional y da salida de emergencia | §5.7, §9.4 |
| B11 | **La evaluación es siempre contra CEMS**, nunca contra MCD64A1 ni FireCCI (jerarquía de independencia de etiquetas) | §3.6, §5.6 D16, §7 R21 |
| B12 | **Ventana fija de 64×64 px a 300 m (19,2 km)** centrada en la quemadura | La impone la ventana de inferencia de TMB-S3 (tiles de 84×83 px, stride 32): la cicatriz y la entrada del modelo tienen que ser el mismo patch (§5.5, D17) |
| B13 | **El preentrenamiento arranca en 2018**, no en 2016 | Antes de mayo 2018 solo hay S3A: otro dominio de iluminación. Además reduce la descarga (§4.2bis, R22) |

### 11.2 Lo que queda pendiente, en orden

| # | Pendiente | Por qué bloquea | Dónde se decide |
|---|---|---|---|
| ~~**P1**~~ | ~~Ejecutar la auditoría de splits~~ | **Hecho.** 62/96 series en TMB-S3; 50 excluidas; quedan 46 | §9.4.1 |
| **P2** | **Definir B2 (advección por viento) con precisión** | Es el criterio de aceptación: "el modelo supera la física sin ML". Sin definición no se puede evaluar | §8, baselines |
| **P3** | **Fijar el alcance de la entrega**: ¿se entrega M-A, M-B o los dos? | Determina el volumen de trabajo. Con la estructura M-A/M-B, la decisión se puede aplazar hasta ver los resultados | §5.7, §9.5 |
| **P4** | **Decidir si *h* se codifica posicionalmente** (embeddings por día) o como escalar normalizado | Cambia la arquitectura del encoder | §9.2 |
| **P5** | **Medir el error de TMB-S3 contra CEMS** sobre las 46 series | Si el F1 es bajo (<80%), el margen de error del modelo se ensancha y hay que decidir si hacer fine-tune | §6.4, §9.5 |
| **P6** | **Definir qué es "el siguiente OLCI"** en el pipeline: la mejor pasada disponible frente a la de menor latencia | Afecta a la latencia del producto y a la reproducibilidad del dataset | §6.1, §6.5 |

> **Recomendación de orden.** P2 primero: es barato (una fórmula) y bloquea la evaluación.
> P3 puede esperar hasta tener resultados: la estructura M-A/M-B está pensada precisamente
> para que la decisión de alcance se tome con los números delante, no antes (§5.7).
> P1 ya está resuelto, que era lo que bloqueaba todo lo demás.

### 11.3 Lo que BLOQUEA la recopilación de datos

Lo anterior son pendientes de **evaluación**. Esto es distinto: son las decisiones sin las
cuales **no se puede empezar a descargar imagen**. Cada una dice qué se decide, por qué
bloquea y cuál es la recomendación.

| # | Decisión que falta | Por qué bloquea la descarga | Recomendación |
|---|---|---|---|
| **BL1** | **Ventana temporal de entrada**: cuántos frames hacia atrás, y cuál es el "frame pre-incendio" | Define el rango de fechas que se pide a CDSE/GEE por cada serie. Sin esto no hay consulta que hacer | **N = 6 frames** terminando en *f*, **más 1 frame pre-incendio** (la última adquisición anterior al primer quemado). Tope de 6: con la mediana de 3 rondas por serie, cubre casi todo el evento y fija la forma del tensor para poder agrupar en lotes |
| **BL2** | **Qué checkpoint de TMB-S3**: 5 bandas o 21, y 2D early-fusion o ConvLSTM | El modelo TMB-S3 que genera la cicatriz **exige un número de bandas concreto**. Si descargamos 5 y luego queremos 21, hay que volver a descargar | **ConvLSTM de 5 bandas** (`Oa04, Oa06, Oa08, Oa17, Oa21`): la ablación de TMB-S3 demuestra paridad 5↔21, la descarga baja **×4** y es la familia que casa con nuestro encoder. Si se dudas, descarga 21 y usa 5 (el disco es barato, el tiempo de descarga no) |
| **BL3** | **Alcance del pool de preentrenamiento**: recuadro geográfico y años | Es lo que multiplica la descarga de OLCI, que es el cuello de botella real | **Recuadro 30-46,5°N / 6°O-31°E** (el mismo `GEO_MED` del script de auditoría) y **2018-2026**. Ni antes de 2016 (no hay OLCI) ni más años (más dominio mezclado, R22) |
| **BL4** | **Qué cuenta como "evento" en MCD64A1**: un parche quemado, un cluster de parches, o un tile mensual | Define los "millones de muestras". Con el criterio equivocado, el pool es 10× mayor o 10× menor | **Cluster DBSCAN como TMB-S3** (ε=150 km) sobre los píxeles quemados de MCD64A1 con `BurnDate` en una ventana de 30 días. Es el mismo criterio que el paper, así que es comparable |
| **BL5** | **¿S1 (radar) entra en la v1?** | S1 son 2 canales más y **duplica la descarga** de Sentinel-1 (10 m para acabar en 300 m) | **S1 a la v2.** TMB-S3 no lo usa, es el ablated channel más caro, y su valor es marginal a 300 m. **FIRMS/VIIRS sí en la v1**: es un CSV/API, coste ~0, y aporta la señal de "dónde hay fuego ahora" |
| **BL6** | **¿Las 9 series no mediterráneas se quedan?** | Son 9 de las 46 series (EMSR894 en el norte de Francia, y lasFrench de ultramar: Nueva Caledonia, Kerguelen) | **Fuera del resultado principal.** El proyecto es del Mediterráneo; incluir Nueva Caledonia debilita la afirmación. Se pueden dejar en el dataset y reportarlas como estrato extra |
| **BL7** | **Máscara de nubes de OLCI**: ¿banderas nativas de CDSE o un proxy? | Decide si hay que descargar los flags aparte (un request más por acquisition) | **Banderas nativas de CDSE.** GEE no las trae, y son las que usa TMB-S3: sin ellas, la entrada no coincide con lo que el modelo preentrenado espera. Es un request más, no un problema |

> **Con estas siete decididas, la recopilación puede empezar.** Ninguna de las dos es
> P2-P6: esas bloquean la **evaluación**, no la **descarga**. Y si una de ellas se
> equivocara, el arreglo es barato: cambiar la regla y reejecutar la ingesta. Lo que no
> es barato es descargar dos veces 40 GB de OLCI.
