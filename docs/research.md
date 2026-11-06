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
| Entrada | Sentinel-3 OLCI (ritmo diario) + Sentinel-1 (radar) + Sentinel-2 (línea base) + ERA5 + GFS + DEM + WorldCover |
| Salida | Probabilidad de quema por píxel a horizonte variable |
| Resolución | 300 m |
| Región | Mediterráneo / UE |
| Modelo | Multi-tarea: cabeza de delineación + cabeza de propagación (§6.4) |
| En operación | Ciclo de ~2-3 días, marcado por la revisita de Sentinel-3 (§6.1) |

**El problema conocido del dataset**: 96 series con señal de crecimiento es un número bajo
para generalizar (WSTS tiene 607, TS-SatFire 179, TMB-S3 246). Es el principal frente
abierto y la razón de las dos palancas de ampliación (§9).

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
| **FireCCI S311** | Quemado + fecha | 300 m | global | 2019-2024 | Sentinel-3 Synergy + VIIRS. Bueno para preentrenamiento |
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

### 3.5 Arquitecturas de referencia

- **ConvLSTM + SegFormer** (TMB-S3): backbone espacial SegFormer, encoder temporal ConvLSTM.
  Es la opción por defecto: probada, con código MIT, y con checkpoints publicados.
- **UNet3D**: convoluciones 3D sobre la pila temporal. Más simple, peor rendimiento.
- **Vision Transformer** (WildFireSpread):Patch embedding sobre la secuencia temporal.
- **ConvNeXt-UPerNet**: backbone más potente, a cambio de más parámetros.

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

El dataset tiene **dos ejes temporales** y una regla que los une:

- **Eje T (frames OLCI)**: una adquisición cada ~2 días. Es el eje de las entradas.
- **Eje T' (rondas CEMS)**: delineaciones en fechas irregulares. Es el eje de las etiquetas.

> **Regla de unión: una muestra por frame OLCI.** Para cada frame *f*:
> entrada = todos los frames hasta *f* + la última ronda con `observed_time ≤ f`;
> target = lo nuevo quemado entre *f* y la ronda siguiente.
> Es la formulación que **reproduce la operación** (se actualiza cada ~2 días) y da hasta
> ~551 candidatas (frame, serie) en vez de 359 transiciones.

```
  EMSR922__AOI01.npz

  ENTRADAS  (eje temporal T = frames OLCI)
  ------------------------------------------------------------------
   OLCI     (T,  5, H, W)  uint16   reflectancia TOA, 5 bandas, ~2 días
   S1       (T,  2, H, W)  float16  VV, VH (radar), ~6 días
   ERA5     (T,  6, H, W)  float16  viento u/v, Tmax, humedad, precipitación
   scar     (T,  H,   W)   float16  quemado conocido en f (fracción quemada)
   scar_ok  (T,)           bool     ¿hay ronda previa? (arranque en frío)
   estáticas(4,  H, W)     float32  elevación, pendiente, orientación, cobertura

  GT        (eje temporal T' = rondas CEMS)
  ------------------------------------------------------------------
   quemado  (T', H, W)      float16  fracción quemada por ronda CEMS [0..1]
   valido   (T', H, W)      uint8    255 = "nadie miró aquí"

  TARGET    (eje temporal T = frames, alineado con las muestras)
  ------------------------------------------------------------------
   delta    (T,  H, W)      float16  lo NUEVO quemado desde f hasta la ronda siguiente <- TARGET
   target_ok(T,)            bool     ¿hay ronda objetivo? (False antes de la 1ª y después de la última)

  METADATOS
  ------------------------------------------------------------------
   código EMSR, AOI, fecha y sensor de cada ronda, URLs de origen
```

El GT vive en el eje de rondas y el target en el de frames; cada frame apunta a su ronda
anterior (`r`) y a la siguiente (`r+1`) mediante los metadatos. Los frames sin ronda
objetivo (antes de la primera o después de la última) se marcan con `target_ok = False` y
no entran en el entrenamiento.

**Regla de causalidad del canal `scar`** (hermana de R12, §6.2): la cicatriz de entrada es
la última delineación con `observed_time ≤ t_frame`. **Nunca** una ronda posterior, que
filtraría quemado futuro al modelo. Dos consecuencias, ambas deseables:

- El canal `scar` puede ir **desfasado** respecto al último frame (hasta un intervalo de
  ronda). Es la situación real de operación: lo que está quemado en el frame lo deduce el
  propio modelo a partir de la imagen, no un polígono externo.
- Las primeras muestras de una serie (antes de la primera ronda) tienen `scar` vacío y
  `scar_ok = False`. Se **incluyen** (§5.6): el modelo tiene que saber arrancar sin cicatriz.

**La contrapartida, y cómo se va a cerrar.** Con la regla causal, el target es
`ronda(r+1) − ronda(r)`: incluye quemado que ocurrió *entre* la ronda *r* y el frame *f*, que
ya estaba visible en la imagen que se le da. Es decir, **la tarea queda algo más fácil que
la real**. Cerrarlo exige saber el quemado exacto en *f*, que solo tiene la cabeza de
delineación: es el plan de dos fases de §5.6 y §6.4.

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
mediterráneo sube a **~162 series y ~490 transiciones**.

### 5.3 Volumen

Con 300 m y un AOI mediano de 370 km² (4.107 píxeles):

| | Por serie | 96 series |
|---|---|---|
| OLCI (T=10, 5 bandas, uint16) | ~400 KB | ~38 MB |
| S1 + ERA5 + estáticas + cicatriz conocida | ~755 KB | ~72 MB |
| Etiquetas + máscaras de validez | ~60 KB | ~6 MB |
| **Total** | **~1,2 MB** | **~115 MB** |

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
el test set signifique algo, lo que cuenta es el número de **eventos independientes**. Con
96 eventos, un split 60/20/20 deja ~58 train / ~19 val / ~19 test, y las métricas a nivel
de evento saldrán con **mucho ruido**.

Comparación: WSTS 607 eventos, TS-SatFire 179, TMB-S3 246. **96 está por debajo del
estándar de la literatura.** Ver §9.

### 5.5 Implicación para el recorte

El área quemada mediana (2.601 ha = 289 píxeles a 300 m) **cabe en menos de medio lado de
un cubo de 64×64**, y dentro de un AOI mediano de 4.107 píxeles solo el 7% está quemado.

→ **Recortar a una ventana fija alrededor de la quemadura** en vez de usar el AOI entero.
El motivo principal **no es la densidad de señal, es el coste**: el AOI mediano ya cabe en
64×64, pero EMSR837 tiene 111.489 píxeles y una ventana de 64×64 lo recorta a 4.096 (×27).

El efecto sobre la densidad sí es real, pero **mucho menor de lo que parece**, porque el AOI
mediano (370 km², ~19 km de lado) es más pequeño que la propia ventana del recorte. Con la
quemadura mediana (~5,1 km de lado equivalente) y un buffer *b* a cada lado:

| buffer | lado de la ventana | área | píxeles | densidad | ganancia |
|---|---|---|---|---|---|
| 0 km | 5,1 km | 26 km² | 289 | 100% | ×14 |
| 2 km | 9,1 km | 83 km² | 921 | 31% | **×4,5** |
| 5 km | 15,1 km | 228 km² | 2.533 | 11% | ×1,6 |
| 10 km | 25,1 km | 630 km² | 7.000 | 4,1% | ×0,6 (peor) |
| 20 km | 45,1 km | 2.034 km² | 22.600 | 1,3% | ×0,2 (peor) |

Con un buffer de **10-20 km el recorte es mayor que el propio AOI mediano**: no recorta nada
y la densidad **baja**. El máximo útil es ~×4 con un buffer de 2 km, y la ganancia real
depende del AOI: en los AOIs grandes sí es de un orden de magnitud.

Esto no añade eventos, pero hace el entrenamiento mucho más eficiente.

### 5.6 Reglas del dataset

Resumen operativo de lo que hay que respetar al construir las muestras. Es la
especificación que se implementa y la que se revisa al auditar una etiqueta.

| # | Regla | Por qué |
|---|---|---|
| **D1** | **Causalidad de la cicatriz**: la entrada `scar` es la última ronda con `observed_time ≤ t_frame` | Una ronda posterior filtra quemado futuro al modelo (R15) |
| **D2** | **Cuantificado, no binario**: la etiqueta es la fracción de área quemada del píxel | La regla OR ("cualquier píxel de 10 m") infla el área un 34% (TMB-S3) |
| **D3** | **Máscara de validez por ronda**: `255` = "nadie miró aquí". El target se evalúa en `valido(r) ∩ valido(r+1)` | Fuera de la máscara no hay dato, no hay "no quemado" (R4, R10) |
| **D4** | **Cambio de sensor**: la transición se conserva, pero se guarda el sensor de cada ronda y la evaluación se estratifica | Parte del "crecimiento" entre rondas es diferencia de detección, no propagación (R14) |
| **D5** | **FEP excluido** de la serie de etiquetas | Es una primera estimación de baja resolución; meterla mete un borde grueso |
| **D6** | **Arranque en frío incluido**: las muestras anteriores a la primera ronda entran, con `scar_ok = False` | El primer ciclo de un incendio real no tiene cicatriz (R16) |
| **D7** | **Sensor y `observed_time` de cada ronda**, siempre, en metadatos y en evaluación | Permite estratificar (R14) y auditar cualquier muestra |
| **D8** | **Split por cluster geográfico**, nunca aleatorio | Fuga espacial (R5) |

**Plan de entrenamiento en dos fases.** El hueco que queda abierto —que en entrenamiento la
cicatriz es gold y en operación es la salida del propio modelo— se cierra así:

```
  FASE A   Entrena SOLO la cabeza de delineación
           etiquetas: CEMS (489 delineaciones) + MCD64A1 + FireCCI
           salida: "qué está quemado en el frame f"

  FASE B   Congela (o afina despacio) la Fase A y usa SU SALIDA
           como canal `scar` de la entrada de la cabeza de propagación
```

Con la Fase B, entrenamiento y operación usan **el mismo mecanismo**, sin excepciones: la
cicatriz sale siempre de una predicción. Además, en la Fase A el modelo de delineación ve
millones de etiquetas baratas (§6.4), lo que regulariza a la cabeza de propagación, que solo
dispone de 96 series.

Mientras la Fase A no exista, se aplica **ruido de corrupción** sobre la cicatriz gold en la
Fase B: se borra un ~15% de los píxeles quemados al azar y se dilata o erosiona el borde un
píxel. Es una red de seguridad barata, no un sustituto de la Fase A.

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

> **La misma regla, aplicada al canal "cicatriz conocida".** Es el gemelo de esta: si la
> cicatriz de entrada proviene de una ronda con `observed_time` **posterior** al último frame
> OLCI, se le está enseñando al modelo parte de la respuesta. Regla D1 (§5.6): la cicatriz de
> entrada es la última ronda con `observed_time ≤ t_frame`.

### 6.3 La cicatriz conocida: por qué CEMS no sirve en operación

En el entrenamiento, la entrada "cicatriz conocida" es la última delineación de CEMS. Pero
**CEMS no existe como servicio en operación**: hay que activarlo (lo pide una autoridad
autorizada) y tarda de días a semanas en producir productos.

Si se entrena con "cicatriz de CEMS" y en operación se usa "cicatriz de otra cosa", aparece
un **train/test shift** silencioso. Tres salidas:

| Opción | Cómo | Veredicto |
|---|---|---|
| (a) Autoregresivo | La predicción del modelo en *t*+1 se usa como entrada en *t*+2 | Funciona, pero **los errores se acumulan** sin ninguna corrección externa |
| (b) Cicatriz de MCD64A1 | Fecha de quema MODIS a 500 m | Simple, pero 500 m contra 300 m y ~6 semanas de latencia |
| (c) **Cabeza de delineación** | El mismo modelo predice también *"¿dónde está quemado ahora?"* | **La correcta** |

### 6.4 El modelo multi-tarea

```
   Sentinel-3  (OLCI, 300 m, ~2 dias)
   Sentinel-1  (GRD, radar, ~6 dias)
   Sentinel-2  (linea base pre-incendio, 20 m)
   ERA5 / GFS  (viento, temperatura, humedad)
   DEM / WorldCover  (topografia, combustible)
   scar-known  (de la CABEZA 1 del ciclo anterior; de CEMS en entrenamiento)
        |
        v
   +--------------------------------------------------------------+
   |                        ENCODER                               |
   |                  ConvLSTM + SegFormer                        |
   +--------------------------------------------------------------+
        |                                         |
        v                                         v
   +--------------------------+     +---------------------------+
   | CABEZA 1: DELINEACION    |     | CABEZA 2: PROPAGACION     |
   |                          |     |                           |
   | "donde esta quemado      |     | "donde ardara en h"       |
   |  ahora"                  |     |                           |
   +--------------------------+     +---------------------------+
```

Por qué esto es claramente mejor:

1. **Hace el pipeline operacional de verdad**: no depende de una activación de emergencia.
2. **Elimina la dependencia de CEMS en inferencia**: en operación la "cicatriz conocida" la
   produce el propio modelo, no un analista.
3. **Da una señal auxiliar densa**: la cabeza de delineación se puede entrenar con
   **MCD64A1, FireCCI y CEMS** (millones de etiquetas baratas), y regulariza a la cabeza
   de propagación, que solo dispone de 96 series.
4. Es lo que hacen los dos trabajos más cercanos al problema: TMB-S3 es un modelo de
   delineación, y el preentrenamiento con MCD64A1 (§9.1) convierte la delineación en una
   tarea auxiliar natural.

> **Matiz importante (no es que "elimina el train/test shift", es que lo cambia de
> naturaleza).** Quitar CEMS del camino de inferencia resuelve el problema serio. Pero
> queda otro más pequeño: en entrenamiento la cicatriz sería *gold* (un polígono de
> analista, a 10-50 cm) y en operación es la salida del propio modelo (a 300 m, con
> errores). Si el modelo se entrena solo con la cicatriz gold, aprende a depender de una
> entrada que en producción no tendrá. Se cubre con dos mecanismos, ambos descritos en
> §5.6:
>
> - **Ruido de corrupción** sobre la cicatriz gold durante el entrenamiento (píxeles
>   quemados borrados al azar, borde dilatado o erosionado un píxel).
> - **Entrenamiento en dos fases**: la Fase A entrena la cabeza de delineación y su salida
>   es la que alimenta a la cabeza de propagación. Entrenamiento y operación usan entonces
>   **el mismo mecanismo**, sin excepciones.
>
> Con la Fase B, además, el target deja de incluir quemado ya visible en la imagen (§5.1).

> **Recomendación: que el modelo sea multi-tarea desde el principio.** No es complicación
> extra: es la diferencia entre un proyecto de investigación y un producto.

### 6.5 El ciclo operativo

```
  T+0 h    Detección de hotspot (FIRMS / VIIRS / GOES)
           -> se abre un "evento"
  T+3 h    Llega OLCI NRT
           Descarga: OLCI, S1, ERA5/GFAS, GFS forecast, DEM, WorldCover
  T+3,5 h  Modelo -> (a) delineación actual   (b) mapa de probabilidad a h
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
| **Modelo multi-tarea** (delineación + propagación) | Hace el pipeline operacional y quita CEMS del camino de inferencia (§6.4) |
| **Horizonte anclado en la última observación** | Si no, el modelo se entrena con horizontes que nunca tendrá en operación (§6.2) |
| **Cicatriz de entrada anclada en `observed_time ≤ t_frame`** | Una ronda posterior filtra quemado futuro al modelo. Es el gemelo de la regla del horizonte (§5.6 D1) |
| **Muestra por frame OLCI**, no por ronda CEMS | Reproduce la cadencia de operación y da más muestras (§5.1) |
| **Entrenamiento en dos fases** + ruido de corrupción en la cicatriz | Cierra el hueco entre la cicatriz gold (entrenamiento) y la predicha (operación) (§5.6) |
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
| **R5** | **Fuga espacial** | Alto: invalida los resultados | Split por cluster geográfico (DBSCAN 150 km, patrón de TMB-S3) + split temporal |
| **R6** | **Desbalance de clases**: el frente nuevo es ~2-7% de los píxeles | Medio | BCE ponderada / focal; muestreo alrededor del frente |
| **R7** | **Deriva orbital de MODIS**: Terra pasó de 10:30 a ~9:00 y Aqua de 13:30 a ~15:50 (sin mantenimiento de órbita desde 2020) | Medio | Marcar pre/post-2020 como dominios distintos; no usar MCD64A1 para etiquetas de años recientes |
| **R8** | **La cicatriz no es propagación** | — | Resuelto por diseño: multi-horizonte con el frente activo |
| **R9** | **Horizonte irregular**: las rondas de CEMS no están espaciadas uniformemente | Medio | Horizonte variable como entrada, o cabezas agrupadas en corto/medio/largo. No prometer "t+1d" fijo |
| **R10** | **El AOI cambia entre rondas**: la máscara de "nadie miró" es la de cada producto | Medio | Usar la máscara de la ronda correspondiente, no un AOI único |
| **R11** | **Train/test shift en la "cicatriz conocida"**: se entrena con delineaciones de CEMS pero en operación no hay CEMS | Alto | Modelo multi-tarea: la cabeza de delineación produce la cicatriz conocida (§6.3-6.4). El shift residual gold↔predicción se cubre con las dos fases y el ruido (§5.6) |
| **R12** | **El horizonte se ancla en `event_time` en vez de en la última observación** | Alto | Anclar siempre en `observed_time` / en el último frame OLCI disponible (§6.2). Verificarlo en la definición del dataset |
| **R13** | **La latencia NRT no está garantizada**: incidents de ground segment (mayo 2026) y ventana de archivo NRT limitada | Medio | Product pipeline que monitorice la latencia real y que degrade con elegancia a NTC; no asumir timings |
| **R14** | **El cambio de sensor entre rondas infla el "crecimiento" medido**: al pasar de una imagen de 10 m a una de 50 cm el analista "ve" un borde que antes no se podía resolver. Parte del incremento no es propagación, es diferencia de detección | Alto: el ruido se concentra en la banda del frente, que es justo lo que se predice | Medir el % de transiciones con sensor distinto; estratificar la evaluación por ese criterio y evaluar la métrica principal en el subconjunto "limpio" (mismo sensor o GSD comparable). Guardar el sensor de cada ronda (D4, D7) |
| **R15** | **Fuga temporal en el canal "cicatriz conocida"**: si la ronda de entrada es posterior al último frame, el modelo recibe parte de la respuesta | Alto: infla las métricas de forma silenciosa | Regla D1: la cicatriz de entrada es la última ronda con `observed_time ≤ t_frame` (§5.1, §5.6). Comprobación obligatoria en el constructor del dataset |
| **R16** | **Arranque en frío**: el primer ciclo de un incendio no tiene cicatriz previa, una entrada que el modelo nunca vio si todas las muestras de entrenamiento tienen ronda ≥2 | Medio: en operación, el primer ciclo es el más visible para el usuario | Incluir en el entrenamiento las muestras anteriores a la primera ronda, con el flag `scar_ok` (D6, §5.6) |
| **R17** | **Hueco gold ↔ predicción en la cicatriz**: en entrenamiento es un polígono perfecto de analista; en operación, la salida del propio modelo a 300 m con errores | Medio: el modelo se entrena con una entrada más limpia de la que tendrá en producción | Entrenamiento en dos fases + ruido de corrupción sobre la cicatriz gold (§5.6, §6.4) |

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
| **Modelo multi-tarea** (delineación + propagación) | Una sola cabeza de propagación, con la cicatriz de CEMS como entrada | CEMS no existe en operación; sin la cabeza de delineación hay train/test shift y el pipeline no es operacional (§6.3) |
| **Horizonte anclado en la última observación** | Anclar en `event_time` o en la ronda de CEMS | Se acreditaría tiempo en el que el fuego ya estaba quemando; el modelo aprendería horizontes que no tendrá en operación (§6.2) |
| **Meteo de pronóstico** (GFS/ECMWF) además del análisis (ERA5) | Solo ERA5-Land | El análisis describe el pasado; para anticipar el frente hace falta el viento futuro (§6.7) |
| **FIRMS/VIIRS como canal de entrada** | Omitirlo, por ser grueso y ruidoso | Es la señal más timely sobre dónde está el fuego ahora mismo; aporta el ancla en tiempo real |
| **Una muestra por frame OLCI** | Una muestra por ronda CEMS / rejilla diaria | Reproduce la cadencia de operación (~551 candidatas frente a 359 transiciones) y hace que el arranque en frío aparezca solo (§5.1) |
| **Cicatriz de entrada causal** (`observed_time ≤ t_frame`) | Usar la última ronda, sea cuando sea | Una ronda posterior filtra quemado futuro al modelo y hace las métricas optimistas sin que se note (§5.6 D1, R15) |
| **Sensor por ronda, evaluación estratificada** | Filtrar las transiciones con sensor distinto | Filtrar encoge un dataset que ya es pequeño; estratificar no pierde nada y el contraste es un resultado propio (§7 R14) |
| **Entrenamiento en dos fases** (delineación → propagación) + ruido de corrupción | Solo cicatriz gold, o solo ruido aleatorio | Cierra el hueco entre la entrada perfecta del entrenamiento y la imperfecta de la operación (§5.6, R17) |
| **FEP excluido** de la serie de etiquetas | Incluirlo como una ronda más | Es una primera estimación de baja resolución: meterla mete un borde grueso en las etiquetas (§5.6 D5) |
| **Métrica principal por evento**, con agregadas como segundo resultado | Solo promedio por píxel | Con ~19 eventos en test, el promedio por píxel da demasiado peso a los AOIs grandes (§8, protocolo de evaluación) |

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

Con ~19 eventos en test, *cómo* se promedia importa tanto como la métrica. Fijado de
antemano, para que no se pueda ajustar después:

| Elemento | Decisión |
|---|---|
| **Métrica principal** | F1 del área **nuevamente quemada**, **promedio por evento** (un evento = una serie; dentro de la serie, promedio de sus muestras) |
| **Incertidumbre** | Bandas de confianza **bootstrap** sobre los eventos del test (1.000 remuestreos) |
| **Métrica secundaria** | F1 agregado por píxel, para poder comparar con WSTS y TS-SatFire |
| **Binarización** | Umbral **0,5 fijado de antemano**, sin calibrar (como TMB-S3) |
| **Máscara** | Solo píxeles válidos: `valido(r) ∩ valido(r+1)` (§5.6 D3) |
| **Estratificación obligatoria** | Los resultados se reportan **separados** entre transiciones con el mismo sensor y transiciones que lo cambian (§7 R14). La métrica principal se da en el subconjunto "limpio", con el número completo como segundo resultado |
| **Comparación** | Todos los baselines (B0, B1, B2) se evalúan **con el mismo protocolo y el mismo split** que el modelo |

> **Por qué el promedio por evento y no por píxel.** Agregar por píxel hace que un solo AOI
> enorme (EMSR837 tiene 111.489 píxeles, 16 veces la mediana) domine la métrica. Con 19
> eventos en test, además, cualquier diferencia pequeña está dentro del ruido: sin bandas de
> confianza, "el modelo bate a B2" no es una afirmación verificable.

---

## 9. Preguntas abiertas y siguientes pasos

### 9.1 Cómo ampliar el dataset (el frente abierto principal)

| Palanca | Qué gana | Esfuerzo |
|---|---|---|
| **Scraper pre-2023** (2012-2022) | +66 series, +131 transiciones → **~162 series** | ~1 día |
| **Preentrenamiento con MCD64A1** (2000-2026, global) | Millones de transiciones para el preentrenamiento; luego afinar con CEMS | 2-3 días |
| **Recorte alrededor de la quemadura** | No añade eventos; recorta el coste por parche (hasta ×27 en AOIs grandes) y sube la densidad de señal ~×4 con buffer de 2 km | 2 horas |
| **VIIRS en vez de OLCI** | Cadencia de 12 h en lugar de 2 días | 2-3 días |
| **EMSN (Risk & Recovery)** | Severidad como tarea adicional (multi-task) | 1 día |

### 9.2 Preguntas sin resolver

- ¿El modelo generaliza a otros regímenes? (boreal, templado, australiano…)
- ¿Cuánta pérdida se puede atribuir a la discrepancia entre Sentinel-3 y las delineaciones
  de alta resolución? → el **experimento de divergencia** (§9.3) lo cuantifica.
- ¿Compensa más tener Sentinel-2 en la serie temporal que más eventos? La ablación de
  TMB-S3 sugiere que el pre-fire importa más que la frecuencia.
- ¿El horizonte debe ser entrada continua o categorizada?

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
  FASE 0   Mesogeos 1 km + WildFireSpread      -> pipeline y baseline (tarea extent)
  FASE 1   CEMS + Sentinel-3 300 m              -> propagación multi-horizonte  <- NÚCLEO
           + Fase A: cabeza de delineación (CEMS + MCD64A1 + FireCCI)
           + Fase B: propagación con la cicatriz producida por la Fase A
  FASE 2   S1/S2 10-20 m                        -> severidad y refinado de borde
```

Dentro de la Fase 1 hay **dos comprobaciones que van antes de entrenar**, porque cambian el
dataset y son baratas: se hacen sobre los metadatos, sin descargar imagen.

| Comprobación | Qué mide | Por qué va primero |
|---|---|---|
| **% de transiciones con sensor distinto** | Cuánto del "crecimiento" medido es en realidad diferencia de detección | Decide si hay que filtrar transiciones y dónde se evalúa (§7 R14) |
| **Auditoría de causalidad** | Que ninguna muestra use una ronda con `observed_time > t_frame` | Si falla, **todas** las métricas están infladas y no hay nada que salvar (§7 R15) |

Si algo va mal, es mucho más barato descubrirlo antes de construir el dataset que después
de entrenar y evaluar.

### 9.5 Preguntas abiertas nuevas (nacidas de §5.6)

- **¿Cuánto cuesta la cicatriz gold frente a la cicatriz predicha?** Se puede responder
  entrenando la Fase B con la cicatriz gold y con la de la Fase A, y comparando. Si la
  diferencia es pequeña, el ruido de corrupción (§5.6) es suficiente y las dos fases no
  hacen falta.
- **¿B2 (advección por viento) aguanta el mismo?** El baseline hay que evaluarlo **con el
  mismo protocolo estratificado** que el modelo (§8). Si B2 mejora en el subconjunto
  "limpio" más que el modelo, el ruido de las etiquetas estaba inflateando al modelo.
- **¿La divergencia Sentinel-3 ↔ CEMS (§9.3) es del mismo tamaño que la divergencia entre
  sensores?** Si sí, buena parte de R14 es en realidad el mismo problema de §9.3 y se
  atacan juntos.

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
