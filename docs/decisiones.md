# Decisiones, riesgos y preguntas abiertas

> **En una frase:** qué está decidido y por qué, qué falta por decidir, y qué riesgos se han
> visto por el camino.
>
> Todo lo que **puede cambiar**: qué está decidido y por qué, qué riesgos se han visto, qué
> falta por decidir y en qué orden. Lo que ya está cerrado y hay que implementar, en
> [`dataset.md`](dataset.md); la justificación de fondo, en
> [`investigacion.md`](investigacion.md).
>
> **Por dónde empezar:** §1.3 (lo que bloquea la descarga) → §1.2 (pendientes de
> evaluación) → §2 (riesgos).

## 1. Log de decisiones

### 1.1 Decisiones cerradas (y dónde están escritas)

| # | Decisión | Dónde |
|---|---|---|
| B1 | El canal de cicatriz lo genera **TMB-S3 congelado** en las tres fases. El umbral espectral queda como ablación | dataset.md §8, operacion.md §4 |
| B2 | El dataset CEMS **excluye los 50 eventos que TMB-S3 se sabe de memoria** (38 de su `train` + 12 de su `val`). Quedan **46 series** (34 que nunca vio + 12 de su `test`). Auditoría **ejecutada** | §2 R18, §3 |
| B3 | Target de ajuste = **`polígono(r+1) − scar_TMB3(f)`**, no `− polígono(r)` | dataset.md §2, D9 |
| B4 | **MCD64A1 es el supervisor principal** del preentrenamiento. FireCCI S3.1.1 queda como ablación / tercera opinión (no como evaluación) | dataset.md §8, investigacion.md §2 |
| B5 | Preentrenamiento **mediterráneo primero** | dataset.md §8 |
| B6 | El preentrenamiento **excluye los 96 eventos de CEMS enteros** (no solo test/val) | dataset.md §7 D15, §2 R20 |
| B7 | El horizonte *h* es una **entrada explícita**; rejilla `{1,2,3,5,7,10,14}` días en preentrenamiento | dataset.md §2, D10 |
| B8 | Etiquetas de preentrenamiento a 300 m por ***nearest*** | dataset.md §7 D12 |
| B9 | **Validación cruzada agrupada (5 folds por cluster)** en vez de split fijo: cada serie se evalúa una vez (las 46 limpias del diseño; 40 en el dataset publicado) | dataset.md §5, dataset.md §7 D14 |
| B10 | **Dos etapas encadenadas y ambas evaluables**: M-A (solo preentrenamiento) y M-B (preentrenamiento + ajuste CEMS). El ajuste es opcional y da salida de emergencia | dataset.md §8, §7 |
| B11 | **La evaluación es siempre contra CEMS**, nunca contra MCD64A1 ni FireCCI (jerarquía de independencia de etiquetas) | investigacion.md §2, dataset.md §7 D16, §2 R21 |
| B12 | **Ventana fija de 64×64 px a 300 m (19,2 km)** centrada en la quemadura | La impone la ventana de inferencia de TMB-S3 (tiles de 84×83 px, stride 32): la cicatriz y la entrada del modelo tienen que ser el mismo patch (dataset.md §6, D17) |
| B13 | **El preentrenamiento arranca en 2018**, no en 2016 | Antes de mayo 2018 solo hay S3A: otro dominio de iluminación. Además reduce la descarga (investigacion.md §3, R22) |

### 1.2 Lo que queda pendiente, en orden

| # | Pendiente | Estado / Resolución en v1 | Dónde se decide |
|---|---|---|---|
| ~~**P1**~~ | ~~Ejecutar la auditoría de splits~~ | **Hecho.** 62/96 series en TMB-S3; 50 excluidas; quedan 46 | §3 |
| ~~**P2**~~ | ~~Definir B2 (advección por viento) con precisión~~ | **Acotado a v2.** En la v1 sin canales meteorológicos en el tensor entrenado, las referencias canónicas implementadas son **B0 (persistencia)**, **B1 (dilatación radial)** y el **trivial "todo quemado"** al mismo umbral operativo; B2 queda diferido a la v2 junto con la ingesta de viento ERA5/GFS | dataset.md §9-§10, `baselines.py` |
| ~~**P3**~~ | ~~Fijar el alcance de la entrega: ¿se entrega M-A, M-B o los dos?~~ | **Hecho.** Se entregan y comparan **ambos** (`M-A` cero-disparo con MCD64A1 y `M-B` ajustado en CEMS, más variante Tversky y entrenamiento directo `prop_v4_integrity`) sobre el mismo test canónico | operacion.md §9, dataset.md §8 |
| ~~**P4**~~ | ~~Decidir si *h* se codifica posicionalmente o como escalar normalizado~~ | **Hecho.** Se codifica de forma continua mediante condicionamiento **FiLM (`HorizonModulation`)** sobre $\log(1 + h_{\text{días}})$ | `src/burn_scar/models/propagation.py` |
| ~~**P5**~~ | ~~Medir el error de TMB-S3 contra CEMS sobre las 46 series~~ | **Hecho.** Cuantificada la brecha entre la etiqueta de ensamblaje (`target`) y el crecimiento real entre rondas (`target_cems`): relación de superficie mediana $37\times$ e IoU $0,033$, publicando siempre ambas métricas por separado | operacion.md §9 |
| ~~**P6**~~ | ~~Definir qué es "el siguiente OLCI" en la cadena de procesamiento~~ | **Hecho.** Identidad canónica `(satélite, segundo UTC)`, emparejamiento de $f$ por cercanía temporal ($\le 2$ min) y filtro del $20\%$ de píxeles sin banda válida | §1.5 A1–A4, A7 |
| ~~**P7**~~ | ~~**Medir el nº real de muestras con la regla "una por frame OLCI"**~~ | **Hecho.** 411 muestras únicas $(f, h)$ sobre 44 series utilizables tras colapsar 3 duplicados CDSE (296 muestras ensambladas en 40 series con capas estáticas válidas). Manifiesto en `data/cems_manifest_46.json` | dataset.md §2, operacion.md §9 |

> **Estado al cierre de v1.** Los puntos P1–P7 quedaron cerrados o acotados explícitamente:
> la cadena completa (`R19` → `M-A` en MCD64A1 → `M-B` en CEMS → calibración Tversky y
> evaluación estratificada contra `B0`/`B1`/trivial) está implementada y verificada en
> `operacion.md` §9.

### 1.3 Decisiones para la recopilación de datos (BL1-BL7) — **Cerradas**

Decisiones cerradas para la ingesta y descarga de imagen. Cada una recoge la decisión
fijada y su motivación.

| # | Decisión | Estado | Acuerdo fijado |
|---|---|---|---|
| **BL1** | **Ventana temporal de entrada** | **Cerrada** | **$N = 6$ frames** terminando en *f*, **más 1 frame pre-incendio** (la última adquisición limpia anterior al primer quemado). Tensor $(T=7, C, 64, 64)$ |
| **BL2** | **Checkpoint de TMB-S3** | **Cerrada** | **ConvLSTM de 5 bandas** (`Oa04, Oa06, Oa08, Oa17, Oa21`) de `links-ads/tmb-s3-models`. Reduce descarga $\times 4$ preservando rendimiento |
| **BL3** | **Alcance del preentrenamiento** | **Cerrada** | Recuadro mediterráneo `GEO_MED` (**30–46,5°N / 6°O–31°E**) y periodo **2018–2026** (constelación dual S3A+S3B). Excluidos los 96 eventos CEMS (D15) |
| **BL4** | **Definición de "evento" en MCD64A1** | **Cerrada** | **Cluster DBSCAN** ($\varepsilon=150$ km, ventana 30 d) sobre píxeles quemados con recorte $64\times 64$ px centrado en el evento (patrón TMB-S3) |
| **BL5** | **Canales de entrada v1** | **Cerrada** | **S1 (radar) a la v2.** En la v1: OLCI (5 bandas) + ERA5/GFS + DEM/WorldCover + **FIRMS/VIIRS** (focos activos + FRP 48 h) |
| **BL6** | **Series no mediterráneas (8 de 44)** | **Cerrada** | **Fuera de la CV principal.** 36 series puramente mediterráneas para los 5 folds; las 8 de ultramar/atlántico quedan como test complementario fuera de dominio (fold -1) |
| **BL7** | **Máscara de nubes de OLCI** | **Cerrada** | **Banderas de calidad nativas de CDSE**. Idéntico a lo que vio TMB-S3 al entrenarse |

> **Con estas siete decisiones cerradas, la recopilación está desbloqueada.** Las tareas
> pendientes P2-P6 bloquean la **evaluación** y modelado, no la ingesta.

### 1.4 Alternativas descartadas (log de decisiones)

Cada decisión, la alternativa descartada y el porqué. Es la sección que hace defendible
el proyecto en una evaluación.

Esta tabla recoge las alternativas descartadas **que no están ya recogidas** en la decisión o
en la regla que las asume. La columna «Dónde» apunta a esa decisión o regla; el motivo de
cada una está allí (§1.1, [`dataset.md`](dataset.md) §7, `operacion.md` §3-4), así que aquí
no se repite.

| Decisión | Alternativa descartada | Motivo | Dónde |
|---|---|---|---|
| Entrada con variables meteorológicas | Solo imágenes | Sin viento el modelo solo puede extrapolar espacialmente; falla cuando el viento cambia | operacion.md §1 |
| Salida multi-horizonte | Predecir solo el perímetro final | La etiqueta por ronda da señal mucho más densa; el perímetro final es un caso particular | dataset.md §2 |
| Región mediterránea | Global / EE.UU. | Global: etiquetas débiles para propagación (sin rondas). EE.UU.: ecosistema de benchmarks más rico pero régimen distinto y más fácil de replicar que de superar | BL3, BL6 |
| 300 m | 10 m / 30 m / 1 km / 375 m VIIRS | 10 m no tiene dinámica; 1 km es demasiado grueso; VIIRS exigiría procesar swaths L1B | investigacion.md §3 |
| CEMS como fuente de etiquetas | MCD64A1 / FireCCI | CEMS es lo único que da una **serie temporal de la cicatriz**. MCD64A1 sí se usará, como contraste y preentrenamiento | B4, B11 |
| Etiquetado cuantificado | Binario (OR lógico) | El binario en el borde es ruido irreducible; la fracción lo convierte en información | D2 |
| Horizonte variable | t+1d / t+3d / t+7d fijos | No existen etiquetas para horizontes fijos: las rondas son irregulares | B7 |
| Entradas: 5 bandas OLCI | 21 bandas | Demostrado que 5 igualan a 21 (TMB-S3); ×4 menos volumen | BL2 |
| GEE como motor de imagen | STAC + rasterio directo | GEE da imágenes, clima, terreno y cobertura con 5-15 líneas cada uno. Ahorra ~2 días | operacion.md §6 |
| `copernicus-ems` para CEMS | Cliente HTTP a mano | Ya resuelve activations, products, máscaras de observación, rasterización tri-estado y métricas enmascaradas | datos.md §2.7 |
| Split por cluster geográfico | Split temporal por año (Mesogeos) o aleatorio | El split por año de Mesogeos mide generalización temporal, no espacial: un test de 2022 puede caer junto a un train de 2019. Con 96 eventos, el aleatorio filtra el mismo incendio entre train y test | B9, D8 |
| Baselines antes de la red | Ir directo a la arquitectura | Si no supera la advección por viento, no hay modelo | dataset.md §9 |
| Horizonte anclado en la última observación | Anclar en `event_time` o en la ronda de CEMS | Se acreditaría tiempo en el que el fuego ya estaba quemando; el modelo aprendería horizontes que no tendrá en operación | operacion.md §2 |
| Meteo de pronóstico (GFS/ECMWF) además del análisis (ERA5) | Solo ERA5-Land | El análisis describe el pasado; para anticipar el frente hace falta el viento futuro | operacion.md §7 |
| FIRMS/VIIRS como canal de entrada | Omitirlo, por ser grueso y ruidoso | Es la señal más timely sobre dónde está el fuego ahora mismo; aporta el ancla en tiempo real | BL5 |
| Una muestra por frame OLCI | Una muestra por ronda CEMS / rejilla diaria | Reproduce la cadencia de operación (actualiza cada ~2-3 días) y hace que el arranque en frío aparezca solo. El nº de muestras con esta regla está sin medir (P7); lo que sí es inmediato es que 455 productos DEL ↔ 359 transiciones | dataset.md §2 |
| Sensor por ronda, evaluación estratificada | Filtrar las transiciones con sensor distinto | Filtrar encoge un dataset que ya es pequeño; estratificar no pierde nada y el contraste es un resultado propio | D4, R14 |
| Métrica principal por evento, con agregadas como segundo resultado | Solo promedio por píxel | Con 46 series evaluadas, el promedio por píxel da demasiado peso a los AOIs grandes | dataset.md §10 |

### 1.5 Decisiones sobre el ensamblaje del dataset (auditoría de specs de 2026-11-15)

Decisiones tomadas al alinear el ensamblador con el spec de `propagation-dataset`. Incluyen
correcciones de cumplimiento y controles de integridad que hacen auditable el dataset.

| # | Decisión | Por qué | Alternativa descartada |
|---|---|---|---|
| **A1** | La consulta de imágenes usa la **extensión del AOI** (el `bbox_wkt` del manifiesto), no el punto del centroide | El downloader filtraba por un punto y el catálogo de referencia por el polígono. OLCI va en tiles fijos y S3A y S3B tienen geometría de órbita distinta: el centroide cae en un hueco de tile de S3B y sus escenas nunca se descargaban. De 128 adquisiciones ausentes, **125 eran de S3B** | Subir solo el límite de resultados de la consulta: el tope de 60 imágenes explica **una** de las 128; el hueco de S3B aparece igual en series que nunca lo alcanzan |
| **A2** | El frame `f` se empareja con su adquisición por **cercanía temporal** (tolerancia de 2 minutos), no por comparación de cadenas | El manifiesto trunca el tiempo a segundos y el fichero se nombra con el tiempo redondeado: un desfase de 1 s dejaba la adquisición de `f` fuera de su propia ventana en **251 de 390 muestras** | Alinear el nombre del fichero con el manifiesto: ata el ensamblador al descargador. Emparejar por `scene_id`: CDSE y GEE no comparten identificadores de producto |
| **A3** | La ventana temporal **termina en `f`** y se descarta la muestra si no hay adquisición utilizable de `f` | El código usaba la última adquisición anterior a `f` y la ponía en el sitio de `f`: la cicatriz y la etiqueta quedaban ancladas a un instante distinto del que define el horizonte | Marcar la muestra (`frame_matched: false`) y propagar el filtro: deja un subconjunto no conforme dentro del dataset de producción |
| **A4** | El descarte de frames sin dato se mide sobre **píxeles sin ninguna banda válida**, no sobre celdas de banda | El spec lo dice de forma explícita y la medida por celda ataba el criterio al número de bandas (5) y trataba igual un píxel con 4 bandas válidas que uno completamente muerto. Afecta a **1 adquisición** del inventario actual | Cambiar el spec a medida por celda: se perdería la semántica de D3 ("nadie miró aquí" es del píxel, no de cada banda) |
| **A6** | Earth Engine se accede con **cuenta de servicio** y el proyecto se registra para EE | La máquina no tiene navegador, así que el flujo OAuth no funciona. La cuenta de servicio necesita **dos roles**: `roles/earthengine.resourceAdmin` y `roles/serviceusage.serviceUsageConsumer`; con solo el primero, `ee.Initialize()` va bien pero la consulta de la colección falla. El proyecto además tiene que estar registrado (paso de navegador) | Credenciales personales (`earthengine authenticate`): no funciona sin navegador. Solo `roles/earthengine.user`: no incluye `serviceusage.services.use` |
| **A5** | El dataset registra la **versión del generador** (huella del contenido de sus ficheros) y el cargador avisa si no coincide | Sin esto, un dataset producido por un ensamblador anterior era indistinguible del actual: el caso real que motivó la auditoría (el dataset por defecto almacenaba `mask_cems` con `0` en vez de `255`) pasó inadvertido porque el loader normalizaba al leer | Número de versión mantenido a mano: se olvida subir. Hash de commit: no captura un árbol de trabajo sucio, que es el estado normal mientras se desarrolla |
| **A7** | La identidad canónica de una adquisición OLCI es `(satélite, segundo UTC)` | CDSE devolvió dos productos para la misma adquisición en tres instantes; al truncarse el tiempo al segundo competían por la misma ruta y el índice contenía 344 entradas para 341 ficheros. La selección del representante es determinista y conserva todos los IDs | Incluir `scene_id` o microsegundos en el nombre: obligaría a rehacer el emparejamiento entre catálogos sin mejorar el raster. Deduplicar solo al cargar: ocultaría el defecto del artefacto almacenado |
| **A8** | DEM y WorldCover se validan antes de ensamblar y una serie inválida se excluye con motivo | `EMSR785_AOI01` tenía WorldCover completamente vacío y produjo 47 entradas de test sin ninguna señal de ausencia de dato. Una capa no válida no puede convertirse en una entrada válida mediante relleno | Aceptar ceros: el modelo no distingue una clase WorldCover válida de la ausencia de dato. Eliminar la serie sin registrar el motivo: rompería la trazabilidad del conjunto |
| **A9** | El arranque en frío se documenta, no se fabrica | Ninguna muestra del dataset vigente tiene `x_scar` completamente vacío. Forzar una cicatriz a cero cambiaría a la vez la entrada y la relación semántica del target | Incluir una copia sintética con `x_scar=0`: produciría un desempeño no defendible |
| **A10** | El horizonte permanece continuo y sin máximo fijo; las bandas son solo diagnóstico | Cada muestra aporta información al aprendizaje compartido y `h` es una entrada explícita del modelo. Filtrar por banda reduciría los datos sin aportar una etiqueta nueva | Limitar el entrenamiento a una banda: descartaría muestras sin aportar una etiqueta nueva. Mantener un máximo fijo en el dataset: quedaría obsoleto al crecer los datos. Limitar a un día: no mejora el soporte y reduce el conjunto |
| **A11** | El índice tiene **versión de esquema** y un bloque `inventory` que reconcilia entradas, muestras únicas, ficheros únicos, series, folds, horizontes, cobertura de cicatriz y series excluidas con motivo | El índice es el contrato entre ensamblaje, cargador, evaluación y cualquier consumidor externo. Un consumidor que adivine los recuentos acaba contando lo que quiere | Un contador aparte por script: cada script informaría de algo distinto. Publicar solo el total: ocultaría la diferencia entre entradas y muestras únicas, que era exactamente el defecto |
| **A12** | La evaluación **separa los denominadores** (entradas, muestras únicas, ficheros únicos, series, muestras evaluables) y usa un único descriptivo de conjunto evaluado para el modelo y los baselines | Comparar F1 de dos evaluadores sobre conjuntos distintos no significa nada; el fallo anterior venía de consumir el índice con duplicados | Un único "n muestras": mezcla niveles distintos y oculta el subconjunto evaluable, que cambia con la etiqueta y con el umbral |
| **A13** | La procedencia es **obligatoria** en cada artefacto: huella del generador, huella del índice, huella de contenido estable, semilla, esquema y huellas de los checkpoints | Un resumen sin la huella del dataset no puede relacionarse con los datos que lo originaron, y el training se vuelve irreproducible | Registrar solo la fecha de generación: dos datasets con la misma fecha pueden ser distintos. Confiar en el commit: no captura un árbol de trabajo sucio |

## 2. Riesgos y trampas
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
| **R11** | **Train/test shift en la "cicatriz conocida"**: se entrena con delineaciones de CEMS pero en operación no hay CEMS | Alto | **Resuelto**: TMB-S3 congelado genera la cicatriz en las tres fases (dataset.md §8, operacion.md §4). El shift residual (la cicatriz nunca es gold) se cubre con ruido de corrupción y con medir el F1 de TMB-S3 sobre nuestro test (§2 R17) |
| **R12** | **El horizonte se ancla en `event_time` en vez de en la última observación** | Alto | Anclar siempre en `observed_time` / en el último frame OLCI disponible (operacion.md §2). Verificarlo en la definición del dataset |
| **R13** | **La latencia NRT no está garantizada**: incidencias del segmento de tierra (mayo 2026) y ventana de archivo NRT limitada | Medio | Product pipeline que monitorice la latencia real y que degrade con elegancia a NTC; no asumir timings |
| **R14** | **El cambio de sensor entre rondas infla el "crecimiento" medido**: al pasar de una imagen de 10 m a una de 50 cm el analista "ve" un borde que antes no se podía resolver. Parte del incremento no es propagación, es diferencia de detección | Alto: el ruido se concentra en la banda del frente, que es justo lo que se predice | Medir el % de transiciones con sensor distinto (**medido: 116 de 188**); estratificar la evaluación por ese criterio y evaluar la métrica principal en el subconjunto "limpio" (mismo sensor o GSD comparable). Guardar el sensor de cada ronda (D4, D7). **Pendiente**: los datos están en el índice, la estratificación todavía no se publica (P8, dataset.md §10) |
| **R15** | **Fuga temporal en el canal "cicatriz conocida"**: si la entrada de cicatriz proviene de una ronda o de un producto posterior al último frame, el modelo recibe parte de la respuesta | Alto: infla las métricas de forma silenciosa | **Resuelto por diseño**: la cicatriz la produce TMB-S3, que solo ve los frames hasta *f* (D1, dataset.md §8). Comprobación: el timestamp máximo de la entrada nunca supera `t_frame` |
| **R16** | **Arranque en frío**: el primer ciclo de un incendio no tiene cicatriz, una entrada que el modelo nunca vio si todas las muestras de entrenamiento tienen fuego ya detectado | Medio: en operación, el primer ciclo es el más visible para el usuario | Incluir en el entrenamiento los frames con `scar` todo a cero (D6, dataset.md §2). El índice marca aparte el estado de la cicatriz (`scar_state`) para poder separarlos |
| **R17** | **La cicatriz de entrada no es gold**: la de TMB-S3 tiene F1 ~66 y su error está justo en el frente, que es lo que se predice | Medio | Ruido de corrupción sobre la cicatriz de entrada (borrado aleatorio ~15% + dilatación/erosión de 1 px) y cuantificación del error de TMB-S3 contra CEMS (operacion.md §4) |
| **R18** | **Colisión con los splits de TMB-S3**: TMB-S3 se entrenó con CEMS y con split geográfico por clusters (DBSCAN 150 km). **Medido**: 62 de las 96 series mediterráneas están en TMB-S3 (38 en `train`, 12 en `val`, 12 en `test`); la distancia mediana al evento de TMB-S3 más cercano es **16,7 km** | Alto: en las 50 series del `train`/`val` la cicatriz de entrada es el perímetro CEMS **memorizado**, así que el target cambia de significado entre train y test | **Resuelto**: D13 excluye esas 50 series (quedan 46). Rehacer el split por año **no** sirve: la colisión es con el split de TMB-S3, no con el nuestro (§3) |
| **R19** | **Desfase de detección de MCD64A1**: la fecha de quemado MODIS llega hasta 6 semanas tarde, así que un píxel quemado el día *f* puede quedar registrado en una fecha posterior. El preentrenamiento "ve" el futuro con retraso | Medio: el modelo aprende un ritmo de propagación ligeramente retrasado | Medir el desfase mediano de `BurnDate` frente a `observed_time` de CEMS sobre nuestros eventos. Se acepta si es < 1-2 días, se corrige con un shift si es mayor |
| **R20** | **Contaminación test ↔ preentrenamiento**: si el preentrenamiento con MCD64A1 incluye un incendio que luego está en test, el modelo ya ha visto esa geografía y esas fechas | Alto: contradice la crítica a Mesogeos y WildFireSpread (R5) | **Resuelto de forma simple**: D15 excluye del pool de MCD64A1 **los 96 eventos de CEMS enteros**, no solo los de test y val. Coste: 96 eventos de miles → irrelevante. Así el preentrenamiento y la evaluación son disjuntos por construcción, y no hace falta rehacer el pool en cada fold de la CV |
| **R21** | **La etiqueta puede ser reconstruible desde la entrada**: si las etiquetas las generó un algoritmo que miró las mismas imágenes que el modelo, el modelo puede aprender a **replicar ese algoritmo** en vez de aprender física de propagación | Alto: el modelo da buenos números y no significan nada | **Jerarquía de independencia de etiquetas** (investigacion.md §2): CEMS (analistas, imagen comercial + S2) es **independiente** de OLCI → es la única válida para evaluar (D16). MCD64A1 y FireCCI51 (MODIS) son parcialmente independientes → valen para preentrenar. FireCCI S3.1.1 se generó **con OLCI** → solo ablación o tercera opinión, nunca el examen |
| **R22** | **Cambio de dominio en mayo 2018**, cuando entra S3B: antes solo hay S3A (una geometría de observación), después hay dos (140° de separación en fase) | Medio: el preentrenamiento mezcla dos dominios de iluminación; y el periodo 2016-2018 tiene menos observaciones por evento | **Empezar el preentrenamiento en 2018** (S3A+S3B, dominio homogéneo) y no antes. Si se quiere usar 2016-2018, marcar el dominio explícitamente, como ya se hace con la deriva orbital de MODIS (R7) |

---

## 3. Paso 0: auditoría de splits — **ejecutada** (resultados reales)

TMB-S3 se entrenó con CEMS y usó **split por cluster geográfico (DBSCAN, 150 km)**. Su
`manifest.parquet` (dataset de Hugging Face `links-ads/tmb-s3`) tiene las columnas
`bbox_id` (`EMSR168-1-0` = activación, AOI, ronda), `split`, `event_time`,
`delineation_time_pre/post` y `geometry_wkt` (el polígono de la caja, en lon/lat).

**La auditoría se ejecutó** (2026-11-07): 770 cajas, 246 activaciones, splits 517 / 135 /
118. Cruzada con las 107 activaciones mediterráneas de CEMS (71 con señal de crecimiento,
**96 series**, el mismo número que el censo de dataset.md §3).

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

> **Del censo de 46 series al dataset publicado.** Las 46 limpias son el punto de partida;
> el manifiesto construido sobre ellas (`data/cems_manifest_46.json`) se queda en **44
> series** —2 se caen al cruzarlas con la API de CEMS (sin detalle, sin AOI o con menos de
> 2 rondas DEL fechadas)— repartidas en **36 mediterráneas y 8 de ultramar/atlántico**, de
> las cuales **10** conservan la etiqueta de control del `test` de TMB-S3. El ensamblador
> publica **40 series** (33 en la CV y 7 fuera de dominio): 4 series del manifiesto no
> dejan ninguna muestra válida, y una de ellas (`EMSR785_AOI01`) queda registrada con su
> motivo de exclusión. Las cifras canónicas están en [`operacion.md`](operacion.md) §9.

> **El criterio de 150 km se descarta como criterio de split.** Aplicado estrictamente deja
> **2 series mediterráneas** (EMSR914 a1 y a2, Albania): inviable como conjunto de
> evaluación. Se conserva como **dato del informe** (la geografía mediterránea de CEMS y
> la de TMB-S3 están solapadas), no como regla.

> **Rehacer el split por año no arregla nada.** La colisión es con el *split de TMB-S3*, no
> con el nuestro: da igual cómo repartamos nuestras series, si un código está en el train
> de TMB-S3, TMB-S3 se lo sabe de memoria. Por eso se elimina el evento, no se recoloca.

**Por qué la exclusión es necesaria y no es una precaución excesiva.** El problema no es que
TMB-S3 "haya visto la geografía": es que **en los 50 eventos memorizados la cicatriz de
entrada es el perímetro CEMS de la ronda anterior, y en los 46 no memorizados es una
estimación con F1 ≈ 66**. Consecuencia: **el target significa cosas distintas**:

| Evento | `scar_TMB3(f)` es… | El target `polígono(r+1) − scar(f)` mide… |
|---|---|---|
| Memorizado (50) | el perímetro CEMS de la ronda *r* | crecimiento **entre rondas** |
| No memorizado (46) | una estimación del quemado en *f* | crecimiento **después de *f*** |

Si se entrena con los dos a la vez, el modelo recibe la misma pregunta con dos respuestas
distintas. La analogía: en 50 ejemplos le das un **mapa perfecto** del charco y en 46 un
**croquis**, y luego lo examinas con croquis. Aprende "el mapa siempre es perfecto".

Este paso va **antes de cualquier descarga de OLCI** porque decide qué se descarga. El
script implementa **D13** (excluye las 50 series del `train`/`val` de TMB-S3) y aporta los
datos que necesitan **D14** y **D15**: los 46 eventos utilizables para la CV agrupada y los
96 que deben quedar fuera del pool de MCD64A1. La construcción de los folds y el marcado
del pool quedan por escribir.

Dentro de la Fase 1 hay **dos comprobaciones que van antes de entrenar**, porque cambian el
dataset y son baratas: se hacen sobre los metadatos, sin descargar imagen.

| Comprobación | Qué mide / Resultado | Por qué va primero |
|---|---|---|
| **% de transiciones con sensor distinto** | **Medido: 116 de 188 transiciones (61,7%) cambian de sensor** (Legion, Sentinel-2, SPOT, WorldView). | Justifica plenamente estratificar la evaluación (§2 R14, D4) para no confundir diferencia de detección con propagación |
| **Auditoría de causalidad** | Que ninguna muestra use una ronda o un producto con `observed_time > t_frame` | Si falla, **todas** las métricas están infladas y no hay nada que salvar (§2 R15) |

Si algo va mal, es mucho más barato descubrirlo antes de construir el dataset que después
de entrenar y evaluar.

## 4. Cómo ampliar el dataset (el frente abierto principal)

| Palanca | Qué gana | Esfuerzo |
|---|---|---|
| **Scraper pre-2023** (2012-2022) | +66 series, +131 transiciones → **~162 series** | ~1 día |
| **Preentrenamiento con MCD64A1** (2018-2026, mediterráneo primero) | Millones de etiquetas baratas para el preentrenamiento; después ajuste con CEMS. **No es una entrada** (dataset.md §8). ⚠️ El pool arranca en **2018**: sin OLCI no hay entrada antes de 2016 (investigacion.md §3) y 2016-2018 mezcla el dominio de S3A (§2 R22). **Implementado** (operacion.md §10); por ahora **no aporta** sobre entrenar desde cero (operacion.md §9) | 2-3 días |
| **FireCCI51 como supervisor a 250 m** (`ESA/CCI/FireCCI/5_1`, en GEE) | Preentrenamiento a **250 m** en vez de 500 m, con **nivel de confianza por píxel** (etiqueta blanda). Acaba en 2020-2022 | 1 día |
| **FireCCI S3.1.1** (300 m, Sentinel-3, 2019-2024) | Preentrenamiento a 300 m con fecha y confianza. **Solo como ablación o tercera opinión** (investigacion.md §2): se generó con OLCI, no sirve para evaluar (R21) | 2-3 días (CEDA, no está en GEE) |
| **VIIRS en vez de OLCI** | Cadencia de 12 h en lugar de 2 días | 2-3 días |
| **EMSN (Risk & Recovery)** | Severidad como tarea adicional (multi-task) | 1 día |

## 5. Preguntas abiertas

Las primeras son de alcance general; las que siguen nacen de las reglas D1-D22 y de la
arquitectura de tres fuentes.

- ¿El modelo generaliza a otros regímenes? (boreal, templado, australiano…)
- ¿Cuánta pérdida se puede atribuir a la discrepancia entre Sentinel-3 y las delineaciones
  de alta resolución? → el **experimento de divergencia** (§6) lo cuantifica.
- ¿Compensa más tener Sentinel-2 en la serie temporal que más eventos? La ablación de
  TMB-S3 sugiere que el pre-fire importa más que la frecuencia.
- ¿Qué pasa si TMB-S3 deja de estar disponible o sus checkpoints cambian? La cicatriz
  generada debería **guardarse en disco** junto al dataset, para que los experimentos sean
  reproducibles.
- **¿Cuánto se pierde al usar la cicatriz de TMB-S3 en vez de la de CEMS?** Se responde
  midiendo el F1 de TMB-S3 contra las delineaciones de CEMS sobre las 46 series limpias.
  Se reportan **dos cifras**: la de las 46 (más muestra, sesgada a lo conocido) y la de las
  **12 que están en el `test` de TMB-S3** (muestra pequeña pero **generalización real**:
  sus pesos no las entrenaron). Si el F1 es alto (>80%), el margen es pequeño; si es bajo,
  el modelo puede estar heredando ese error y hay que considerar el fine-tune (operacion.md §4).
- **¿B2 (advección por viento) aguanta el mismo?** El baseline hay que evaluarlo **con el
  mismo protocolo estratificado** que el modelo (dataset.md §10). Si B2 mejora en el subconjunto
  "limpio" más que el modelo, el ruido de las etiquetas estaba inflateando al modelo.
- **¿La divergencia Sentinel-3 ↔ CEMS (§6) es del mismo tamaño que la divergencia entre
  sensores?** Si sí, buena parte de R14 es en realidad el mismo problema de §6 y se
  atacan juntos.
- **¿El ajuste con CEMS (M-B) mejora sobre el solo preentrenamiento (M-A)?** Es la
  comparación central del proyecto (dataset.md §8). Si el ajuste no aporta, se entrega M-A y el
  proyecto queda más simple. Si M-A ya bate B2, es un resultado en sí mismo.
  → **Respuesta provisional (2026-11-16): todavía no.** Medido sobre el mismo test (25
  muestras, 7 series, umbral 0,10), M-B da 12,26% de F1 contra la etiqueta de referencia y
  **14,80%** el entrenamiento desde cero; M-A en cero-disparo se queda en 10,61%. Con ese
  tamaño de muestra **ninguna de las diferencias es significante**, pero hoy **no hay ni un
  indicio a favor del preentrenamiento**, y el umbral del M-B es aún menos estable
  (dispersión 0,50). Detalle y reservas en [`operacion.md`](operacion.md) §9.
- **¿El horizonte se comporta de forma monótona?** Si predecir a 7 días da menos quemado
  que predecir a 3, hay un problema de calibración del modelo con *h*, no de propagación.
- **¿FireCCI S3.1.1 está de acuerdo con CEMS?** Como tercera opinión: si la divergencia
  entre FireCCI y CEMS es pequeña, refuerza la confianza en las etiquetas de CEMS; si es
  grande, es el dato de ruido de etiquetas que había (investigacion.md §2, R21).

---

## 6. Experimento de divergencia (1 día, resuelve la mayor incógnita)

```
  Para las 46 series limpias (o las 96, aceptando el sesgo de TMB-S3):
    1. Calcular el quemado desde Sentinel-3 solo (caída de NIR + visible)
    2. Comparar con el polígono CEMS, píxel a píxel a 300 m

    Resultado:
      acuerdo > 90%  ->  Sentinel-3 y CEMS son compatibles. Etiquetado
                        cuantificado y adelante.
      acuerdo < 70%  ->  la discrepancia ES el ruido. S1 + MCD64A1 pasan a
                        ser obligatorios, y el dato "% de divergencia" se
                        convierte en un resultado del propio trabajo.
```

Es también una cifra muy valiosa para el informe: "la divergencia entre delineación de
alta resolución y observación de 300 m es del X%".

## 7. Escalera de implementación

```
  PASO 0   Auditoría de splits (SOLO metadatos, sin descargar imagen)  <- HECHO
           Cruzar manifest.parquet de TMB-S3 con nuestros eventos
           -> 46 series limpias + CV agrupada de 5 folds
  FASE 0   Mesogeos 1 km + WildFireSpread      -> cadena y referencia de pipeline
           (tarea extent; los baselines de propagación son B0/B1/B2, dataset.md §9)
  FASE 1a  Preentrenamiento con MCD64A1 (supervisor, Mediterráneo)
           -> modelo M-A (que nunca ha visto CEMS)
  FASE 1b  Ajuste con CEMS (46 series, 5 folds)  -> modelo M-B
           + cicatriz de entrada: TMB-S3 congelado, en las tres fases
  FASE 2   S1/S2 10-20 m                        -> severidad y refinado de borde
```

> **FASE 1a y FASE 1b son independientes**: si se acaba el tiempo, FASE 1a ya da un
> resultado defendible (M-A), y FASE 1b solo lo mejora. El informe reporta los dos
> (dataset.md §8).
