# Dataset — especificación

> **En una frase:** qué es una muestra, qué se predice sobre ella, qué reglas hay que
> cumplir al construirla y cómo se mide el acierto.
>
> Especificación de lo que hay que construir: cómo se hace una muestra, qué es el
> objetivo, qué reglas se cumplen y cómo se evalúa. Todo lo que **puede cambiar** está en
> [`decisiones.md`](decisiones.md); el porqué de cada decisión, en
> [`investigacion.md`](investigacion.md).

## 1. Glosario

Siglas y términos que aparecen en la documentación, explicados. Los que están en cursiva
son siglas o nombres en inglés.

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
| **Regla OR** | Se marca un píxel como quemado si **cualquier** píxel de 10 m dentro lo está. Infla el área |
| **Target (objetivo)** | Lo que hay que acertar: el mapa de probabilidad de quema en el horizonte pedido |
| **Baseline (modelo de referencia)** | B0, B1 y B2: modelos sin aprendizaje automático contra los que se compara el nuestro |
| **Ground truth (GT)** | La "verdad" contra la que se mide el modelo |
| **Split** | Cómo se reparte el dataset en train / validación / test |
| **Fuga espacial** | Que el mismo incendio (o su vecino) aparezca en train y test. Infla los resultados |
| **U-Net / SegFormer / ConvLSTM / ViT** | Arquitecturas de red neuronal. U-Net y SegFormer = extractoras espaciales; ConvLSTM = añade memoria temporal; ViT = transformer |
| **IoU / F1** | Métricas de acierto, 0-100 |
| **BCE ponderada** | Pérdida de entrenamiento que compensa el desbalance de clases |
| **zarr / netCDF / GeoTIFF** | Formatos de almacenamiento |
| **STAC** | Catálogo estándar para buscar escenas satelitales por metadatos |
| **UTM** | Sistema de proyección por zonas, en metros |
| **NRT / NTC** | *Near/Non Time Critical*: la versión de un producto satelital que llega en horas (NRT) o en días (NTC) |

---

## 2. Composición

Una **serie** = un (incendio, AOI) con delineaciones fechadas. Un archivo por serie.

El dataset tiene **tres ejes temporales** y una regla que los une:

- **Eje T (frames OLCI)**: una adquisición cada ~2 días. Es el eje de las entradas.
- **Eje T' (rondas CEMS)**: delineaciones en fechas irregulares. Es el eje de las etiquetas.
- **Eje h (horizonte)**: cuántos días hasta el objetivo. Es una **entrada** del modelo.

> **Regla de unión: una muestra por frame OLCI.** Para cada frame *f*:
> entrada = todos los frames hasta *f* + la cicatriz en *f* (producida por TMB-S3);
> target = lo nuevo quemado entre *f* y *f* + *h*.
> Es la formulación que **reproduce la operación** (se actualiza cada ~2 días) y hace que el
> arranque en frío aparezca solo.
>
> ⚠️ **El número de muestras con esta regla está sin medir (P7).** El censo de §3 cuenta
> **productos CEMS** (455 DEL + 96 FEP = 551), no frames OLCI: una serie tiene del orden de
> 4-5 adquisiciones OLCI durante el evento, y solo las que caen dentro del intervalo de una
> ronda y tienen ronda objetivo son muestras. El dato sale de cruzar las adquisiciones OLCI
> con las rondas CEMS, en la ingesta. Lo que sí es inmediato: 455 productos DEL frente a 359
> transiciones.

```
  EMSR922__AOI01.npz

  ENTRADAS  (eje temporal T = frames OLCI)
  ------------------------------------------------------------------
   OLCI     (T,  5, H, W)  uint16   radiancia TOA, 5 bandas, ~2 días
   ERA5     (T,  5, H, W)  float16  viento u/v, T, humedad, precipitación
   GFS      (T,  ?, H, W)  float16  viento y humedad PREVISTOS 24-72 h (bloque sin cerrar, operacion.md §7)
   FIRMS    (T,  ?, H, W)  float16  focos activos + FRP (entra en la v1 por BL5; sin acotar)
   S1       (T,  2, H, W)  float16  VV, VH (radar), ~6 días — v2, fuera de la v1 (BL5)
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

**La cicatriz de entrada la produce TMB-S3, no CEMS** (§8). Esto tiene tres
consecuencias:

- **Causalidad resuelta por construcción.** TMB-S3 solo ve los frames hasta *f*, así que
  su salida no puede contener quemado posterior. Ya no hace falta ninguna regla de
  `observed_time` para el canal de cicatriz: el problema desaparece.
- **La cicatriz va fresca.** Ya no está desfasada respecto al último frame, que era el
  problema de la regla causal (decisiones.md §2 R15).
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
   entre TMB-S3 y CEMS (mismo número que el F1 de la cicatriz, operacion.md §4).
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
de un incendio real (decisiones.md §2 R16). El flag `scar_ok` marca los frames en los que la predicción
de TMB-S3 no es utilizable (frame completamente nublado, fallo de inferencia); esos se
excluyen o se tratan aparte.

## 3. El censo medido (Mediterráneo, 2023-2026)

| Magnitud | Valor |
|---|---|
| Activaciones de incendio en la API (global) | 131 |
| Activaciones mediterráneas | 107 (14 países: Albania, Bosnia y Herz., Bulgaria, Croacia, Francia, Grecia, Italia, Montenegro, Portugal, Serbia, Eslovenia, España, Siria, Túnez; «Francia» incluye territorio francés de ultramar, ver BL6) |
| Series con ≥1 delineación | 130 |
| **Series con ≥2 delineaciones (señal de crecimiento)** | **96** |
| **Transiciones de crecimiento** | **359** |
| Productos de delineación | 489 |
| Área de AOI | min 5 km² · p25 161 · **mediana 370** · p75 949 · máx 10.034 km² |
| Superficie total | 92.210 km² |
| Píxeles a 300 m | **1.024.559** en total; mediana 4.107 por serie |
| Productos CEMS por serie (DEL + FEP) | mediana 4 · media 5,7 · máx 26 |
| Productos CEMS (serie, producto) | 551 (= 455 DEL + 96 FEP) |
| Distribución de rondas por serie | 2→33, 3→25, 4→8, 5→6, 6→2, 7→5, 8→1, 9→6, 10→1, 11→4, 13→1, 16→1, 17→1, 18→1, 25→1 |

> **Nota.** Este censo cuenta **productos CEMS**, no frames OLCI: el total (551) son 455
> delineaciones + 96 FEP, por eso la mediana de *rondas* es 3 y la de *productos* 4 (media
> 4,74 vs 5,74). **El FEP se excluye de la serie de etiquetas** (D5, §7): es una estimación
> de baja resolución y meterla metería un borde grueso. El nº de frames OLCI por AOI (el eje
> T, §2) es otro dato y está sin medir (P7).

Añadiendo el periodo pre-2023 (2012-2022, accesible por otra vía, decisiones.md §2 R1) el total
mediterráneo sube a **~162 series y ~490 transiciones**. ⚠️ Con la auditoría de splits
(decisiones.md §3) muchas de esas series también caen dentro del `train`/`val` de TMB-S3, así que el
saldo neto tras filtrar es **menor** que +66: hay que repetir la auditoría sobre el
periodo pre-2023 antes de contar con esas series.

## 4. Volumen

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

## 5. Por qué "pequeño en bytes" no significa "pequeño para aprender"

Hay que separar tres magnitudes, que se confunden:

```
  BYTES      ~115 MB       -> irrelevante, es una ventaja
  PÍXELES    10,7 k px/serie (mediana 4.107; 4.096 px por muestra tras D17)
                            -> suficiente para aprender por píxel
  EVENTOS    96             -> ESTE es el límite real
```

Todos los parches de un mismo incendio son el mismo incendio. Para generalizar, y para que
el test set signifique algo, lo que cuenta es el número de **eventos independientes**.

Comparación: WSTS 607 eventos, TS-SatFire 179, TMB-S3 246. **96 está por debajo del
 estándar de la literatura.** Ver decisiones.md §4.

### El límite real son 46, y además el reparto fijo los desperdicia

Dos cosas se suman, y conviene no confundirlas:

**1. La auditoría de splits deja 46 series utilizables** (decisiones.md §3): 50 de las 96 están en el
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
medida y el bootstrap (§10) se hace sobre 46 eventos en vez de 9.

## 6. El recorte: una tile fija de 64×64 px (19,2 km)

**El tamaño de la ventana no se elige: lo impone el modelo.** TMB-S3 infiere con una
ventana deslizante de **64×64 px con stride 32**, y sus tiles son de **84×83 px** a 300 m
(~25 × 25 km) — verificado en el `.zarr` del dataset (`s3olci` con shape `(T, 21, 84, 83)`).

A 300 m, **64 px = 19,2 km**. Ese es el lado de la ventana que ve el encoder, y es lo que
hay que recortar para que la entrada coincida con lo que el modelo preentrenado espera.

```
  TILE DE TMB-S3      84 × 83 px  =  25,0 × 24,9 km
  VENTANA DE INFERENCIA 64 × 64 px  =  19,2 × 19,2 km   (stride 32)
  LO QUE DESCARGAMOS  64 × 64 px  =  19,2 × 19,2 km   centrada en la quemadura
```

> ⚠️ **A verificar contra el código de TMB-S3:** con un tile de 84×83 y ventana de 64 con
> stride 32 caben **1** posiciones por eje (`floor((84−64)/32)+1 = 1`), no 2: para tener 2
> posiciones el stride efectivo tendría que ser ≤20. Si el código de TMB-S3 usa 2, el dato
> erróneo es el del stride (32). Hasta que se compruebe, se asume 1.

> **Decisión de diseño (D17): la ventana es 64×64 px a 300 m, centrada en el centro de la
> quemadura, no un recorte de buffer variable.** Es lo que hace que la cicatriz de TMB-S3
> y la entrada del modelo de propagación sean **el mismo patch en las dos redes**.

**El motivo principal del recorte es el coste, no la densidad.** El AOI mediano (4.107
píxeles) cabe de sobra en 64×64, pero EMSR837 tiene 111.489 píxeles: la ventana lo recorta
a 4.096 (**×27**).

El efecto sobre la densidad sí es real, pero mucho menor de lo que parece, porque el AOI
mediano (~19 km de lado) tiene **exactamente el tamaño de la ventana**. Con la quemadura
mediana (~5,1 km de lado equivalente):

| buffer (por lado) | lado resultante | píxeles | densidad | ganancia |
|---|---|---|---|---|
| 0 km | 5,1 km | 289 | 100% | ×14 |
| 2 km | 9,1 km | 921 | 31% | ×4,5 |
| 5 km | 15,1 km | 2.533 | 11% | ×1,6 |
| **7,05 km (el tope de 19,2 km)** | **19,2 km** | **4.096** | **7%** | **×1,0** |
| 20 km | 39,2 km | 17.069 | — | descartado: excede el recorte fijo (D17) |

El buffer va **por lado**: con una quemadura mediana de 5,1 km, una ventana de 19,2 km deja
**7,05 km de margen a cada lado** — ese es el buffer implícito y es el máximo posible. Es
decir, **no hay elección**: o se recorta a 19,2 km y se pierde el 93% del tile en "no
quemado", o se sube la resolución (no es una opción) o se reduce el campo de visión del
modelo (lo que invalidaría el preentrenamiento de TMB-S3). Por encima de 7,05 km el buffer
no existe: la ventana ya está topeada (D17).

> **Conclusión.** El argumento de "×4 de densidad con buffer de 2 km" era correcto como
> cálculo de densidad, pero **no era aplicable**: una ventana de 9,1 km (30 px) no es una
> entrada válida para un encoder entrenado con ventanas de 64 px. La densidad real es del
> 7%, que es baja pero asumible: el frente nuevo son 10-100 px sobre 4.096, y la pérdida
> ponderada + el muestreo alrededor del frente (decisiones.md §2 R6) lo compensan. En los AOIs grandes
> (EMSR837 y compañía) el recorte sí es de un orden de magnitud.

Esto no añade eventos, pero hace el entrenamiento mucho más eficiente: el total del dataset
baja de ~115 MB (96 series) a ~55 MB (46 series), y la descarga de OLCI —el cuello de
botella real— baja en la misma proporción.

## 7. Reglas del dataset

Resumen operativo de lo que hay que respetar al construir las muestras. Es la
especificación que se implementa y la que se revisa al auditar una etiqueta.

| # | Regla | Por qué |
|---|---|---|
| **D1** | **La cicatriz la produce TMB-S3** (no CEMS), en las tres fases | Es un modelo: no requiere activación de emergencia, no puede filtrar el futuro y es idéntico en entrenamiento y operación (§8) |
| **D2** | **Cuantificado, no binario**: la etiqueta CEMS es la fracción de área quemada del píxel | La regla OR ("cualquier píxel de 10 m") infla el área un 34% (TMB-S3) |
| **D3** | **Máscara de validez por ronda**: `255` = "nadie miró aquí". El target se evalúa en `valido(r+1) ∩ scar_ok(f)` | Fuera de la máscara no hay dato, no hay "no quemado" (R4, R10). `valido(r)` no interviene: con D9 el perímetro de la ronda anterior ya no entra en el target |
| **D4** | **Cambio de sensor**: la transición se conserva, pero se guarda el sensor de cada ronda y la evaluación se estratifica | Parte del "crecimiento" entre rondas es diferencia de detección, no propagación (R14) |
| **D5** | **FEP excluido** de la serie de etiquetas | Es una primera estimación de baja resolución; meterla mete un borde grueso |
| **D6** | **Arranque en frío incluido**: los frames en los que TMB-S3 aún no detecta nada (`scar` todo a cero) entran en el entrenamiento | El primer ciclo de un incendio real no tiene cicatriz (R16) |
| **D7** | **Sensor y `observed_time` de cada ronda**, siempre, en metadatos y en evaluación | Permite estratificar (R14) y auditar cualquier muestra |
| **D8** | **Split por cluster geográfico**, nunca aleatorio | Fuga espacial (R5) |
| **D9** | **El target se mide desde *f*, no desde la ronda anterior** | `polígono(r+1) − scar_TMB3(f)`, no `polígono(r+1) − polígono(r)`. Medir desde la ronda cuenta quemado ya visible (§2) |
| **D10** | **El horizonte *h* es una entrada del modelo** | Sin él no se puede derivar el perímetro final "agregando horizontes", ni comparar limpiamente con B2 (§2) |
| **D11** | **El preentrenamiento excluye los eventos que se solapan en espacio y tiempo con test y val** | Si no, la afirmación de generalización espacial no es verificable (R20) |
| **D12** | **Etiquetas de preentrenamiento a 300 m por *nearest*** | La arquitectura y la resolución no cambian entre fases; solo el target del preentrenamiento sale en bloques de 500 m |
| **D13** | **Del dataset CEMS se excluyen los 50 eventos en el `train`/`val` de TMB-S3** | En ellos la cicatriz de entrada es el perímetro CEMS memorizado, y el target significa "crecimiento entre rondas" en vez de "crecimiento después de *f*". Quedan 46 series (decisiones.md §3, R18) |
| **D14** | **Validación cruzada agrupada (5 folds por cluster), no split fijo** | Un reparto 60/20/20 sobre 46 series evalúa 9 y desperdicia 37. La CV evalúa **las 46**, cada una una vez (§5) |
| **D15** | **Del pool de preentrenamiento con MCD64A1 se excluyen los 96 eventos de CEMS, enteros** | Así el preentrenamiento y toda la evaluación CEMS son disjuntos **por construcción**. Coste: 96 eventos de miles → irrelevante (R20) |
| **D16** | **La evaluación se hace siempre contra CEMS**, nunca contra MCD64A1 ni FireCCI | CEMS lo dibujaron analistas mirando imágenes **distintas** de las del modelo. Con FireCCI S3.1.1 el modelo podría limitarse a replicar su clasificador (R21) |
| **D17** | **Ventana fija de 64×64 px a 300 m (19,2 km), centrada en la quemadura** | Es la ventana de inferencia de TMB-S3 (stride 32 sobre tiles de 84×83 px). La cicatriz y la entrada del modelo de propagación tienen que ser **el mismo patch** (§6) |
| **D18** | **La secuencia de entrada incluye siempre un frame pre-incendio** | TMB-S3 demuestra que el modelado temporal solo mejora con una adquisición pre-fire (investigacion.md §2, hallazgo 1). Sin ella, el modelo solo ve "el fuego sigue donde iba" |

## 8. Arquitectura de tres fuentes

El proyecto usa **tres fuentes de datos, cada una en un rol distinto**. La regla que las
gobierna: **cambia la supervisión, nunca las entradas**.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  ENTRADAS (idénticas en las tres fases)                          │
  │    frames OLCI (5 bandas)  ·  ERA5 + GFS  ·  FIRMS/VIIRS         │
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

### El ajuste con CEMS es opcional, y esa opcionalidad es un resultado

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

## 9. Baselines (el termómetro del proyecto)

| # | Baseline | Idea | Por qué importa |
|---|---|---|---|
| **B0** | Persistencia | El fuego no crece | Suelo trivial |
| **B1** | Dilatación radial | El frente avanza isotropemente | Geometría pura |
| **B2** | **Advección por viento** | El frente avanza en la dirección del viento | **Física sin ML** |

> **Criterio de aceptación: el modelo debe superar B2 en F1 de área recién quemada.**
> Si una red no bate "mueve el frente según el viento", no está aprendiendo propagación:
> está aprendiendo a rellenar ruido.

## 10. Protocolo de evaluación

Con 46 series evaluadas una vez cada una, *cómo* se promedia importa tanto como la métrica.
Fijado de antemano, para que no se pueda ajustar después:

| Elemento | Decisión |
|---|---|
| **Reparto** | **CV agrupada, 5 folds por cluster geográfico** (D14). Cada serie se evalúa exactamente una vez |
| **Métrica principal** | F1 del área **nuevamente quemada**, **promedio por evento** (un evento = una serie; dentro de la serie, promedio de sus muestras) |
| **Incertidumbre** | Bandas de confianza **bootstrap** sobre las **46** series evaluadas (1.000 remuestreos) |
| **Métrica secundaria** | F1 agregado por píxel, para poder comparar con WSTS y TS-SatFire |
| **Binarización** | Umbral **0,5 fijado de antemano**, sin calibrar (como TMB-S3) |
| **Máscara** | Solo píxeles válidos: `valido(r+1) ∩ scar_ok(f)` (§7 D3). `valido(r)` no interviene: con D9 el perímetro de la ronda anterior ya no entra en el target |
| **Estratificación obligatoria** | Los resultados se reportan **separados** entre transiciones con el mismo sensor y transiciones que lo cambian (decisiones.md §2 R14). La métrica principal se da en el subconjunto "limpio", con el número completo como segundo resultado |
| **Estrato TMB-S3** | Las 12 series que están en el `test` de TMB-S3 se reportan **aparte**: sus pesos no las entrenaron, así que miden si TMB-S3 generaliza de verdad (decisiones.md §3) |
| **Comparación** | Todos los baselines (B0, B1, B2) se evalúan **con el mismo protocolo y los mismos folds** que el modelo, y se reportan los dos modelos (M-A sin ajuste, M-B con ajuste) |

> **Por qué el promedio por evento y no por píxel.** Agregar por píxel hace que un solo AOI
> enorme (EMSR837 tiene 111.489 píxeles, 27 veces la mediana) domine la métrica. Con 46
> eventos evaluados, además, cualquier diferencia pequeña está dentro del ruido: sin bandas
> de confianza, "el modelo bate a B2" no es una afirmación verificable.

---
