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
> ✓ **Censo canónico (2026-11-18):** el manifiesto `data/cems_manifest_46.json` contiene
> **411 adquisiciones únicas** sobre **44 series candidatas** (414 productos OLCI, 3
> colapsados por duplicidad). El dataset ensamblado contiene **296 muestras únicas en 40
> series**: 29 (fold 0), 15 (fold 1), 108 (fold 2), 96 (fold 3), 23 (fold 4) y 25 fuera de
> dominio (fold -1). Horizonte $h$: mediana 0,903 días, p75 1,042 y máximo 7,024.
> El inventario completo —incluidas las exclusiones y sus motivos y la cobertura de
> arranque en frío— vive en `dataset_index.json` y se publica en
> [`operacion.md`](operacion.md) §9.

Una **muestra** es un parche de **64×64 px a 300 m (19,2 km)** centrado en la quemadura,
con una ventana temporal de **7 frames OLCI** que termina en el frame *f*. Hay que
distinguir dos cosas: lo que se **guarda en disco** y lo que **devuelve el cargador**.

**En disco** (`data/processed/samples/*.npz`, un fichero por muestra):

```
   x_olci       (7, 5, 64, 64)  float16  radiancia TOA, 5 bandas, ~2 días
   x_scar       (64, 64)        float16  quemado en f, según TMB-S3 (fracción quemada)
   x_dem        (64, 64)        float32  elevación en m (Copernicus DEM)
   x_worldcover (64, 64)        uint8    clase de cobertura ESA WorldCover
   horizon_days ()              float32  h en días (entrada explícita del modelo)

   target       (64, 64)        float16  lo NUEVO quemado en (f, f+h]      <- TARGET
   mask         (64, 64)        uint8    1 = evaluable, 255 = "nadie miró aquí"
   target_cems  (64, 64)        float16  crecimiento real entre rondas CEMS (referencia)
   mask_cems    (64, 64)        uint8    validez común de las dos rondas
```

**En memoria** (`BurnScarDataset` en `src/burn_scar/dataset.py`): los mapas ganan eje de
canal, todo pasa a `float32` y se normaliza; `has_reference` **se deriva** (vale 1 si el
paquete trae `target_cems`):

```
   x_olci       (7, 5, 64, 64)  float32  radiancia TOA / 1000
   x_scar       (1, 64, 64)     float32  cicatriz en f
   x_dem        (1, 64, 64)     float32  elevación en km
   x_worldcover (1, 64, 64)     float32  clase de cobertura / 100
   horizon_days (1,)            float32  h en días
   target       (1, 64, 64)     float32  lo NUEVO quemado en (f, f+h]
   mask         (1, 64, 64)     uint8    1 = evaluable, 255 = ignorado
   target_cems  (1, 64, 64)     float32  crecimiento real entre rondas CEMS
   mask_cems    (1, 64, 64)     uint8    validez común de las dos rondas
   has_reference(1,)            float32  1 si la muestra trae target de referencia
   meta                        dict    sample_id, series_id, fold, banda de horizonte…
```

**Lo que el modelo todavía no consume**, aunque esté previsto: Sentinel-1 (radar),
Sentinel-2 (línea base pre-incendio), ERA5/GFS (meteo observada y prevista) y FIRMS/VIIRS
(focos activos). Son la v2 (BL5) y están fuera del modelo entrenado: entrar en ellos exige
definir antes su formato, su latencia y su coste de descarga.

Las etiquetas CEMS rasterizadas (fracción quemada y máscara de validez **por ronda**)
viven aparte, en `data/processed/cems_labels/`: el eje de las rondas (`T'`) es distinto
del eje de los frames (`T`) y se cruzan con los metadatos del índice.

El GT vive en el eje de rondas y el target en el de frames; cada frame apunta a la ronda
siguiente (`r+1`) mediante los metadatos, y de ahí sale el horizonte real
`h = observed_time(r+1) − t_frame`. Los frames sin ronda objetivo (antes de la primera o
después de la última) no entran en el entrenamiento: su validez es `255` en todas partes y
su métrica no es informativa.

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
de un incendio real (decisiones.md §2 R16). El índice marca por muestra el estado de la
cicatriz (`scar_state`: `detected` o `cold_start`) y su cobertura; los frames en los que
TMB-S3 no pudo producir una predicción utilizable se tratan como arranque en frío.

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

### El límite real son las series, y además el reparto fijo las desperdicia

Dos cosas se suman, y conviene no confundirlas:

**1. La auditoría de splits deja 46 series utilizables** (decisiones.md §3): 50 de las 96 están en el
`train`/`val` de TMB-S3, que se las sabe de memoria, y su cicatriz de entrada significa
otra cosa. De esas 46 limpias, el manifiesto construye **44 series** (36 mediterráneas y 8 de
ultramar/atlántico, de las cuales 10 son el estrato de control del `test` de TMB-S3) y el
ensamblador publica **40 series** con muestras válidas: **33 en la CV** (folds 0-4) y **7 fuera
de dominio** (fold -1, el conjunto de test).

**2. Un reparto fijo 60/20/20 sobre 46 series desperdicia el 80%.** Repartir 46 en tres
bolsillos deja 9 series como test y **37 series que nunca se evalúan**: sirven para
entrenar y no aportan ninguna medida. Con tan pocos eventos, cada serie de test vale más
que una serie de train.

→ **Se sustituye el split fijo por validación cruzada agrupada** (D14): las series se
parten en **5 folds por cluster geográfico**; en cada fold entrenan ~4/5 y se evalúan las
restantes; al final **cada serie se ha evaluado exactamente una vez**.

| | Reparto fijo 60/20/20 | **CV agrupada, 5 folds** |
|---|---|---|
| Series evaluadas | 9 | **todas, cada una una vez** |
| Entrenamientos | 1 | 5 |
| Series de train por fold | 28 | ~26 |
| Precisión de la estimación | dispersión sobre 9 series | dispersión sobre las series evaluadas |

El coste es entrenar 5 veces. El beneficio es que **el conjunto pequeño de CEMS deja de
ser una limitación y pasa a ser el diseño del experimento**: cada serie aporta su propia
medida y la incertidumbre se estima sobre todas las series evaluadas en vez de sobre 9.

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
| **D3** | **Máscara de validez por ronda**: `255` = "nadie miró aquí". El target se evalúa solo en los píxeles con `valido(r+1)` | Fuera de la máscara no hay dato, no hay "no quemado" (R4, R10). `valido(r)` no interviene: con D9 el perímetro de la ronda anterior ya no entra en el target |
| **D4** | **Cambio de sensor**: la transición se conserva y se guarda el sensor de cada ronda, para poder estratificar la evaluación (pendiente: §10) | Parte del "crecimiento" entre rondas es diferencia de detección, no propagación (R14) |
| **D5** | **FEP excluido** de la serie de etiquetas | Es una primera estimación de baja resolución; meterla mete un borde grueso |
| **D6** | **El arranque en frío se mide y se declara, sin fabricarlo**: una muestra con `scar` completamente vacío solo existe si TMB-S3 no detecta nada en una adquisición genuina | El dataset canónico vigente tiene **0** muestras de este tipo; se declara `cold_start_represented: false` y no se publica un desempeño para ese escenario (decisiones.md §1.5, A9) |
| **D7** | **Sensor y `observed_time` de cada ronda**, siempre, en los metadatos del índice | Permite estratificar (R14) y auditar cualquier muestra |
| **D8** | **Split por cluster geográfico**, nunca aleatorio | Fuga espacial (R5) |
| **D9** | **El target se mide desde *f*, no desde la ronda anterior** | `polígono(r+1) − scar_TMB3(f)`, no `polígono(r+1) − polígono(r)`. Medir desde la ronda cuenta quemado ya visible (§2) |
| **D10** | **El horizonte *h* es una entrada del modelo** | Sin él no se puede derivar el perímetro final "agregando horizontes", ni comparar limpiamente con B2 (§2) |
| **D11** | **El preentrenamiento excluye los eventos que se solapan en espacio y tiempo con test y val** | Si no, la afirmación de generalización espacial no es verificable (R20) |
| **D12** | **Etiquetas de preentrenamiento a 300 m por *nearest*** | La arquitectura y la resolución no cambian entre fases; solo el target del preentrenamiento sale en bloques de 500 m |
| **D13** | **Del dataset CEMS se excluyen los 50 eventos en el `train`/`val` de TMB-S3** | En ellos la cicatriz de entrada es el perímetro CEMS memorizado, y el target significa "crecimiento entre rondas" en vez de "crecimiento después de *f*". Quedan 46 series (decisiones.md §3, R18) |
| **D14** | **Validación cruzada agrupada (5 folds por cluster), no split fijo** | Un reparto 60/20/20 sobre las series limpias evalúa 9 y desperdicia 37. La CV evalúa **todas**, cada una una vez (§5) |
| **D15** | **Del pool de preentrenamiento con MCD64A1 se excluyen los 96 eventos de CEMS, enteros** | Así el preentrenamiento y toda la evaluación CEMS son disjuntos **por construcción**. Coste: 96 eventos de miles → irrelevante (R20) |
| **D16** | **La evaluación se hace siempre contra CEMS**, nunca contra MCD64A1 ni FireCCI | CEMS lo dibujaron analistas mirando imágenes **distintas** de las del modelo. Con FireCCI S3.1.1 el modelo podría limitarse a replicar su clasificador (R21) |
| **D17** | **Ventana fija de 64×64 px a 300 m (19,2 km), centrada en la quemadura** | Es la ventana de inferencia de TMB-S3 (stride 32 sobre tiles de 84×83 px). La cicatriz y la entrada del modelo de propagación tienen que ser **el mismo patch** (§6) |
| **D18** | **La secuencia de entrada incluye un frame pre-incendio siempre que haya uno utilizable**; si no lo hay, la ventana se completa con las últimas 7 adquisiciones utilizables hasta `f` y la muestra **no se descarta** | TMB-S3 demuestra que el modelado temporal solo mejora con una adquisición pre-fire (investigacion.md §2, hallazgo 1). Sin ella, el modelo solo ve "el fuego sigue donde iba" |
| **D19** | **Una adquisición se descarta cuando más del 20 % de sus píxeles carecen de dato válido en todas sus bandas** (o cuando no tiene ninguna señal válida: adquisición nocturna). El porcentaje se mide sobre **píxeles**, no sobre celdas de banda | La medida por celda ataba el criterio al número de bandas (5) y trataba igual un píxel con 4 bandas válidas que uno completamente muerto. La regla de D3 ("nadie miró aquí") es del píxel, no de cada banda (decisiones.md §1.5, A4) |
| **D20** | **La ventana temporal termina en el frame `f`** y la muestra se descarta si no hay adquisición utilizable de `f` | La cicatriz de entrada y el target miden quemado posterior a `f`; si la ventana acaba en otra adquisición, ambos quedan anclados a un instante distinto del que define el horizonte `h`. `f` se empareja con su fichero por cercanía temporal (tolerancia de 2 min), porque el manifiesto trunca a segundos y el fichero se nombra con el tiempo redondeado (decisiones.md §1.5, A2 y A3) |
| **D21** | **Una adquisición OLCI tiene identidad canónica `(satélite, segundo UTC)`** | Productos del catálogo que representen la misma adquisición se colapsan de forma determinista, se conservan sus identificadores para auditoría y nunca se sobrescribe un raster ni se duplica una muestra |
| **D22** | **DEM y WorldCover se validan antes de ensamblar** | WorldCover solo admite las clases ESA WorldCover v200 `{10,20,30,40,50,60,70,80,90,95,100}`; una capa ausente, vacía, con valores no finitos o con clases desconocidas excluye la serie con un motivo explícito |

El índice canónico usa la versión de esquema 2. Su bloque `inventory` reconcilia entradas,
muestras únicas, ficheros únicos, series y folds; registra la huella del manifiesto de
entrada, las series excluidas y sus motivos, la cobertura de cicatriz y la distribución de
horizontes. Los horizontes se conservan como valores continuos y se dividen únicamente en
estratos de diagnóstico `(0,1]`, `(1,3]`, `(3,7]`, `(7,14]` y `(14,+inf)`; ninguna banda
descarta una muestra ni filtra las métricas globales. El dataset y la evaluación **no
imponen ningún horizonte máximo**: el dominio del modelo es `h > 0` y el rango observado se
publica como soporte empírico, mientras que cualquier política operativa futura de máximo o
de advertencia de extrapolación queda fuera del alcance de esta documentación.

La procedencia de cada artefacto es obligatoria: el índice registra la versión del
generador, la huella SHA-256 del manifiesto de entrada y una huella de contenido estable
(`dataset_content_sha256`) que no depende de su cabecera temporal, de modo que dos
regeneraciones del mismo dataset sean comparables.

## 8. Arquitectura de tres fuentes

El proyecto usa **tres fuentes de datos, cada una en un rol distinto**. La regla que las
gobierna: **cambia la supervisión, nunca las entradas**. La implementación actual de
propagación consume OLCI, DEM, WorldCover, la cicatriz de TMB-S3 y el horizonte; meteorología
y radar siguen fuera del modelo entrenado.

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
| DEM, WorldCover | ✓ | ✓ | ✓ |
| Meteo | — (fuera del modelo actual) | — | — |
| Horizonte *h* | rejilla `{1,2,3,5,7,10,14}` d | el real de CEMS | el que se consulte |
| **Target** | `BurnDate ∈ (f, f+h]` (MCD64A1) | `polígono(r+1) − scar_TMB3(f)` (CEMS) | — |
| **Resolución del target** | 500 m → 300 m (*nearest*) | 300 m fraccional | — |
| **Eventos** | MCD64A1, Mediterráneo, **excluidos los 96 de CEMS** (D15) | 40 series ensambladas (D13) | — |

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
  ETAPA 2  Ajuste             CEMS     (40 series ensambladas, 5 folds agrupados)
            ↓
  EVALUACIÓN                   CEMS     (cada serie evaluada una vez)
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

| # | Baseline | Idea | Por qué importa | Estado |
|---|---|---|---|---|
| **B0** | Persistencia | El fuego no crece | Suelo trivial | implementado |
| **B1** | Dilatación radial | El frente avanza isotropemente | Geometría pura | implementado |
| — | Todo quemado | Predice quemado en todo el mapa | La **escala mínima** de cada etiqueta: por debajo, no se ha aprendido nada | implementado |
| **B2** | **Advección por viento** | El frente avanza en la dirección del viento | **Física sin ML** | **pendiente** (P2) |

> **Criterio de aceptación: el modelo debe superar B2 en F1 de área recién quemada.**
> Si una red no bate "mueve el frente según el viento", no está aprendiendo propagación:
> está aprendiendo a rellenar ruido. B2 todavía **no está implementado**: definirlo con
> precisión (qué viento, qué velocidad de avance) es la pendiente P2 de
> [`decisiones.md`](decisiones.md) §1.2, y es lo que bloquea la evaluación.

## 10. Protocolo de evaluación

Con pocas series evaluadas una vez cada una, *cómo* se promedia importa tanto como la
métrica. Fijado de antemano, para que no se pueda ajustar después; las dos filas marcadas
como **pendientes** son las que todavía no están implementadas en los scripts de
evaluación.

| Elemento | Decisión |
|---|---|
| **Reparto** | **CV agrupada, 5 folds por cluster geográfico** (D14). Cada serie se evalúa exactamente una vez |
| **Métrica principal** | F1 del área **nuevamente quemada**, **promedio por evento** (un evento = una serie; dentro de la serie, promedio de sus muestras) |
| **Incertidumbre** | Dispersión **entre folds**: el F1 por evento se reporta como media ± desviación típica de los 5 folds, y el umbral operativo se congela con la mediana de los umbrales óptimos por fold (su dispersión se publica aparte) |
| **Métrica secundaria** | F1 agregado por píxel, para poder comparar con WSTS y TS-SatFire |
| **Binarización** | Umbral **0,5 fijado de antemano**, sin calibrar (como TMB-S3) |
| **Máscara** | Solo píxeles válidos: `valido(r+1)` (§7 D3). `valido(r)` no interviene: con D9 el perímetro de la ronda anterior ya no entra en el target |
| **Estratificación obligatoria** | **Pendiente de implementar.** El índice guarda el sensor de cada ronda (`sensor_from`, `sensor_to`, `sensor_changed`) y la marca de control de TMB-S3 (`in_tmb_test`), pero **ningún script de evaluación los usa todavía**: hoy no se publica el subconjunto "limpio" ni el estrato de TMB-S3. Es una deuda conocida (decisiones.md §2 R14) |
| **Estrato TMB-S3** | **Pendiente de implementar**: las series del `test` de TMB-S3 están marcadas en el manifiesto (`in_tmb_test`), pero todavía no se reportan aparte. Miden si TMB-S3 generaliza de verdad (decisiones.md §3) |
| **Comparación** | Todos los baselines (B0, B1 y, cuando exista, B2) se evalúan **con el mismo protocolo y los mismos folds** que el modelo, y se reportan los dos modelos (M-A sin ajuste, M-B con ajuste) |
| **Denominadores** | La evaluación informa por separado de **entradas, muestras únicas, ficheros únicos, series y muestras evaluables**, y usa un único descriptivo (`evaluated_set`) para el modelo y para los baselines. Un índice con `sample_id` o rutas repetidos se rechaza antes de calcular métricas |
| **Bandas de horizonte** | Las métricas se reportan también por banda `(0,1]`, `(1,3]`, `(3,7]`, `(7,14]` y `(14,+inf)`, **solo como diagnóstico**. Las métricas globales se calculan sobre **todas** las muestras únicas: ninguna banda filtra el entrenamiento ni la evaluación, y no hay horizonte máximo |
| **Arranque en frío** | El estrato de arranque en frío se reporta **con su estado**: si el conjunto no contiene ninguna muestra con `x_scar` vacío, se declara "no representado" y **no se publica un desempeño** para ese escenario |
| **Procedencia** | Cada artefacto registra la versión del generador, la huella SHA-256 del índice, la huella de contenido estable, la semilla, el esquema y el soporte de horizontes |

> **Por qué el promedio por evento y no por píxel.** Agregar por píxel hace que un solo AOI
> enorme (EMSR837 tiene 111.489 píxeles, 27 veces la mediana) domine la métrica. Con pocas
> series evaluadas, además, cualquier diferencia pequeña está dentro del ruido: sin una
> medida de dispersión, "el modelo bate a B2" no es una afirmación verificable.

---

## 11. Pipeline de construcción (de los datos a las muestras)

Cinco pasos, cada uno con su script. Es la ruta que produce el dataset canónico; los
comandos exactos y las latencias están en [`operacion.md`](operacion.md) §8.

| # | Paso | Script | Qué produce |
|---|---|---|---|
| 1 | **Auditoría de splits** | `scripts/audit_tmb_splits.py` | Qué series vio TMB-S3 y cuáles quedan limpias (decisiones.md §3) |
| 2 | **Manifiesto CEMS** | `scripts/build_cems_manifest.py` | `data/cems_manifest_46.json`: series, rondas, sensores, horizontes candidatos |
| 3 | **Vectores CEMS** | `scripts/download_cems_vectors.py` | GeoJSON por ronda: `observedEventA` (quemado) y `notAnalysedA` (no observado) |
| 4 | **Rasterización** | `scripts/rasterize_cems_labels.py` | Etiquetas fraccionales a 300 m en 64×64, con máscara tri-estado (D2, D3, D17) |
| 5 | **Imágenes y ensamblaje** | `scripts/download_gee_olci.py` → `scripts/build_pytorch_dataset.py` | Parches OLCI + DEM + WorldCover, cicatriz de TMB-S3, ventana temporal de 7 frames y el índice |

**Los pasos 3-5 son la cadena que hay que repetir tras cualquier cambio** en el generador
de cicatriz, en la ventana temporal o en las reglas de filtrado: la cicatriz y el target
quedan *horneados* en los `.npz`, así que regenerar el dataset no es opcional
(operacion.md §8).

**Qué hace el ensamblador, en orden** (`scripts/build_pytorch_dataset.py`):

1. Compone la ventana de **7 frames**: una adquisición pre-incendio (si la hay) + hasta 6
   post-incendio que terminan en `f`; si no hay pre-incendio utilizable, se rellena con las
   últimas 7 adquisiciones (BL1, D18, D20).
2. Descarta las adquisiciones inservibles: más del **20 % de píxeles sin ninguna banda
   válida** o sin ninguna señal (nocturnas) (D19).
3. Sustituye cada celda de banda sin dato por la **media de su banda** en el frame. Nunca
   por ceros: en las bandas visibles, el 0 parecería quemado.
4. Calcula `x_scar` con **TMB-S3 congelado** sobre el frame `f` (D1).
5. Calcula los dos objetivos: el de ensamblaje `clip(CEMS(r+1) − scar(f), 0, 1)` y el de
   referencia `clip(CEMS(r+1) − CEMS(r_f), 0, 1)`, con sus máscaras (D9, D3).
6. Asigna cada muestra a su fold geográfico y escribe el índice con su inventario
   (entradas, muestras, series, exclusiones, soporte de horizontes y de cicatriz).

**Lo que se comprueba después**: los invariantes de la ventana temporal (terminar en `f`,
orden temporal) en `tests/test_dataset_invariants.py`; la codificación canónica de las
máscaras en `tests/test_generator_version.py`; y la coherencia entre la documentación y
el índice en `tests/test_documentation_consistency.py`.

---
