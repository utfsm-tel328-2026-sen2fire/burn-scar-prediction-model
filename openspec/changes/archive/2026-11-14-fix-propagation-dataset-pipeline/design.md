# Design: Corrección del pipeline de dataset y evaluación de propagación

## Context

Véase `proposal.md` para la motivación. Restricciones relevantes que condicionan el diseño:

- El checkpoint TMB-S3 disponible (`convlstm_segformer_prepost_subset/model.safetensors`) fue entrenado con el código de `links-ads/tmb-s3`. Su módulo de losses define `_FIRE_CH = 0` y `_CLOUD_CH = 1` en `src/bafep/modules/mean_teacher.py`, y entrena el canal 0 con `WeightedBCELoss` (un único logit contra un target 0/1), mientras que el canal 1 recibe un peso `lambda_cloud = 0`.
- El checkpoint **no** tiene canal de flags pre/post: `cell.conv.weight` tiene forma `(256, 70, 3, 3)`, es decir 6 canales de entrada + 64 de estado oculto. Por tanto no hace falta pasar `pre_post_flag`; basta con que el frame pre-incendio ocupe la primera posición de la secuencia.
- Las frames descargadas desde GEE son `OaXX_radiance` (radiancia ×1000). El 24,4% de las 2121 frames descargadas son completamente "fill" (32768) y una fracción adicional son frames nocturnos con todos los píxeles a cero.
- La normalización upstream (`S3OLCIMeanTeacherTransform._MEAN/_STD`) está calculada sobre reflectancia TOA ×10000; el proyecto usa z-score por imagen sobre radiancia. Son magnitudes distintas (ver `proposal.md`, riesgo residual).
- `data/processed/` está en `.gitignore`: el dataset no se versiona y cada investigador lo regenera en local.
- El entorno de desarrollo ROCm (AMD) reporta la GPU Navi 21 como `gfx1030` y el tensor `torch.cuda.is_available()` devuelve `True`, igual que en CUDA: el código que detecta el backend a partir de ese valor asume NVIDIA y activa `pin_memory` en AMD, donde es contraproducente.
- El modelo de propagación tiene 780 k parámetros y consume 32 ms por lote de 8 en la GPU de referencia: el entrenamiento de este proyecto está lejos de saturar la GPU, de modo que cualquier sobrecoste por época proviene de la carga de datos y no del cómputo.

## Goals / Non-Goals

**Goals:**
- Que el dataset generado contenga un target con quemado real en la mayoría de las muestras.
- Que la ventana temporal refleje la composición con la que se entrenó TMB-S3 (1 pre-incendio + 6 post-incendio) y no contenga frames sin información.
- Que ninguna métrica pueda reportar un desempeño perfecto sobre una verdad vacía.
- Que los números de la red sean comparables con los baselines binarios B0/B1.

**Non-Goals:**
- Convertir la radiancia de GEE a reflectancia TOA ×10000 y adoptar las medias/std fijas de upstream (requiere los coeficientes ADB del sensor OLCI y la geometría solar).
- Cambiar la definición del target (`clip(CEMS − scar, 0, 1)`): es una decisión de diseño documentada (D9) y se mantiene.
- Rediseñar la arquitectura del modelo de propagación y el preentrenamiento con TMB-S3.
- Modificar la descarga de datos crudos: el filtrado se aplica en la fase de ensamblaje, sin volver a descargar.

## Decisions

### Decisión 1: Leer el canal 0 con sigmoid en lugar de softmax sobre 2 canales
- **Elección**: `predict_burn_scar` devuelve `sigmoid(logits[:, :, 0])`.
- **Razón**: el canal 0 es un logit único entrenado con BCE contra el target binario de quemado; aplicar softmax sobre las dos clases y tomar el índice 1 interpretaba erróneamente la cabeza de nubes como "probabilidad de quemado". El empíricamente medido: IoU media frente a CEMS de 0,42 con el canal 0 frente a 0,09 con el canal 1.
- **Alternativas consideradas**: (a) invertir los índices del softmax — descartado, porque el canal 1 no está entrenado como quemado; (b) reentrenar TMB-S3 localmente — descartado por coste y por romper la comparabilidad con el checkpoint publicado.

### Decisión 2: Composición de la ventana con frame pre-incendio y filtro de calidad
- **Elección**: el corte pre-incendio es la hora de adquisición de la primera ronda CEMS de la serie; la ventana se compone con la adquisición pre-incendio utilizable más reciente seguida de las seis post-incendio utilizables que terminan en f.
- **Razón**: BL1 define 1 frame pre-incendio + 6 post-incendio, y es la composición con la que se entrenó TMB-S3. El 100% de las series (44/44) dispone de frames pre-incendio en los datos crudos, así que la restricción no reduce el número de muestras.
- **Alternativas consideradas**: (a) mantener la ventana actual de 7 frames post-incendio — descartado, reproduce la diferencia de distribución con el entrenamiento upstream; (b) usar la fecha de ignición en lugar de la primera ronda CEMS — descartado por falta de ese dato en el manifiesto; la primera ronda es una aproximación razonable (es cuando CEMS detecta el incendio).

### Decisión 3: Filtro de calidad de adquisición con los mismos umbrales que upstream
- **Elección**: se descartan las adquisiciones con más del 20% de píxeles sin dato y las que no tienen ningún píxel válido con señal (frames nocturnos).
- **Razón**: el 44,1% de las ventanas actuales contenía frames nocturnos y el 96,9% de las muestras tenía al menos un frame inútil. El umbral del 20% es el que documenta la ficha del dataset upstream ("acquisitions with more than 20% invalid pixels were discarded").
- **Alternativas consideradas**: rellenar los frames nocturnos con ceros — descartado porque una frame de ceros es indistinguible de un input degenerado y aporta señal falsa al ConvLSTM.

### Decisión 4: Sustituir el centinela de nodata por la media de la banda
- **Elección**: cada píxel sin dato se sustituye por la media de su banda dentro del mismo frame, y el valor 32768 deja de almacenarse.
- **Razón**: 32768 no es radiancia; al dividir por 1000 en el `Dataset` entraban como outliers de orden de magnitud imposible (32,7 frente a valores de 0,05–0,9). La media de la banda es un valor neutro y no inventa burned areas como sí haría el cero (en bandas visibles, 0 = "muy oscuro" = aspecto de quemado).
- **Alternativas consideradas**: (a) rellenar con cero — descartado por el sesgo hacia "quemado" en visible; (b) añadir un canal de validez al modelo — descartado por ahora, cambiaría la firma de entrada del modelo y excede el alcance; queda como opción si la calidad sigue siendo insuficiente.

### Decisión 5: NaN en lugar de 1,0 para métricas no informativas
- **Elección**: `compute_metrics_numpy` devuelve NaN cuando el target no tiene ningún píxel quemado por encima del umbral; un nuevo agregador por evento ignora los NaN e informa de cuántas series han podido evaluarse.
- **Razón**: el comportamiento anterior (1,0) era la causa de que un dataset vacío se midiera como un 100% perfecto. NaN hace que el fallo sea visible en el informe en lugar de hiding.
- **Alternativas consideradas**: devolver 0 en lugar de NaN — descartado porque 0 es indistinguible de "el modelo no predice nada", que es un resultado legítimo de un baseline; con NaN el agregado puede informar del número de eventos evaluados.

### Decisión 6: `pos_weight` derivado del lote
- **Elección**: `masked_bce_dice_loss` calcula el peso positivo como el cociente entre píxeles válidos no quemados y quemados del lote, acotado a un rango configurable, salvo que el llamante providea un valor explícito.
- **Razón**: con el dataset corregido los píxeles quemados son ~4,7% del área válida; el valor fijo 4,0 apenas compensa ese desbalance y empuja al modelo a predecir ausencia de quemado (recall bajo, F1 engañosamente bajo).
- **Alternativas consideradas**: (a) mantener 4,0 — descartado por lo anterior; (b) focal loss — no implementado en este cambio, se deja como mejora posterior si el recall sigue siendo insuficiente.

### Decisión 7: Reporte del umbral óptimo en la evaluación
- **Elección**: `evaluate_fold` evalúa las métricas en un barrido de umbrales (0,1–0,7) y reporta el F1 a 0,5, el mejor F1 y el umbral que lo alcanza.
- **Razón**: un modelo probabilístico desbalanceado opera en un umbral distinto de 0,5; los baselines B0/B1 producen predicciones binarias y no tienen umbral, así que el F1 a umbral fijo no es una comparación justa.
- **Alternativas consideradas**: mantener solo 0,5 — descartado por lo anterior.

### Decisión 8: Umbral operativo único, congelado desde validación y medido en test
- **Elección**: el umbral operativo es la **mediana de los umbrales óptimos de los folds de validación**; ese valor se congela y se usa para evaluar el conjunto de test (fold -1, fuera de dominio) con el checkpoint de cada fold. El umbral se guarda junto a los checkpoints.
- **Razón**: elegir época y umbral en el mismo fold de validación es doble uso de la misma muestra y produce cifras optimistas. Congelar un único umbral, elegido sin ver el test, hace que el número final sea publicable y comparable con la literatura (upstream reporta con `--burnt-vote-threshold 0,5`, de ahí que se siga reportando también el F1 a 0,5).
- **Alternativas consideradas**: (a) un umbral por fold — descartado, en operación solo puede existir uno y el ensemble de folds no define un umbral único; (b) calibrar las probabilidades con Platt/isotónica y fijar 0,5 — es la solución estándar y más limpia, pero requiere un conjunto de calibración adicional que el proyecto no tiene; queda como mejora posterior; (c) fijar 0,5 sin más — descartado, con ~5% de positivos el modelo opera muy por debajo de 0,5.

### Decisión 9: El umbral solo hace falta para decisiones binarias
- **Elección**: el informe documenta que la estimación de área quemada se obtiene sumando las probabilidades, sin umbral, y que el umbral solo se aplica cuando se necesita una decisión binaria (alertas, mapas de riesgo categorizados).
- **Razón**: el área esperada quemada no depende de un corte arbitrario; introducir un corte añade una discontinuidad artificial en el producto. Fijar un único criterio para ambos usos confundiría el significado de los números.
- **Alternativas consideradas**: usar el umbral también para el área — descartado por lo anterior.

### Decisión 10: Configuración de la carga de muestras según el backend
- **Elección**: una función de resolución elige `(num_workers, pin_memory)` según el backend: en ROCm se usa `num_workers=0` y `pin_memory=False`; en CUDA se mantienen 2 procesos auxiliares y memoria fijada. Ambos valores se pueden forzar desde la línea de comandos.
- **Razón**: medido en la RX 6800 XT con el dataset completo, una época de entrenamiento tardaba **74 s** con 2 procesos auxiliares y memoria fijada, frente a **1,7 s** sin ellos (43×). La GPU nunca fue el cuello de botella: el coste estaba en bifurcar procesos desde un proceso con el contexto HIP ya inicializado y en reservar memoria fijada con `hipHostMalloc`, que en ROCm es carísimo. En CUDA el comportamiento habitual no sufre ese problema, y con este dataset (≈2 ms por muestra) los procesos auxiliares no compensan ni allí.
- **Alternativas consideradas**: (a) `persistent_workers=True` — recorta el coste a partir de la segunda época (1,2 s) pero la primera y la validación siguen pagando la bifurcación, y no elimina el coste de la memoria fijada; (b) procesos auxiliares sin memoria fijada — peor (82 s por época); (c) ampliar el dataset hasta que los procesos auxiliares compensen — descartado, porque el cuello de botella no es de datos sino de proceso.

### Decisión 11: Corrupción de la cicatriz vectorizada
- **Elección**: la elección entre dilatación y erosión de la cicatriz de entrada se hace con una única tirada por lote y se aplica de forma vectorizada, en lugar de un bucle por muestra que sortea la GPU en cada iteración.
- **Razón**: el bucle original consultaba un número aleatorio por muestra, lo que bloquea la GPU en cada muestra del lote; en un backend donde cada sincronización se paga cara, ese coste es puro desperdicio. El comportamiento observable no cambia: cada muestra recibe una única decisión morfológica.
- **Alternativas consideradas**: mantener el bucle por muestra — descartado por lo anterior.

## Risks / Trade-offs

- **[El dataset preexistente queda obsoleto]** → La regeneración es determinista y local (≈20 s con GPU); los ficheros `.npz` no se versionan, así que no hay conflicto de merge. Documentar el orden obligatorio: regenerar antes de entrenar.
- **[Menos muestras tras el filtrado]** → El filtro de calidad puede reducir el número de muestras por serie; se registra el recuento por serie en el índice para que la pérdida de cobertura sea visible.
- **[Relleno por media puede sesgar los píxeles de borde]** → El margen afectado es de ~4,6% de los píxeles y los nodata se concentran en el borde del swath, fuera del centro de la quemadura; se acepta como mal menor frente a dejar el centinela.
- **[El `pos_weight` por lote es ruidoso]** → Con batch size 8 y ~64×64 píxeles válidos hay miles de píxeles por lote, por lo que la estimación del cociente es estable; el acotado evita valores extremos en lotes sin positivos.
- **[La normalización sigue sin coincidir con upstream]** → Se deja explícitamente fuera de alcance (ver Non-Goals); el impacto medido es una cicatriz con IoU 0,42 frente a CEMS, suficiente para que el target sea informativo. Pendiente de una fase de calibración posterior.
- **[Forzar la carga en un solo proceso puede penalizar en CUDA]** → Los valores por defecto se resuelven por backend y se pueden forzar con `--num-workers` y `--pin-memory`; con un dataset mucho mayor habría que re-medir antes de fijar la configuración.
- **[El umbral óptimo resultó inestable entre folds]** → En el primer entrenamiento completo los umbrales por fold fueron 0,1 / 0,1 / 0,3 / 0,1 / 0,3 (dispersión 0,2), con el umbral congelado en 0,1 el modelo alcanza recall 86% pero precisión 25%. El entrenamiento ya lo advierte en el informe; la causa probable es la falta de calibración de probabilidades con ~5% de positivos, no un fallo del pipeline.

## Migration Plan

1. Aplicar el cambio de código (tareas de `tasks.md`).
2. Regenerar el dataset en local: `uv run --extra rocm python scripts/build_pytorch_dataset.py` (sobrescribe `data/processed/`).
3. Verificar la distribución del target y la ausencia de frames nocturnos en el dataset nuevo.
4. Reevaluar los baselines sobre el dataset nuevo para tener la referencia de comparación.
5. Reentrenar la red.

El ajuste de rendimiento de la carga de datos (grupo 7 de `tasks.md`) no requiere regenerar el dataset: solo cambia cómo se leen las muestras ya generadas.

**Rollback**: revertir el commit de código y regenerar el dataset con la versión anterior del script; el dataset no está versionado, por lo que no hay migraciones de datos que revertir.

## Open Questions

- Si tras la corrección el recall sigue siendo bajo, ¿el `pos_weight` por lote basta o hace falta focal loss o umbrales múltiples por fold? Depende de los resultados del entrenamiento y no cambia el diseño de este cambio.
- ¿Merece la pena una fase de calibración radiancia→reflectancia TOA para adoptar las medias/std fijas de upstream? Requiere decidir el origen de los coeficientes ADB del sensor OLCI; es un cambio independiente.
- El umbral óptimo fue inestable entre folds (dispersión 0,2) y el número publicable en test (F1 33,8%, IoU 23,3%) triplica al baseline B1 (F1 10,8%) pero queda lejos de TMB-S3 upstream (F1 69,45 / IoU 59,81). ¿Basta con calibrar las probabilidades (Platt/isotónica) sobre un conjunto de calibración, o el salto restante viene de la normalización radiancia→reflectancia TOA y de la definición del target?
