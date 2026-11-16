# Design

## Context

Ver `proposal.md` para la motivación. Estado actual relevante:

- El manifiesto CEMS (`scripts/build_cems_manifest.py`) construye las muestras a partir del catálogo **CDSE**, cuyas marcas de tiempo traen precisión sub-segundo (`2026-09-13T08:49:48.959375`). El `sample_id` y el `frame_time` se derivan de ahí.
- La descarga (`scripts/download_gee_olci.py`) consulta el catálogo **GEE** (`COPERNICUS/S3/OLCI`) y nombra cada fichero con `system:time_start` redondeado al segundo (`20260913T084949`). La marca de tiempo real queda además dentro del `.npz` (`timestamp=frame_dt.isoformat()`).
- El ensamblador (`scripts/build_pytorch_dataset.py`) selecciona los frames comparando **nombres de fichero** con `f_key`, donde `f_key` es el tiempo CDSE **truncado** a segundos. Un desfase de un segundo basta para que la adquisición de `f` quede fuera de su propia ventana.
- La cicatriz de entrada se toma del **último frame de la ventana** (`scar_f = scar_probs[-1]`), no de un frame identificado por su marca de tiempo. Si la ventana no termina en `f`, la cicatriz tampoco está en `f`.

Restricciones: no hay nuevos datos externos más allá de los catálogos ya consultados; el dataset se genera en GPU (TMB-S3 congelado) y las 390 muestras actuales se ensamblan en pocos minutos; los números publicados en `docs/operacion.md` se derivan del dataset actual y quedan obsoletos al regenerarlo.

## Goals / Non-Goals

**Goals:**
- Que la ventana temporal de toda muestra termine en la adquisición que corresponde a `f`.
- Que la ventana sea siempre una secuencia cronológica.
- Que no se construya una muestra con una imagen que no representa a `f`.
- Que el inventario de adquisiciones por serie no esté truncado por un tope de consulta.
- Que la regresión sea detectable automáticamente, no solo por auditoría manual.

**Non-Goals:**
- No se cambia la arquitectura del modelo, el protocolo de entrenamiento, los hiperparámetros ni el sistema de evaluación de métricas.
- No se unifica el catálogo CDSE con el de GEE: se asume que ambos seguirán discrepando en sub-segundo y se hace el emparejamiento tolerante.
- No se reescribe el formato del paquete de muestra (`mask`, `mask_cems`, `target`, `target_cems`): el loader ya normaliza las máscaras y esa desviación se trata en un change aparte.
- No se cambia el criterio de filtrado de frames sin dato (trata de la granularidad píxel/celda): es otra desviación, con su propio change.

## Decisions

### D1. Emparejar `f` por cercanía temporal, no por comparación de cadenas

**Elegido:** parsear las marcas de tiempo a `datetime` y elegir como frame `f` la adquisición utilizable **más cercana** en el tiempo a `frame_time`, siempre que la diferencia no supere una tolerancia de 2 minutos. A partir de ahí, `f_key` es ese frame y el resto de la ventana se construye en torno a él.

**Por qué:** el desfase observado entre catálogos es de ~1 s y es estructural (truncado vs redondeo), no accidental. Cualquier comparación de cadenas truncadas volverá a fallar en cuanto cambie el origen de una de las dos marcas. La cercanía temporal es además robusta a que la descarga pase a usar otro catálogo.

**Alternativas descartadas:**
- *Alinear el nombre del fichero con el tiempo del manifiesto.* Exigiría que la descarga conociera la lista CDSE y renombrara las descargas; ata el constructor al descargador y no resuelve el caso en que la escena de GEE no exista en CDSE.
- *Emparejar por `scene_id`.* Los identificadores de producto de CDSE y GEE difieren (distinta base de procesamiento), así que no hay correspondencia estable.
- *Redondear la marca en los dos extremos.* Corregiría el caso actual, pero reintroduce el mismo problema en cuanto una de las dos fuentes cambie de precisión.

**Tolerancia de 2 minutos:** las adquisiciones OLCI sobre el mismo AOI están separadas por días, luego una ventana de 2 minutos no puede solapar dos adquisiciones distintas. La regla de "más cercana" hace el emparejamiento unívoco incluso sin ese supuesto.

### D2. Descartar la muestra cuando no hay adquisición de `f`, en vez de marcarla

**Elegido:** si no existe ninguna adquisición utilizable dentro de la tolerancia, la muestra **no se ensambla**; el descarte se cuenta y se informa.

**Por qué:** el spec exige que la ventana termine en `f`. Una muestra sin adquisición de `f` no cumple el contrato, y sus dos componentes derivados (cicatriz y target) estarían anclados a otra imagen. Mantenerla con una marca de calidad obligaría a que cada consumidor del dataset la filtrara, que es exactamente la clase de ambigüedad que este change elimina.

**Alternativa descartada:** conservar la muestra con `frame_matched: false` en el índice. Preserva cobertura, pero deja un subconjunto no conforme dentro del dataset de producción y obliga a propagar el filtro a entrenamiento, evaluación y visualización.

**Consecuencia asumida:** el tamaño del dataset puede reducirse si el inventario de adquisiciones no cubre `f`. Por eso D3 (inventario completo) es requisito previo, no accesorio: primero se completa el inventario y después se descarta lo que de verdad no existe.

### D3. La consulta de imágenes usa la cobertura del AOI, no un punto

**Elegido:** `scripts/download_gee_olci.py` consulta la colección OLCI con la extensión del área de interés —el `bbox_wkt` que ya viaja en el manifiesto CEMS— en lugar del punto del centroide. Además se sube el límite de resultados de la consulta a un valor holgado y se avisa cuando se alcance.

**Por qué:** el inventario descargado no es una versión recortada del catálogo de referencia, sino una versión con **otro criterio de intersección**. El downloader filtra por un punto (`filterBounds(point)`, centroide) y el manifiesto consulta el catálogo CDSE por el polígono del AOI. OLCI se distribuye en tiles fijos y S3A y S3B tienen geometría de órbita distinta, de modo que el centroide puede caer en un hueco de tile de S3B mientras que el AOI sí intersecta tiles de S3B. La evidencia local es concluyente: de las 128 adquisiciones ausentes, **125 son de S3B y solo 3 de S3A**, y el mismo hueco de S3B aparece en series que nunca alcanzan el límite de la consulta (15 de 76 escenas S3B ausentes en series sin tope), lo que descarta el truncamiento como causa principal.

**Alternativas descartadas:**
- *Subir únicamente el límite de resultados.* Explicado arriba: el límite de `toList(60)` solo justifica **una** de las 128 ausencias, porque 13 series lo alcanzan pero la selección no se limita a las escenas más recientes. Se mantiene subirlo, pero como robustez, no como la corrección.
- *Mantener el punto y ampliar el parche de descarga.* No cambia qué escenas devuelve la colección.
- *Adoptar GEE como catálogo de referencia.* Inviabiliza el proyecto: el manifiesto CEMS define rondas, etiquetas y productos vectoriales de Copernicus EMSR, y CDSE es donde están.

**Efecto sobre el dataset:** descargar la escena S3B no garantiza que su parche sea utilizable. Si el centroide cae en el hueco de tile de S3B, el parche de 19,2 km centrado en él llega casi sin dato y el filtro de píxeles sin dato lo descarta. En ese caso el descarte de D2 es el comportamiento correcto, no una pérdida evitable: la adquisición de `f` no es observable dentro de la ventana del dataset, y sustituirla por una imagen de otro día (que es lo que hace el código hoy) es peor que no tenerla.

### D4. Composición de la ventana en ramas mutuamente excluyentes

**Elegido:** construir la ventana con dos ramas, ambas por construcción cronológicas:

- Si existen adquisiciones post-incendio utilizables: `[pre_fire] + post[-6:]`, donde `pre_fire` es la adquisición utilizable más reciente anterior a la primera ronda CEMS y `post` son las adquisiciones utilizables entre ese corte y `f` (en orden ascendente, la última es `f`).
- Si no hay adquisiciones post-incendio utilizables: las últimas siete adquisiciones utilizables hasta `f` (`past[-7:]`), rellenando con el frame más antiguo.

**Por qué:** la rama actual mezcla pre y post cuando no hay frames post disponibles, y produce secuencias que van hacia atrás (36/390 muestras actuales). Con la regla "el pre-fire entra solo si es estrictamente anterior al primer post", las dos ramas son ordenadas por construcción y no hace falta una comprobación posterior.

**Alternativa descartada:** componer la lista y luego ordenarla. Ordenar barajaría el frame pre-incendio fuera de la primera posición, que es lo que el spec fija.

### D5. Verificación en el propio informe del ensamblaje
**Elegido:** el resumen del ensamblaje imprime, además del recuento de muestras, tres cifras de control: ventanas que terminan en `f`, ventanas ordenadas cronológicamente y muestras descartadas por falta de adquisición de `f`.

**Por qué:** la desviación que motiva este change era invisible en los informes: el dataset se genera, se entrena y se publica sin que nada delate que la ventana no terminaba en `f`. Un contador que se lee en cada ejecución convierte el invariante en algo auditable a ojo, y el propio informe sirve de criterio de aceptación de la regeneración.

### D6. La lógica de ventana vive en el paquete, no en el script

**Elegido:** extraer el emparejamiento de `f` y la composición de la ventana a funciones puras en `src/burn_scar/dataset.py` (módulo ya importable desde los tests), dejando `scripts/build_pytorch_dataset.py` como orquestador que las invoca.

**Por qué:** la lógica que decide qué frames entran en la ventana es la que falló, y hoy vive en el cuerpo de un `for` anidado dentro de un script con acceso a GPU y a datos en disco: no se puede testear sin dataset. Como funciones puras, se testean con listas de marcas de tiempo sintéticas, sin GPU y sin datos, que es la única forma de que la regresión quede cubierta por la suite.

**Alternativa descartada:** testear el script completo sobre el dataset real. Requiere GPU, TMB-S3 y el dataset presente; el test se saltaría en la mayoría de los entornos y no detectaría el caso del desfase de un segundo de forma aislada.

## Risks / Trade-offs

- **El dataset puede reducirse** → el descarte de D2 elimina muestras cuya adquisición de `f` no está disponible. Mitigación: D3 recupera las escenas que el catálogo de referencia lista y la consulta por punto no expone; el resto corresponde a adquisiciones no observables en la ventana del dataset, cuyo descarte es correcto. El recuento de descartes queda publicado en el informe para que la diferencia entre ambos casos sea visible.
- **La consulta por AOI descarga escenas que el parche no cubre** → su parche llegará casi sin dato y el filtro de píxeles sin dato las descartará. Mitigación: es el comportamiento correcto (D3 y D2); el coste es tráfico de descarga innecesario, aceptable frente a perder la adquisición de `f`.
- **Subir el tope de la consulta encarece la descarga** → más imágenes por serie que las 60 actuales, y además las escenas de S3B que la consulta por punto excluía. Mitigación: el rango temporal ya está acotado (14+1 días); el límite queda holgado frente al máximo observado y el coste es una llamada de listado por serie.
- **Los números publicados quedan obsoletos** → el dataset cambia y con él las métricas. Mitigación: republicar el informe completo (CV, test al umbral congelado, baselines al mismo umbral) y marcar explícitamente las cifras anteriores como sustituidas.
- **La tolerancia de 2 minutos podría emparejar `f` con otra adquisición** → solo si dos adquisiciones del mismo AOI caen dentro de esa ventana. Mitigación: la regla de "más cercana" desambigua y las adquisiciones del mismo AOI están separadas por días.
- **El emparejamiento tolerante oculta un futuro desajuste real de catálogos** → si un día los catálogos divergen en horas, la tolerancia no lo detectaría. Mitigación: la cercanía se calcula y se registra; una diferencia grande queda visible en el informe de control de D5.

## Migration Plan

1. Descargar el inventario completo de adquisiciones con la consulta por AOI (D3) y verificar, serie a serie, cuántas adquisiciones del catálogo de referencia quedan disponibles y cuántas siguen sin cobertura observable en el parche del centroide.
2. Aplicar D1, D2 y D4 en el ensamblador y regenerar el dataset en modo `scar` y en modo `cems`.
3. Comprobar los contadores de D5: ventanas que terminan en `f` = total, ventanas ordenadas = total, descartes explicados.
4. Reentrenar los 5 folds y reevaluar el conjunto de test con el umbral operativo congelado.
5. Recalcular los baselines al mismo umbral operativo y actualizar `docs/operacion.md` y `docs/decisiones.md` con las cifras nuevas, marcando las anteriores como sustituidas.
6. Rollback: el dataset, los checkpoints y la documentación anteriores son regenerables; no hay migración de estado en base de datos ni esquema de datos incompatible.

## Open Questions

Ninguna. La tolerancia, el descarte y la rama de composición quedan fijados por D1, D2 y D4, y su aceptación es verificable con los contadores de D5.
