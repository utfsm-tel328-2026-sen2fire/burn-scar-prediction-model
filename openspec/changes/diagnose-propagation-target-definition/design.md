# Design: Diagnóstico de la definición del target de propagación

## Context

Véase `proposal.md` para la motivación. Restricciones y hechos medidos que condicionan el diseño:

- El target de producción es `clip(CEMS(r+h) − scar_TMB3(f), 0, 1)` (decisión D9 de `decisiones.md`, documentada en `docs/operacion.md` §4). El término `scar_TMB3(f)` es una fotografía de un detector, no un estado del incendio.
- Medido sobre las 390 muestras del dataset actual: la IoU media entre ese target y el crecimiento real `clip(CEMS(r+h) − CEMS(f), 0, 1)` es **0,112**; la superficie mediana a umbral 0,5 es **77 píxeles** frente a **10** del crecimiento real; 358/390 muestras tienen quemado a 0,5 con el target actual y 270/390 con el real.
- Las 390 muestras tienen una ronda CEMS previa o igual al instante de su frame actual, así que el crecimiento real es calculable para todas ellas.
- La ronda CEMS de *f* está a **0,99 días** del frame *f* de mediana (p75 1,98; máximo 16,99) y el horizonte *h* mediano es **0,92 días**: los dos intervalos son comparables pero no idénticos, y en el 44% de las muestras la separación entre la ronda y el frame supera un día.
- La entrada del modelo (`x_scar`) es la cicatriz de TMB-S3 y **no puede cambiar** en este experimento: es la variable que se quiere mantener constante para aislar el efecto de la etiqueta.
- `data/processed/` está en `.gitignore`: cada workstation regenera su dataset en local (≈20 s con GPU).

## Goals / Non-Goals

**Goals:**
- Medir, con el mismo modelo y las mismas entradas, cuánto del desempeño actual es propagación real y cuánto es acertar una etiqueta contaminada.
- Poder reentrenar con el enunciado limpio sin tocar el modelo ni el protocolo, para saber qué F1 da cada definición de la tarea.
- Que ningún número de desempeño pueda publicarse sin el contexto de contra qué etiqueta se midió.
- Retirar la comparación con TMB-S3 upstream como "brecha a cerrar".

**Non-Goals:**
- Redefinir el target de producción. Este change aporta el instrumento y la evidencia; la decisión es posterior.
- Cambiar la definición de la cicatriz de entrada, la normalización de las bandas o la arquitectura.
- Entrenar con el target de referencia y publicar ese modelo como producto.
- Medir la calidad de TMB-S3 contra CEMS sobre nuestras series (el P5 pendiente), que es un experimento aparte.

## Decisions

### Decisión 1: El target de referencia usa la última ronda CEMS conocida en el instante f
- **Elección**: el crecimiento de referencia es `clip(CEMS(r+h) − CEMS(r_f), 0, 1)`, donde `r_f` es la última ronda de monitorización CEMS cuyo tiempo de adquisición es anterior o igual al frame actual f.
- **Razón**: es la única forma de conocer "qué estaba quemado en f" con la misma fuente de verdad que el futuro (CEMS), y por tanto la única que permite que la etiqueta no dependa del detector que genera la entrada. Las 390 muestras la tienen.
- **Alternativas consideradas**: (a) interpolar CEMS entre rondas — descartado, no hay información entre rondas y la interpolación inventaría quemado; (b) usar la ronda CEMS inmediatamente posterior a f — descartado, introduciría información del futuro en la etiqueta; (c) usar `scar_TMB3(f)` — descartado, es el target actual y es justamente lo que se quiere contrastar.

### Decisión 2: El dataset guarda los dos targets y una validez común
- **Elección**: cada `.npz` guarda el target de propagación (`target`, según el modo de ensamblaje), el crecimiento de referencia (`target_cems`) y la validez común a las dos rondas (`mask`), que es la intersección de la validez de la ronda de f y la de la ronda objetivo.
- **Razón**: guardar la referencia permite **reevaluar el modelo actual sin reentrenar**, que es la medición más informativa y la más barata; y la validez común evita que el modelo sea evaluado sobre píxeles que no eran válidos en una de las dos rondas.
- **Alternativas consideradas**: (a) calcular el crecimiento de referencia en tiempo de evaluación, recargando CEMS — descartado, ataría la evaluación al layout de las etiquetas rasterizadas; (b) un dataset aparte solo con la referencia — descartado, duplicaría 32 MB y obligaría a sincronizar dos versiones.

### Decisión 3: El modo por defecto no cambia y el target de producción queda intacto
- **Elección**: el ensamblaje por defecto sigue produciendo el target actual; el modo de crecimiento de referencia es un conmutador explícito y escribe en su propio directorio.
- **Razón**: la definición del target de producción es una decisión de diseño documentada (D9) y este change es un instrumento de diagnóstico, no su revisión. Además, regenerar el dataset de producción con un target distinto invalidaría la comparación con el run ya publicado.
- **Alternativas consideradas**: (a) cambiar el target por defecto y comparar hacia atrás — descartado, mezcla el diagnóstico con el cambio de producto; (b) sustituir el target actual en el dataset de producción — descartado por lo mismo.

### Decisión 4: La validity de la comparación depende de qué se comparen
- **Elección**: el informe distingue dos comparaciones y descarta explícitamente una tercera:
  - **(A) Mismo modelo, dos etiquetas.** El modelo entrenado con el target actual se evalúa contra el target actual y contra el de referencia. Responde: ¿cuánto de lo que el modelo "acierta" es propagación real?
  - **(B) Cada modelo contra su propia etiqueta.** El modelo entrenado con el target actual se mide contra ese target, y el entrenado con la referencia contra la referencia. Responde: ¿qué F1 da cada definición de la tarea?
  - **(C) DESCARTADA.** El modelo entrenado con la referencia no se evalúa contra el target actual ni al revés: son tareas distintas y el número no significa nada.
- **Razón**: los F1 absolutos de etiquetas distintas **no son comparables** entre sí: el target actual tiene 7,7 veces más superficie, así que un predictor trivial ya puntúa alto. Por eso cada F1 se acompaña del **F1 del predictor trivial "todo quemado"** calculado contra esa misma etiqueta, que es la escala mínima que hay que superar. Sin ese trivial, un F1 alto no distingue "aprendió a propagar" de "la etiqueta es grande".
- **Alternativas consideradas**: (a) comparar F1 absolutos entre etiquetas — descartado por lo anterior; (b) usar IoU en lugar de F1 — no, el F1 por evento es la métrica de referencia del proyecto (D10) y el IoU se reporta como acompañamiento.

### Decisión 5: La comparación con TMB-S3 upstream se retira como brecha
- **Elección**: la documentación deja de presentar el F1 de TMB-S3 upstream (69,45 / IoU 59,81) como meta o como distancia que cerrar, y explica que evalúa detección de quemado con etiquetas MCD64A1 y regla OR sobre su propio test set, mientras que el proyecto evalúa pronóstico de crecimiento con etiqueta de incremento CEMS.
- **Razón**: son dos preguntas distintas; usar una como vara de la otra induce a interpretar un número que no significa eso. La vara del proyecto es B1 (10,80%), que usa la misma información disponible que el modelo.
- **Alternativas consideradas**: (a) mantener la comparación con una nota al pie — descartado, el marco "lejos de upstream" persiste aunque se matice; (b) omitir la cifra sin explicar — descartado, el número es real y útil como referencia de la calidad del generador de cicatriz, solo que no como meta.

## Risks / Trade-offs

- **[El crecimiento de referencia no es el crecimiento exacto en `[f, f+h]`]** → Mide entre rondas CEMS; la ronda de *f* está a ~1 día del frame de mediana y el horizonte mediano es 0,92 días. Como instrumento de diagnóstico el desajuste es aceptable; si el diagnóstico obliga a redefinir el target de producción, esa redefinición sí deberá tratar la frontera temporal con cuidado.
- **[Cambia el formato de los `.npz`]** → El dataset local se regenera (≈20 s) y el cargador tolera la ausencia de `target_cems` en muestras antiguas, de modo que un dataset previo sigue siendo utilizable con el reporte doble marcado como "no disponible".
- **[El modo de referencia puede dar un F1 más bajo solo por ser un target más pequeño]** → El trivial "todo quemado" se calcula contra cada etiqueta y da la escala; la comparación válida es (A) y (B) de la Decisión 4, no el contraste absoluto entre etiquetas.
- **[Que el desempeño contra la referencia sea bajo no prueba que el modelo sea peor]** → Solo prueba que el modelo no reproduce esa verdad, que es justo lo que se quería medir.
- **[El diagnóstico puede salir concluyente y obligar a un rediseño grande]** → Es el objetivo. Si obligara a redefinir el target de producción, ese trabajo iría en un change aparte, con su propio análisis de la frontera temporal y de la disponibilidad de CEMS en operación.

## Migration Plan

1. Regenerar el dataset de producción con el target de referencia añadido (el target por defecto no cambia).
2. Reevaluar el checkpoint existente contra la referencia, sin reentrenar.
3. Ensamblar el dataset de diagnóstico en su propio directorio y reentrenar con el mismo protocolo.
4. Redactar la tabla comparativa y la conclusión.

**Rollback**: el target por defecto no ha cambiado, así que basta con volver a la versión anterior del script de ensamblaje y regenerar el dataset; los `.npz` no se versionan y no hay migraciones de datos.

## Open Questions

- Si el diagnóstico muestra que el target de producción debe cambiar, ¿cuál sería la definición correcta **en operación**, donde no existe CEMS? La respuesta más probable es hacer la etiqueta consistente con la entrada (mismo generador en *f* y en el futuro), pero eso es un change propio con su propia evaluación.
- ¿Merece la pena medir la calidad de TMB-S3 contra CEMS sobre nuestras series (el P5 pendiente) para saber si el 0,112 de IoU viene de la cicatriz o de la definición del crecimiento? Es independiente de este change y puede hacerse en paralelo.
