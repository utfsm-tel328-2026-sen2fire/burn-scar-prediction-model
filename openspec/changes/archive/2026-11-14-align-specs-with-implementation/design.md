# Design: Alineación de los specs con la implementación

## Context

Véase `proposal.md` para la motivación. Restricciones observadas en el repositorio que
condicionan el diseño:

- La máscara de validez tiene **tres convenciones distintas** en el mismo proyecto: el
  rasterizador CEMS emite `{1, 255}` (`scripts/rasterize_cems_labels.py:160`), el dataset
  de propagación guarda `mask` con esos mismos valores pero la validez común entre rondas
  con `{0, 1}` (`scripts/build_pytorch_dataset.py:78`), y tanto el cargador como las
  pruebas y la documentación (`src/burn_scar/dataset.py:52`,
  `docs/refinamiento_documentacion.md:152`)
  dan por hecho `{1, 255}` donde `255` significa "ignorar".
- La métrica binaria selecciona los píxeles válidos con `mask == 1`
  (`src/burn_scar/baselines.py:37`), de modo que `0` y `255` se excluyen igual: el
  redondeo actual no rompe las métricas, pero **sí rompe la visualización**, que atenúa
  exclusivamente las zonas con `mask == 255` (`src/burn_scar/visualization.py:83`).
- La agregación por evento ya cuenta series evaluables y series totales
  (`src/burn_scar/baselines.py:105-106`) pero no cuenta muestras, que es lo que el spec de
  evaluación exige.
- El promedio de F1 por muestra del visualizador se calcula con `np.mean` sobre la lista
  completa (`scripts/visualize_predictions.py:172`), de modo que un único NaN lo convierte
  en NaN, en contra de la regla NaN-aware del spec de evaluación.
- El umbral operativo se consolida **después** de los cinco folds
  (`scripts/train_propagation.py:557-570`), mientras que la selección de checkpoint ocurre
  dentro de cada fold (`scripts/train_propagation.py:365-380`): el umbral operativo no
  puede estar disponible durante el entrenamiento.
- `data/processed/` no se versiona: el dataset de producción se regenera en local y el
  dataset de diagnóstico vive en un directorio aparte.

## Goals / Non-Goals

**Goals:**
- Que cada término usado en los specs tenga un único significado, incluido "target de
  propagación" y "píxel sin dato".
- Que la máscara de validez tenga una única codificación en todo el proyecto.
- Que los informes entreguen los recuentos que los specs prometen.
- Que la semántica NaN de la evaluación se respete también en la visualización.
- Que ningún número publicado quede invalidado por el cambio.

**Non-Goals:**
- Redefinir el target de producción (decisión abierta en el diagnóstico del target).
- Cambiar el umbral operativo, el protocolo de entrenamiento o el modelo.
- Tocar el rasterizador CEMS ni el modelo de datos de `data/raw`.
- Regenerar el dataset de producción: su máscara ya cumple la codificación canónica.

## Decisions

### Decisión 1: `255` es el marcador de "ignorar" en todas las máscaras
- **Elección**: el ensamblaje emite `{1, 255}` tanto en la máscara del target de
  ensamblaje como en la validez común entre rondas, y el cargador normaliza cualquier
  valor distinto de `1` a `255`.
- **Razón**: `255` es lo que ya significa "ignorar" en el rasterizador CEMS, en el
  cargador, en la documentación y en la visualización. Fijar `0` obligaría a tocar el
  rasterizador, las pruebas y la documentación, y a reinterpretar una convención
  heredada del formato de etiquetas.
- **Alternativas consideradas**: (a) `0` = ignorar — descartado por lo anterior y porque
  `0` es un valor "falso" que se confunde con un píxel no observado por ser nulo;
  (b) aceptar ambas codificaciones — descartado: deja el defecto abierto y convierte la
  ambigüedad en un contrato.

### Decisión 2: La normalización vive en el cargador, no en un régénero global
- **Elección**: `src/burn_scar/dataset.py` convierte la validez común a `{1, 255}` al
  cargar, en lugar de obligar a regenerar cualquier dataset existente.
- **Razón**: los `.npz` no se versionan y el dataset de producción **no cambia** con este
  trabajo; normalizar en el punto de lectura hace que un dataset antiguo siga siendo
  utilizable y que el de diagnóstico se pueda regenerar cuando quiera.
- **Alternativas consideradas**: regenerar todo — descartado por coste y porque no
  cambiaría el resultado; validar y rechazar datasets viejos — descartado: sería una
  ruptura de compatibilidad sin ningún beneficio.

### Decisión 3: Una muestra sin ronda previa es "no evaluable", no "vacía"
- **Elección**: cuando no hay ronda CEMS anterior, la validez común se emite íntegramente
  como `255` (no evaluable), en lugar de `0`.
- **Razón**: con la codificación canónica, "no evaluable" y "evaluable" quedan distinguibles
  en el paquete; las métricas de esa muestra contra la referencia salen NaN (no
  informativas), que es exactamente lo que exige el spec de evaluación, y la visualización
  atenúa la muestra completa en lugar de mostrar datos que nadie observó.
- **Alternativas consideradas**: emitir `0` — descartado por la Decisión 1; descartar la
  muestra — descartado: el spec exige conservarla.

### Decisión 4: Los recuentos de muestras evaluables se calculan donde ya se ignoran los NaN
- **Elección**: `aggregate_series_metrics` calcula `n_samples_evaluated` (muestras con
  métrica informativa) y `n_samples_total` a la vez que ya cuenta series, y los informes
  los propagan por etiqueta.
- **Razón**: duplicar el filtrado de NaN en cada consumidor reintroduciría el tipo de
  divergencia que este change elimina; la agregación por evento es el único punto donde
  se decide qué es informativo.
- **Alternativas consideradas**: contar en cada script consumidor — descartado por el mismo
  motivo; informar solo de series — descartado: el spec exige ambos recuentos y el número
  de muestras es el que revela el tamaño real del subconjunto evaluable.

### Decisión 5: La visualización hereda la semántica NaN y el umbral de evaluación
- **Elección**: el promedio de F1 por fold se calcula sobre las muestras informativas y el
  informe añade el recuento de informativas sobre el total; la condición de "muestra con
  quemado" se evalúa con el umbral de decisión en uso en lugar de un `0,5` fijo.
- **Razón**: el visualizador reutiliza `compute_metrics_numpy`, que ya devuelve NaN; el
  `0,5` fijo hacía que, con el umbral operativo congelado en `0,10`, la priorización de
  muestras con quemado no coincidiera con las métricas mostradas.
- **Alternativas consideradas**: filtrar las muestras no informativas del conjunto
  completo — descartado: el spec pide informar cuántas se han excluido, no ocultarlas;
  mantener el `0,5` — descartado por la incoherencia con el umbral usado.

### Decisión 6: Las alineaciones de redacción no tocan el código
- **Elección**: la definición única del target de propagación, la corrección del escenario
  del umbral, el fallback de la ventana, la granularidad de "píxel sin dato" y la
  precedencia de aceleradores son cambios **solo de spec**: el código ya hace lo que el
  spec debe decir.
- **Razón**: el código de la ventana temporal y el de la capa Nix coinciden con el
  comportamiento deseado; lo que estaba mal era la descripción. Reescribir el código para
  que encaje con una redacción equivocada habría introducido regresiones en un pipeline ya
  validado.
- **Alternativas consideradas**: reescribir el código para ajustarse a la redacción
  original — descartado: la redacción original era la ambigua.

## Risks / Trade-offs

- **[Datasets antiguos con validez común en `{0, 1}`]** → El cargador normaliza al leer,
  de modo que siguen siendo utilizables; el dataset de diagnóstico se regenera para que el
  paquete en disco sea canónico.
- **[El aspecto de los paneles del modo `cems` cambia]** → Los píxeles ignorados ahora se
  atenúan, que era el objetivo; los paneles del modo por defecto no cambian.
- **[El informe gana una línea y una cifra]** → Quien compare capturas de informe antiguas
  verá una diferencia; es información adicional, no un cambio de métrica.
- **["Evaluable" puede confundirse con "muestra con quemado"]** → El informe nombra el
  recuento como muestras evaluables junto al total, y el spec define evaluable como "métrica
  informativa"; una muestra con quemado en la verdad pero sin píxel válido no es evaluable.
- **[Los números publicados siguen vigentes]** → La máscara del modo por defecto no cambia,
  la selección de checkpoints no cambia y los recuentos son informativos; no hace falta
  reentrenar ni reevaluar el modelo de producción.

## Migration Plan

1. Aplicar los cambios de código (grupos 2 y 3 de `tasks.md`).
2. Regenerar el dataset de diagnóstico (`--target-mode cems`) para emitir la validez común
   en la codificación canónica; el dataset de producción no se toca.
3. Ejecutar la suite completa de pruebas.
4. Opcional: reevaluar el run existente con `--eval-only` para ver los nuevos recuentos de
   muestras evaluables.

**Rollback**: revertir el commit de código. El dataset de producción es idéntico antes y
después; el de diagnóstico se regenera con la versión anterior del script de ensamblaje.

## Open Questions

- Si el recuento de muestras evaluables debe también exponerse como métrica de TensorBoard
  (hoy solo se informa por consola y en el JSON del umbral). Es una decisión de la
  capacidad de tracking, no de estas, y no cambia ningún requisito.
