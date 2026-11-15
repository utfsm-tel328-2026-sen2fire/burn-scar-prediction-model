# Proposal: Alineación de los specs con la implementación

## Why

La revisión de `openspec/specs/` detectó **8 defectos**: cuatro son contradicciones
(tres dentro de los propios specs y una entre capacidades) y cuatro son requisitos que
el código no cumple o que los specs no especifican con precisión. Dos de ellos son
peligrosos para quien implemente a partir de los specs:

- El spec de dataset define **de dos formas incompatibles** qué es "el target de
  propagación", según el modo de ensamblaje, sin calificar la primera.
- La máscara de validez usa **dos codificaciones distintas** (`255` = ignorar en el
  ensamblaje por defecto, `0` = ignorar en el modo de crecimiento de referencia), lo que
  hace que la visualización deje de atenuar los píxeles ignorados en ese modo mientras
  las métricas sí los excluyen.

Además, el spec de evaluación exige informar del "número de muestras y de series
evaluables" y el código solo produce el recuento de series, y el spec de visualización
ordena y promedia por F1 sin tener en cuenta la semántica NaN que el propio spec de
evaluación establece. Todo esto se corrige antes de que un nuevo change se apoye en los
specs como contrato.

## What Changes

- **Un único significado de "target de propagación"**: el requisito "Non-degenerate
  propagation target" pasa a distinguir explícitamente el modo por defecto (`scar`) del
  modo de crecimiento de referencia (`cems`), eliminando la definición doble.
- **Fin de la contradicción del umbral**: el escenario de selección de pesos deja de
  exigir que las métricas registradas reflejen el F1 del umbral operativo (que solo
  existe al terminar la CV) y pasa a declarar que el checkpoint conserva las métricas de
  su fold en el umbral óptimo de ese fold. Se añade que el umbral operativo no interviene
  en la selección de checkpoint ni de época.
- **Codificación canónica de la máscara de validez**: `1` = píxel evaluable, `255` = píxel
  ignorado, para la máscara del target de ensamblaje y para la validez común entre rondas.
  El ensamblaje en modo `cems` deja de emitir `0` como marcador de ignorado.
- **Precisión de "píxel sin dato"**: se separan los dos conceptos que el spec mezclaba
  (píxel sin dato válido en ninguna banda, para el filtro de adquisiciones, frente a
  celda de banda ausente, para la imputación) y se documenta que la imputación es por
  banda, de modo que un píxel parcialmente observado conserva sus bandas válidas.
- **Fallback de la ventana temporal**: el caso sin adquisición pre-incendio utilizable se
  describe como "las últimas siete adquisiciones utilizables disponibles", con el mismo
  relleno de frames del caso general.
- **Precedencia de aceleradores en la capa Nix**: se documenta el orden NVIDIA > AMD > CPU
  cuando la máquina expone device nodes de varios aceleradores.
- **Recuentos de muestras evaluables**: los informes de evaluación pasan a informar del
  número de muestras evaluables (informativas) y del total, por cada etiqueta, que es lo
  que el spec ya exigía y el código no producía.
- **Visualización coherente con la semántica NaN**: el ranking `best`/`worst` y el promedio
  de F1 del fold excluyen las muestras no informativas, el informe indica cuántas se han
  excluido, y la detección de muestras con quemado usa el umbral de evaluación en lugar de
  0,5 fijo.

## Capabilities

### New Capabilities
*(Ninguna.)*

### Modified Capabilities
- `propagation-dataset`: definición única del target de propagación según modo,
  codificación canónica de la máscara de validez, precisión de la granularidad de "píxel
  sin dato" y del fallback de la ventana temporal.
- `propagation-evaluation`: fin de la contradicción del escenario de umbral y recuentos de
  muestras evaluables por etiqueta.
- `prediction-visualization`: tratamiento de las muestras no informativas (NaN) en el
  ranking y en el promedio del fold.
- `nix-development-environment`: precedencia documentada cuando hay device nodes de varios
  aceleradores.

## Impact

- **Archivos modificados**: `scripts/build_pytorch_dataset.py` (codificación de la validez
  común y de la máscara sin ronda previa), `src/burn_scar/dataset.py` (normalización de la
  máscara de referencia al cargar), `src/burn_scar/baselines.py` (recuentos de muestras
  evaluables), `scripts/train_propagation.py` (propagación de los recuentos a los informes),
  `src/burn_scar/visualization.py` (umbral de evaluación en el cómputo de píxeles quemados),
  `scripts/visualize_predictions.py` (promedio y ranking sin NaN).
- **Pruebas**: `tests/test_dataset.py`, `tests/test_visualization.py` y
  `tests/test_propagation.py` se amplían para cubrir la codificación de la máscara y el
  tratamiento de las métricas no informativas.
- **Datos**: el dataset de diagnóstico (`--target-mode cems`) debe regenerarse para emitir
  la máscara en la codificación canónica (≈20 s con GPU, local). El dataset de producción
  (`--target-mode scar`) **no cambia**: ya usaba `1`/`255`.
- **Números publicados**: no se invalidan. La máscara del modo por defecto no cambia, la
  selección de checkpoints no cambia y los recuentos son informativos adicionales. No hace
  falta reentrenar ni reevaluar el modelo de producción.
- **Sin cambios en el modelo**: ni arquitectura, ni entradas, ni protocolo de
  entrenamiento, ni hiperparámetros.
- **Fuera de alcance**: la decisión de fondo sobre si redefinir el target de producción
  (abierta en el diagnóstico del target) y la medición de la calidad de TMB-S3 frente a
  CEMS (P5 pendiente) siguen siendo decisiones posteriores.
