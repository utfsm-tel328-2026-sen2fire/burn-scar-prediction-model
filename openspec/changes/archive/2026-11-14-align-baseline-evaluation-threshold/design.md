# Design: Umbral de evaluación comparable entre modelo y baselines

## Context

Véase `proposal.md` para la motivación. Hechos medidos y restricciones que condicionan el
diseño:

- `compute_metrics_numpy` aplica **el mismo umbral a la predicción y a la verdad**
  (`src/burn_scar/baselines.py:42-43`). Bajar el umbral de 0,50 a 0,10 no cambia la
  predicción de un baseline binario, pero hace que la verdad incluya como quemados los
  píxeles con algo de superficie quemada, lo que además cambia el conjunto de muestras
  evaluables (de 54/78 a 76/78 contra la etiqueta de referencia).
- Los tres baselines predicen binario: B0 devuelve ceros, el trivial devuelve unos y B1
  devuelve una dilatación binaria (`src/burn_scar/baselines.py:127,146,171`).
- El umbral operativo se persiste junto a los checkpoints en `operational_threshold.json`,
  con la clave `operational_threshold` (`scripts/train_propagation.py:675-691`).
- `evaluate_baselines.py` evalúa hoy con un 0,50 fijo incrustado en el código
  (`scripts/evaluate_baselines.py:49,56`), sin opción de cambiarlo.
- El modelo se evalúa al umbral operativo congelado, que en el run actual es 0,10.

## Goals / Non-Goals

**Goals:**
- Poder medir los baselines al mismo umbral que el modelo, sin recalcular nada.
- Que el informe diga siempre con qué umbral y con qué subconjunto de muestras se ha medido.
- Que la comparación entre modelo y baselines sea reproducible con los scripts del repo.

**Non-Goals:**
- Cambiar la definición de la métrica binaria o el criterio de agregación por evento.
- Reentrenar, regenerar el dataset o reevaluar el modelo: los números del modelo no cambian.
- Cambiar el umbral por defecto de `evaluate_baselines.py`: sigue siendo 0,50 si no se pide
  otra cosa, para no alterar el comportamiento de una herramienta ya en uso.
- Decidir si el umbral operativo debería ser otro (es la pregunta abierta del diagnóstico).

## Decisions

### Decisión 1: El umbral es un parámetro de la evaluación, no una constante
- **Elección**: `evaluate_baselines.py` acepta `--threshold` y `--operational-threshold-from`;
  si no se indica ninguno, usa 0,50 como hasta ahora.
- **Razón**: el umbral fija la discretización de la verdad, y por tanto la definición de lo
  que se está midiendo; ocultarlo en una constante hace imposible una comparación justa.
- **Alternativas consideradas**: (a) cambiar el valor por defecto a 0,10 — descartado porque
  rompe el comportamiento en uso y ata la herramienta al run más reciente; (b) evaluar
  siempre en los dos umbrales — descartado por ahora: duplica el informe y no hace falta
  para el uso real, que es comparar contra un run concreto.

### Decisión 2: El umbral operativo se lee del run, no se escribe a mano
- **Elección**: `--operational-threshold-from CHECKPOINT_DIR` lee `operational_threshold.json`
  de ese directorio y usa su `operational_threshold`.
- **Razón**: el umbral congelado es el resultado de la validación cruzada; escribirlo a mano
  en la línea de comandos es la forma más fácil de comparar contra el número equivocado.
  Leerlo del run garantiza que modelo y baselines usan el mismo valor.
- **Alternativas consideradas**: (a) pedir el valor a mano — descartado por lo anterior;
  (b) buscar el run más reciente automáticamente — descartado: introduce un estado
  implícito y puede elegir el run equivocado.

### Decisión 3: Si se indican las dos cosas, se falla
- **Elección**: `--threshold` explícito y `--operational-threshold-from` a la vez son un
  error con un mensaje que explica cuál usar.
- **Razón**: el umbral es parte de la definición de la métrica; un conflicto silencioso
  produciría un número sin explicación.
- **Alternativas consideradas**: priorizar el explícito — descartado porque el error es
  más barato que un resultado que nadie sabe interpretar.

### Decisión 4: El informe declara umbral y origen
- **Elección**: la cabecera del informe de baselines imprime el umbral y su procedencia
  ("explícito" o "umbral operativo del run `<dir>`").
- **Razón**: es el mismo criterio que ya se aplica al informe del modelo, y es lo que
  permite leer un número suelto sin contexto.

## Risks / Trade-offs

- **[Los baselines a 0,50 y a 0,10 no son el mismo número]** → Se documenta que la
  comparación exige el mismo umbral, y el informe lo declara siempre. El cambio de las
  cifras ya está aplicado en `docs/operacion.md`.
- **[El umbral cambia el subconjunto de muestras evaluables]** → El informe ya informa del
  recuento de evaluables, de modo que el número va siempre acompañado del tamaño del
  subconjunto sobre el que se calcula.
- **[Un umbral congelado de un run antiguo no encaja con un dataset distinto]** → El informe
  nombra el run del que procede el umbral, de modo que la correspondencia es visible.

## Migration Plan

1. Añadir la resolución del umbral y sus opciones al script de baselines.
2. Cubrir la resolución con pruebas unitarias.
3. Actualizar la documentación del diagnóstico con los comandos y las cifras al mismo umbral.
4. Reejecutar la evaluación de baselines contra el run de producción para comprobar que el
   informe nuevo reproduce las cifras de la tabla.

**Rollback**: revertir el commit; el comportamiento por defecto (0,50) se conserva, así que
revertir no deja nada a medias.

## Open Questions

- Ninguna que afecte a los specs o al enfoque. Si en el futuro se decide que el umbral
  operativo pase a leerse siempre desde el run por defecto, sería otro change.
