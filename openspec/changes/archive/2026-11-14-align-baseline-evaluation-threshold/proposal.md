# Proposal: Umbral de evaluación comparable entre modelo y baselines

## Why

La reevaluación del diagnóstico (con el contador de muestras evaluables añadido en
`align-specs-with-implementation`) puso al descubierto que **la comparación entre el modelo y
los baselines no era justa**: los baselines se miden a umbral fijo 0,50 y el modelo al
umbral operativo congelado 0,10.

El umbral de `compute_metrics_numpy` se aplica **a la vez** a la predicción y a la verdad, y
eso tiene dos efectos distintos según quién prediga:

- En el **modelo**, el umbral es una regla de decisión sobre probabilidades.
- En los **baselines** (B0, B1 y el trivial), la predicción es binaria (0,0 o 1,0), así que
  el umbral no cambia la predicción: solo cambia **la discretización de la verdad** y, con
  ella, **qué muestras son medibles**.

Consecuencia medida sobre el conjunto de test (fold -1, 78 muestras, 8 series):

| Etiqueta | Umbral 0,50 | Umbral 0,10 |
|---|---|---|
| De ensamblaje | 78/78 evaluables | 78/78 evaluables |
| De referencia | **54/78** evaluables | **76/78** evaluables |

Y por tanto los números que se comparaban no eran el mismo número:

| | Ensamblaje | Referencia |
|---|---|---|
| B1 a 0,50 (54/78 muestras) | 12,00% | 3,05% |
| B1 a 0,10 (76/78 muestras) | 15,80% | 6,88% |
| Trivial a 0,50 (54/78 muestras) | 11,34% | 3,17% |
| Trivial a 0,10 (76/78 muestras) | 16,03% | 7,46% |
| Modelo a 0,10 (76/78 muestras) | 33,80% | 12,41% |

La documentación afirmaba que el modelo sacaba "cuatro veces el trivial" contra el
crecimiento real; comparando al mismo umbral y sobre las mismas muestras, la razón es
**1,66×** (12,41% frente a 7,46%) y la vara real es ~7,5%, no ~3%. La conclusión "el modelo
sí aprende propagación" se sostiene, pero la magnitud estaba inflada por una comparación
mal planteada.

## What Changes

- `scripts/evaluate_baselines.py` acepta el **umbral de evaluación** con el que se miden los
  baselines, y puede leerlo del **umbral operativo congelado de un run** ya entrenado, de
  modo que la comparación con el modelo use el mismo umbral, la misma discretización de la
  verdad y el mismo subconjunto de muestras evaluables.
- El informe de baselines declara **qué umbral se ha usado y de dónde sale**, para que un
  número no pueda leerse sin ese contexto.
- Se documenta que, para predicciones binarias, el umbral no es una regla de decisión sino
  la discretización de la verdad, y que por eso los recuentos de muestras evaluables son
  parte del número.
- Se corrige la documentación del diagnóstico con las cifras medidas al mismo umbral.

## Capabilities

### New Capabilities
*(Ninguna.)*

### Modified Capabilities
- `propagation-evaluation`: la evaluación de los baselines de referencia SHALL poder usarse
  al mismo umbral que el modelo (incluido el umbral operativo congelado de un run) y SHALL
  declarar el umbral empleado y su origen en el informe.

## Impact

- **Archivos modificados**: `scripts/evaluate_baselines.py` (opción de umbral, lectura del
  umbral operativo del run, declaración en el informe), `docs/operacion.md` (comandos de
  reproducción y cifras del diagnóstico al mismo umbral).
- **Pruebas**: `tests/test_baselines.py` (resolución del umbral de evaluación).
- **Datos y checkpoints**: no se tocan. No hace falta reentrenar ni regenerar nada.
- **Números**: los que se publican no cambian (el diagnóstico ya se ha reevaluado con el
  código actual); lo que cambia es la comparación entre modelo y baselines, que pasa a
  estar medida a un mismo umbral.
- **Sin cambios en el modelo**: ni arquitectura, ni entradas, ni protocolo de entrenamiento,
  ni hiperparámetros.
