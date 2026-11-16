# Tasks

## 1. Resolución del umbral de evaluación

- [x] 1.1 En `scripts/evaluate_baselines.py`, añadir una función `resolve_eval_threshold(threshold, run_dir)` que devuelva el umbral de evaluación y su origen, con el umbral operativo congelado leído de `operational_threshold.json` del run indicado. Verificar con pruebas unitarias que con `--threshold` explícito devuelve ese valor, que sin opciones devuelve 0,50 y que con un run devuelve el umbral congelado.
- [x] 1.2 Rechazar con un mensaje explícito la combinación de `--threshold` explícito y `--operational-threshold-from`. Verificar que el error indica cuál de las dos opciones usar.
- [x] 1.3 Propagar el umbral resuelto a la evaluación de B0, B1 y el trivial, contra las dos etiquetas. Verificar que las métricas del informe cambian con el umbral y que el recuento de muestras evaluables acompaña a cada etiqueta.

## 2. Informe

- [x] 2.1 Imprimir en la cabecera del informe el umbral de evaluación usado y su origen (explícito o umbral operativo del run). Verificar que el informe nombra el run cuando el umbral viene de él.
- [x] 2.2 Declarar en el informe que el umbral de una predicción binaria solo afecta a la discretización de la verdad y al conjunto de muestras evaluables. Verificar que el texto aparece en la salida del script.

## 3. Documentación y verificación

- [x] 3.1 En `docs/operacion.md`, actualizar el bloque "Cómo reproducir el diagnóstico" con el comando que evalúa los baselines al umbral operativo del run. Verificar que el comando existe y se ejecuta.
- [x] 3.2 En `docs/operacion.md`, dejar constancia de que el umbral de los baselines por defecto es 0,50 y de que la comparación con el modelo exige el mismo umbral. Verificar que no contradice la tabla del diagnóstico.
- [x] 3.3 Ejecutar la suite completa de pruebas y `ruff` y verificar que no hay errores nuevos ni líneas añadidas por encima de 100 caracteres.
- [x] 3.4 Reejecutar la evaluación de baselines contra el run de producción con el umbral operativo y verificar que reproduce las cifras de la tabla del diagnóstico (B1 15,80% / 6,88% y trivial 16,03% / 7,46% contra ensamblaje y referencia).
