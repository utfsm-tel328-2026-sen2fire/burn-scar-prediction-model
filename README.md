# burn-scar-prediction-model

Predicción multi-horizonte del crecimiento de cicatrices de incendio forestal en el
Mediterráneo, construida íntegramente con datos satelitales y climáticos abiertos.

## Qué es este proyecto

El objetivo es un modelo que, observando la situación actual de un incendio (imágenes
de satélite + clima + terreno), prediga **qué zonas se quemarán en las próximas horas**
y con qué probabilidad.

- **Entrada**: Sentinel-3 OLCI (imagen multiespectral, revisita ~2 días), Sentinel-2 (línea
  base pre-incendio, 20 m), ERA5 (meteo observada) + GFS/ECMWF (meteo prevista) +
  FIRMS/VIIRS (fuego activo), Copernicus DEM (topografía) y ESA WorldCover (combustible).
  Sentinel-1 (radar) entra en la v2 (`docs/decisiones.md` §1.3 BL5).
- **Salida**: probabilidad de quema por píxel, para un **horizonte explícito** en días
  (1, 3, 7 días): el modelo recibe *h* como entrada y devuelve el mapa a ese horizonte.
- **Etiquetas**: delineaciones de quemado fechadas del Copernicus Emergency Management
  Service (CEMS), que encadenadas dan la curva de crecimiento real de cada incendio.
- **Alcance**: región mediterránea (más 9 series no mediterráneas de las 46, `docs/decisiones.md`
  §1.3 BL6). Resolución de trabajo 300 m.
- **Modelo**: encoder ConvLSTM + SegFormer + una cabeza de propagación. La cicatriz de
  quemado ("¿dónde está quemado ahora?") la genera **TMB-S3 congelado** — un modelo ya
  publicado para el mismo sensor y la misma resolución (sus etiquetas también son CEMS, pero
  con la regla OR y solo el perímetro final: ver `docs/operacion.md` §4) —, y es
  idéntico en entrenamiento y operación. Eso es lo que hace que el pipeline pueda
  funcionar sin CEMS.
- **Cadencia del producto**: cada ~2-3 días, la que marca la revisita de Sentinel-3.

> La cicatriz de quemadura en sí no "crece" cuando el fuego se apaga: lo que crece es
> el fuego. Por eso la tarea es **predicción de propagación**, no delineación. Ver
> [`docs/investigacion.md`](docs/investigacion.md) §2.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/README.md`](docs/README.md) | Índice de la documentación y mapa de qué dato vive en qué fichero |
| [`docs/decisiones.md`](docs/decisiones.md) | Decisiones tomadas y pendientes, riesgos, preguntas abiertas |
| [`docs/dataset.md`](docs/dataset.md) | Especificación del dataset: qué es una muestra, qué es el target, reglas y protocolo de evaluación |
| [`docs/operacion.md`](docs/operacion.md) | Qué necesita el modelo en inferencia: entradas, latencias, ciclo operativo |
| [`docs/investigacion.md`](docs/investigacion.md) | Estado del arte, por qué 300 m, cómo verificar los números |
| [`docs/datos.md`](docs/datos.md) | Catálogo de fuentes: dónde está cada dato, IDs exactos, endpoints, licencias |
| [`docs/experiment-tracking.md`](docs/experiment-tracking.md) | Guía de TensorBoard compartido, checkpoints y sincronización NFS |

## Decisiones clave

| Decisión | Elección | Motivo |
|---|---|---|
| Entrada del modelo | Imágenes libres + clima y terreno libres | Sin viento el modelo no puede aprender propagación, solo extrapolar espacialmente |
| Salida | Probabilidad de quema por píxel, multi-horizonte | Es la formulación que da el crecimiento; el perímetro final se deriva de ella |
| Región | Mediterráneo | Mejor calidad de etiquetas (CEMS) y régimen de fuego exigente |
| Resolución | 300 m | Banda de resolución útil (300-375 m): el techo de la frecuencia con datos abiertos |
| Etiquetas | CEMS EMSR, delineaciones fechadas | Es lo más próximo a una serie temporal real del crecimiento, gratis y global |
| Etiquetado | Cuantificado (fracción quemada por píxel) | Un binario en el borde es ruido; una fracción lo convierte en información |
| Horizonte | Variable y **explícito** (entrada del modelo) | CEMS no puede enseñar horizontes: sus rondas son irregulares. MCD64A1 sí, y el producto final exige consultar el modelo a un *h* concreto |
| Entrenamiento | Baselines B0/B1/B2 antes de la red | Si no supera la advección por viento, no está aprendiendo propagación |
| Modelo | Encoder + 1 cabeza de propagación, con **TMB-S3 congelado** como generador de cicatriz | CEMS no existe en operación; TMB-S3 es la opción (c) de `docs/operacion.md` §3 sin el coste de entrenar un segundo modelo |
| Horizonte | Anclado en la última observación | Anclarlo en la hora del evento acreditaría tiempo en el que el fuego ya quemaba |
| Muestra | Una por frame OLCI, no por ronda CEMS | Reproduce la cadencia de operación (actualiza cada ~2-3 días) y hace que el arranque en frío aparezca solo. El nº exacto de muestras está sin medir (`docs/decisiones.md` P7) |
| Cicatriz de entrada | La salida de TMB-S3, no una ronda de CEMS | Es un modelo: solo ve los frames hasta *f*, así que no puede filtrar quemado futuro, y el mecanismo es idéntico en las tres fases |
| Target | `polígono(r+1) − scar_TMB3(f)` | Medir desde la ronda anterior cuenta quemado **ya visible** en la imagen (unos 2/3 del incremento entre rondas) |
| Preentrenamiento | MCD64A1 como **supervisor**, no como entrada | En operación no existe (6 semanas de retraso), pero como supervisor da millones de etiquetas baratas. Nunca entra en inferencia |
| **Dataset CEMS** | **46 series** (se excluyen las 50 que TMB-S3 se sabe de memoria) | En esas 50 la cicatriz de entrada es el perímetro CEMS memorizado y el target significa "crecimiento entre rondas" en vez de "después de *f*". Auditoría ejecutada: 62 de las 96 series están en TMB-S3 |
| **Reparto** | **CV agrupada de 5 folds**, no split fijo | Un reparto 60/20/20 sobre 46 series evalúa 9 y desperdicia 37. La CV evalúa las 46, cada una una vez |
| **Ventana** | **64×64 px a 300 m (19,2 km)**, centrada en la quemadura | No es una elección: es la ventana de inferencia de TMB-S3 (tiles de 84×83 px, stride 32). La cicatriz y la entrada tienen que ser el mismo patch |
| **Periodo del preentrenamiento** | **2018-2026** | OLCI no existe antes de abril 2016 (S3A), y S3B entra en mayo 2018: antes de eso solo hay una geometría de observación |
| **Dos etapas** | **M-A** (solo preentrenamiento) y **M-B** (preentrenamiento + ajuste CEMS) | El ajuste es una inicialización, así que se pueden reportar los dos. Convierte el conjunto pequeño de CEMS en el diseño del experimento y da salida de emergencia |
| **Evaluación** | **Siempre contra CEMS**, nunca MCD64A1 ni FireCCI | CEMS lo dibujaron analistas mirando imágenes **distintas** de las del modelo en el frente (el S2 del modelo es solo línea base pre-incendio); con FireCCI S3.1.1 el modelo podría limitarse a replicar su clasificador |
| Evaluación | Por evento, con bootstrap, estratificada por sensor | Con 46 series, el promedio por píxel da demasiado peso a los AOIs grandes |

## Estado

**Fase de diseño.** No hay implementación del modelo todavía: lo que hay es el entorno y las
utilidades de experimento (`src/`, `scripts/`, `tests/`). El dataset está diseñado y medido
(ver `docs/dataset.md` §2), con las reglas de construcción fijadas (`docs/dataset.md` §7), la
arquitectura de tres fuentes definida (`docs/dataset.md` §8) y la **auditoría de splits ya
ejecutada** (`docs/decisiones.md` §3).

## Estructura

```
.
├── README.md
├── docs/
│   ├── README.md           # índice y mapa de la documentación
│   ├── decisiones.md       # decisiones, riesgos y preguntas abiertas
│   ├── dataset.md          # especificación del dataset y protocolo de evaluación
│   ├── operacion.md        # inferencia: entradas, latencias, ciclo
│   ├── investigacion.md    # estado del arte y por qué 300 m
│   ├── datos.md            # catálogo de datos y licencias
│   └── experiment-tracking.md  # TensorBoard compartido, checkpoints, NFS
├── src/burn_scar/         # utilidades de tracking (el modelo todavía no)
├── scripts/               # auditoría de splits, entorno, launcher de TensorBoard
├── tests/
├── openspec/              # flujo de trabajo OpenSpec (specs de entorno y tracking)
├── pyproject.toml
└── .env.example
```

## Próximos pasos

1. **Decidir las 7 decisiones que bloquean la descarga** (`docs/decisiones.md` §1.3): ventana
   temporal, checkpoint de TMB-S3, alcance del pool de preentrenamiento, definición de
   "evento" en MCD64A1, si entra Sentinel-1, qué series no mediterráneas se quedan y la
   máscara de nubes. Son baratas de decidir y caras de corregir **después** de descargar.
2. Construir la ingesta: CEMS vía API → cubos GEE → etiquetado cuantificado, aplicando las
   reglas D1-D18 de `docs/dataset.md` §7, en tiles de 64×64 px a 300 m.
3. Implementar los baselines B0/B1/B2 con el protocolo de evaluación ya fijado.
4. **Preentrenar con MCD64A1** (M-A) y luego **ajustar con las 46 series de CEMS** (M-B),
   en 5 folds agrupados. La cicatriz de entrada la produce TMB-S3 en todos los pasos.
5. Comparar M-A y M-B contra los baselines y estratificar los resultados por sensor.
6. Ampliar el dataset (scraper pre-2023, FireCCI51 como supervisor a 250 m).

> El script de auditoría ya está hecho y su resultado está en `docs/decisiones.md` §3:
> ```
> python3 scripts/audit_tmb_splits.py
> ```

## Cómo citar los datos

Cada fuente tiene su norma de licencia y atribución. Están recogidas en
[`docs/datos.md`](docs/datos.md) §7 — conviene revisarlas antes de
distribuir cualquier dataset derivado.
