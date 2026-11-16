# burn-scar-prediction-model

Predicción multi-horizonte (continua) del crecimiento de cicatrices de incendio
forestal en el Mediterráneo, construida íntegramente con datos satelitales y climáticos
abiertos.

## Qué es este proyecto

El objetivo es un modelo que, observando la situación actual de un incendio (imágenes de
satélite, terreno y cobertura del suelo), prediga **qué zonas se quemarán en los próximos
días** y con qué probabilidad.

- **Datos que usa el modelo actual**: imágenes de Sentinel-3 OLCI (cada ~2 días), Copernicus
  DEM, cobertura del suelo ESA WorldCover, la cicatriz actual estimada por TMB-S3 y el
  horizonte continuo de predicción. Sentinel-1, Sentinel-2 y meteorología no forman parte
  del modelo entrenado.
- **Lo que devuelve**: un mapa de probabilidad de **nueva quema**, píxel a píxel, para un
  horizonte continuo que se pide explícitamente.
- **Con qué se mide**: con las delineaciones del Copernicus Emergency Management Service
  (CEMS), el único dato público que dibuja el crecimiento de un incendio mediante
  analistas mirando imágenes **distintas** de las del modelo.
- **Alcance y resolución**: región mediterránea, 300 m por píxel. El producto se actualiza
  cada ~2-3 días, que es lo que tarda Sentinel-3 en pasar por el mismo sitio.
- **Modelo**: un encoder ConvLSTM + SegFormer con una cabeza de propagación. La cicatriz
  ("¿dónde está quemado ahora?") la genera **TMB-S3 congelado**, un modelo ya publicado
  que usa el mismo sensor y la misma resolución: así la cicatriz se produce igual al
  entrenar y al predecir, y la cadena de datos funciona sin CEMS.

> La cicatriz de quemadura en sí no "crece" cuando el fuego se apaga: lo que crece es el
> fuego. Por eso la tarea es **predicción de propagación**, no delineación. Ver
> [`docs/investigacion.md`](docs/investigacion.md) §2.

## Las seis decisiones que definen el proyecto

| | Decisión | Por qué |
|---|---|---|
| 1 | **La cicatriz la pone TMB-S3, congelado** | CEMS no existe en operación; con un modelo, la cicatriz se genera igual al entrenar y al predecir |
| 2 | **Se audita qué sabe TMB-S3** | Se sabe de memoria 50 de nuestras 96 series: esas series quedan fuera del dataset |
| 3 | **El objetivo se mide desde la última imagen** | Medir desde la ronda anterior contaría quemado que ya se ve en la imagen |
| 4 | **Validación cruzada en 5 bloques** | Con 46 series, un reparto fijo evaluaría 9 y desperdiciaría 37 |
| 5 | **La resolución es 300 m** | Es la banda útil: a más resolución no hay imágenes tan frecuentes |
| 6 | **Se evalúa siempre contra CEMS** | Es el único etiquetado hecho por analistas mirando otras imágenes |

El detalle completo, con las alternativas descartadas y el porqué, está en
[`docs/decisiones.md`](docs/decisiones.md). Estas seis son el resumen de **B1-B13**
(`decisiones.md` §1.1) y **D1-D20** (`dataset.md` §7).

## Estado

**Modelo entrenado y dataset canónico regenerado.** El dataset vigente contiene **295
muestras únicas en 40 series**, con horizontes continuos, inventario de calidad y capas
estáticas validadas. El run canónico es `prop_v4_integrity` (semilla 42), con umbral
operativo congelado en 0,10.

Conjunto de test (fold -1, 25 muestras, 7 series), modelo y baselines al mismo umbral y sobre
el mismo subconjunto:

| Modelo | F1 (ensamblaje) | F1 (referencia, propagación real) |
|---|---:|---:|
| B1 dilatación radial | 15,94% | 7,30% |
| Trivial "todo quemado" | 16,29% | 8,42% |
| **Modelo de producción** (`prop_v4_integrity`) | **36,51%** | **14,80%** |

El número publicable de propagación es el de la **referencia** (14,80%): la etiqueta de
ensamblaje incluye quemado que TMB-S3 no detectó. El arranque en frío **no está
representado** y no se publica un desempeño para él; las bandas de horizonte se reportan solo
como diagnóstico y no filtran nada. La tabla completa, con IoU, precisión, recall, soporte
por banda y recuentos evaluables, está en
[`docs/operacion.md`](docs/operacion.md) §9.

El run anterior `prop_v4` queda como **histórico y no publicable**: su índice tenía entradas
duplicadas y una serie de test con WorldCover vacío, así que sus métricas no son
comparables con las del run canónico.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/README.md`](docs/README.md) | Índice y mapa de qué dato vive en qué fichero |
| [`docs/decisiones.md`](docs/decisiones.md) | Decisiones tomadas y pendientes, riesgos, preguntas abiertas |
| [`docs/dataset.md`](docs/dataset.md) | Qué es una muestra, qué se predice, reglas de construcción y protocolo de evaluación |
| [`docs/operacion.md`](docs/operacion.md) | Qué necesita el modelo en inferencia: entradas, retrasos, ciclo operativo |
| [`docs/investigacion.md`](docs/investigacion.md) | Estado del arte, por qué 300 m, cómo verificar los números |
| [`docs/datos.md`](docs/datos.md) | Catálogo de fuentes: dónde está cada dato, IDs exactos, endpoints, licencias |
| [`docs/experiment-tracking.md`](docs/experiment-tracking.md) | Guía de TensorBoard compartido, checkpoints y sincronización NFS |
| [`docs/nixos.md`](docs/nixos.md) | Cómo trabajar el proyecto en NixOS con devenv (opcional) |

> ¿No sabes qué es una AOI, una delineación o el NBR? → [Glosario](docs/dataset.md#1-glosario).

## Próximos pasos

1. Ampliar el conjunto y recuperar `EMSR785_AOI01` con una fuente estática que no devuelva
   WorldCover vacío.
2. Obtener observaciones genuinas de arranque en frío o mantenerlo explícitamente fuera del
   alcance de desempeño.
3. Implementar B2 (advección por viento) y comparar contra el mismo conjunto y umbral.
4. Evaluar la calibración de las salidas y la estabilidad del umbral por fold y horizonte.
5. Ampliar el soporte de la cola de horizontes antes de dar significado a más de tres días.

La auditoría reproducible del dataset y la procedencia de los runs están en
`scripts/audit_dataset_integrity.py` y en `operational_threshold.json`, respectivamente.

## Estructura

```
.
├── README.md
├── docs/          # la documentación (ver docs/README.md)
├── src/burn_scar/ # modelo, dataset, métricas y utilidades compartidas
├── scripts/       # ingesta, ensamblado, entrenamiento, evaluación y auditoría
├── tests/
├── openspec/      # flujo de trabajo OpenSpec (specs de entorno y tracking)
├── pyproject.toml
└── .env.example
```

## Cómo citar los datos

Cada fuente tiene su norma de licencia y atribución. Están recogidas en
[`docs/datos.md`](docs/datos.md) §7 — conviene revisarlas antes de distribuir cualquier
dataset derivado.
