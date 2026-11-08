# burn-scar-prediction-model

Predicción multi-horizonte (a 1, 3 y 7 días) del crecimiento de cicatrices de incendio
forestal en el Mediterráneo, construida íntegramente con datos satelitales y climáticos
abiertos.

## Qué es este proyecto

El objetivo es un modelo que, observando la situación actual de un incendio (imágenes de
satélite + clima + terreno), prediga **qué zonas se quemarán en los próximos días** y con
qué probabilidad.

- **Datos que usa**: imágenes de Sentinel-3 (cada ~2 días), Sentinel-2 (una foto anterior
  al incendio), Sentinel-1 (radar, a partir de la v2), el viento y la temperatura **tanto
  observados como previstos**, los focos de calor activos, el relieve y el tipo de
  vegetación. Todo abierto y gratuito.
- **Lo que devuelve**: un mapa de probabilidad de quema, píxel a píxel, para un horizonte
  que se pide explícitamente (1, 3 o 7 días).
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
(`decisiones.md` §1.1) y **D1-D18** (`dataset.md` §7).

## Estado

**Diseño cerrado, recopilación desbloqueada.** El dataset está diseñado y medido, la
auditoría de splits (PASO 0) está hecha y las siete decisiones de recopilación
(**BL1–BL7**, [`docs/decisiones.md`](docs/decisiones.md) §1.3) han sido cerradas. El siguiente
paso es implementar el pipeline de ingesta y metadatos para las 46 series limpias de CEMS, y en
paralelo cerrar la definición formal de **B2 (P2)** para el criterio de evaluación.

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

1. **Decidir las 7 decisiones que bloquean la descarga**
   ([`docs/decisiones.md`](docs/decisiones.md) §1.3): ventana temporal, checkpoint de
   TMB-S3, alcance del pool de preentrenamiento, definición de "evento" en MCD64A1, si
   entra Sentinel-1, qué series no mediterráneas se quedan y la máscara de nubes. Son
   baratas de decidir y caras de corregir **después** de descargar.
2. Construir la ingesta: CEMS vía API → cubos GEE → etiquetado cuantificado, aplicando las
   reglas D1-D18 ([`docs/dataset.md`](docs/dataset.md) §7), en tiles de 64×64 px a 300 m.
3. Implementar los modelos de referencia B0/B1/B2 con el protocolo de evaluación ya fijado.
4. **Preentrenar con MCD64A1** (M-A) y luego **ajustar con las 46 series de CEMS** (M-B),
   en 5 folds agrupados. La cicatriz de entrada la produce TMB-S3 en todos los pasos.
5. Comparar M-A y M-B contra los modelos de referencia y estratificar los resultados por
   sensor.

> El script de auditoría ya está hecho y su resultado está en
> [`docs/decisiones.md`](docs/decisiones.md) §3:
> ```
> python3 scripts/audit_tmb_splits.py
> ```

## Estructura

```
.
├── README.md
├── docs/          # la documentación (ver docs/README.md)
├── src/burn_scar/ # utilidades de tracking (el modelo todavía no)
├── scripts/       # auditoría de splits, entorno, launcher de TensorBoard
├── tests/
├── openspec/      # flujo de trabajo OpenSpec (specs de entorno y tracking)
├── pyproject.toml
└── .env.example
```

## Cómo citar los datos

Cada fuente tiene su norma de licencia y atribución. Están recogidas en
[`docs/datos.md`](docs/datos.md) §7 — conviene revisarlas antes de distribuir cualquier
dataset derivado.
