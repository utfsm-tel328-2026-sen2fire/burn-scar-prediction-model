# burn-scar-prediction-model

Predicción multi-horizonte del crecimiento de cicatrices de incendio forestal en el
Mediterráneo, construida íntegramente con datos satelitales y climáticos abiertos.

## Qué es este proyecto

El objetivo es un modelo que, observando la situación actual de un incendio (imágenes
de satélite + clima + terreno), prediga **qué zonas se quemarán en las próximas horas**
y con qué probabilidad.

- **Entrada**: Sentinel-3 OLCI (imagen multiespectral diaria), Sentinel-1 (radar),
  Sentinel-2 (línea base pre-incendio), ERA5 (viento, temperatura, humedad),
  Copernicus DEM (topografía) y ESA WorldCover (combustible).
- **Salida**: probabilidad de quema por píxel, para un horizonte variable (la siguiente
  ronda de monitorización del incendio).
- **Etiquetas**: delineaciones de quemado fechadas del Copernicus Emergency Management
  Service (CEMS), que encadenadas dan la curva de crecimiento real de cada incendio.
- **Alcance**: UE, región mediterránea. Resolución de trabajo 300 m.
- **Modelo**: multi-tarea — una cabeza de delineación ("¿dónde está quemado ahora?") y
  otra de propagación ("¿dónde arderá?"). La primera es lo que hace que el pipeline
  pueda funcionar en operación, donde CEMS no existe.
- **Cadencia del producto**: cada ~2-3 días, la que marca la revisita de Sentinel-3.

> La cicatriz de quemadura en sí no "crece" cuando el fuego se apaga: lo que crece es
> el fuego. Por eso la tarea es **predicción de propagación**, no delineación. Ver
> [`docs/research.md`](docs/research.md) §3.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/research.md`](docs/research.md) | Estado del arte, por qué 300 m, diseño del dataset, riesgos, log de decisiones |
| [`docs/data-sources.md`](docs/data-sources.md) | Catálogo de fuentes: dónde está cada dato, IDs exactos, endpoints, licencias |

## Decisiones clave

| Decisión | Elección | Motivo |
|---|---|---|
| Entrada del modelo | Imágenes libres + clima y terreno libres | Sin viento el modelo no puede aprender propagación, solo extrapolar espacialmente |
| Salida | Probabilidad de quema por píxel, multi-horizonte | Es la formulación que da el crecimiento; el perímetro final se deriva de ella |
| Región | Mediterráneo / UE | Mejor calidad de etiquetas (CEMS) y régimen de fuego exigente |
| Resolución | 300 m | Banda de resolución útil (300-375 m): el techo de la frecuencia con datos abiertos |
| Etiquetas | CEMS EMSR, delineaciones fechadas | Es lo más próximo a una serie temporal real del crecimiento, gratis y global |
| Etiquetado | Cuantificado (fracción quemada por píxel) | Un binario en el borde es ruido; una fracción lo convierte en información |
| Horizonte | Variable, no fijo | Las rondas de CEMS son irregulares: no hay etiquetas para un t+1d fijo |
| Entrenamiento | Baselines B0/B1/B2 antes de la red | Si no supera la advección por viento, no está aprendiendo propagación |
| Modelo | Multi-tarea: delineación + propagación | CEMS no existe en operación; la cabeza de delineación da la cicatriz conocida y una señal auxiliar densa |
| Horizonte | Anclado en la última observación | Anclarlo en la hora del evento acreditaría tiempo en el que el fuego ya quemaba |

## Estado

**Fase de diseño.** No hay implementación todavía. El dataset está diseñado y medido
(ver `docs/research.md` §5), pero no construido.

## Estructura

```
.
├── README.md
├── docs/
│   ├── research.md        # investigación, decisiones, riesgos
│   └── data-sources.md    # catálogo de datos y licencias
└── openspec/              # flujo de trabajo OpenSpec (vacío por ahora)
```

## Próximos pasos

1. Construir la ingesta: CEMS vía API → cubos GEE → etiquetado cuantificado.
2. Implementar los baselines B0/B1/B2 y fijar el protocolo de evaluación.
3. Entrenar el primer modelo y compararlo contra los baselines.
4. Ampliar el dataset (scraper pre-2023, preentrenamiento con MCD64A1).

## Cómo citar los datos

Cada fuente tiene su norma de licencia y atribución. Están recogidas en
[`docs/data-sources.md`](docs/data-sources.md) §7 — conviene revisarlas antes de
distribuir cualquier dataset derivado.
