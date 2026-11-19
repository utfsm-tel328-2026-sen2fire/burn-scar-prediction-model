# Índice de la documentación

> **En una frase:** qué se ha decidido, qué hay que construir, qué necesita el modelo en
> producción y de dónde sale cada dato.
>
> Cada documento tiene una audiencia y un trabajo, y **cada dato vive en un solo
> fichero**: el resto lo enlaza, no lo repite.
>
> Empieza por el [README de la raíz](../README.md). ¿No sabes qué es una AOI, una
> delineación o el NBR? → [Glosario](dataset.md#1-glosario).

## Los documentos

| Documento | Qué contiene | Naturaleza |
|---|---|---|
| `README.md` (raíz) | Qué es el proyecto, estado, puesta en marcha, decisiones clave, próximos pasos | punto de entrada |
| [`dataset.md`](dataset.md) | Glosario, qué es una muestra, el target, el pipeline de construcción, las reglas D1-D22, baselines y protocolo de evaluación | **cerrado** |
| [`operacion.md`](operacion.md) | Qué necesita el modelo en inferencia, cómo se regenera el dataset, los resultados canónicos y el preentrenamiento MCD64A1 | **cerrado** |
| [`decisiones.md`](decisiones.md) | Decisiones cerradas (B), pendientes (P, BL), alternativas descartadas, riesgos (R), preguntas abiertas | **cambia** |
| [`investigacion.md`](investigacion.md) | Estado del arte, por qué 300 m, cómo verificar los números | referencia |
| [`datos.md`](datos.md) | Catálogo de fuentes: dónde está cada dato, IDs, endpoints, licencias | referencia |
| [`experiment-tracking.md`](experiment-tracking.md) | TensorBoard compartido, checkpoints, NFS, paneles de predicción en PNG | guía |
| [`nixos.md`](nixos.md) | Cómo trabajar en NixOS con devenv: shell, acelerador, tareas | guía |

Fuera de `docs/`, `artefactos/canonicos.json` registra las sumas de verificación de los
índices de dataset, del umbral operativo y de los checkpoints del run canónico; se
comprueba con `scripts/verify_canonical_artifacts.py`. En `artefactos/resultados/` están
los **JSON de resultados publicables** (baselines sobre el dataset canónico, auditorías de
integridad y una reproducción local del run), versionados para que las cifras de
[`operacion.md`](operacion.md) §9 se puedan comprobar.

El corte es **lo que puede cambiar vs. lo que ya está cerrado**: `decisiones.md` es el único
sitio donde se toca lo que está abierto; `dataset.md` y `datos.md` son especificación.

Y fuera de `docs/` hay dos carpetas de material visual: `outputs/` guarda los **paneles PNG
de las predicciones** (la lectura cualitativa del frente; se explican en
[`experiment-tracking.md`](experiment-tracking.md) §6) y `artefactos/resultados/` los
**JSON de resultados publicables**.

## Casa canónica de cada dato

Regla: **cada dato vive en un solo fichero**. El resto lo enlaza, no lo repite. Esta tabla
es el contrato: si un número cambia, se cambia en su fila, y en ningún otro sitio.

| Dato | Casa canónica |
|---|---|
| Censo CEMS (activaciones, series, rondas, transiciones) | `dataset.md` §3 |
| Series limpias tras la auditoría y su reparto TMB-S3 | `decisiones.md` §3 (y D13) |
| Nº de adquisiciones candidatas y muestras por frame OLCI | `dataset.md` §2 (cerrado: P7) |
| Definición del target | `dataset.md` §2 |
| Máscara de evaluación | `dataset.md` §10 (y D3) |
| Rejilla de horizontes de preentrenamiento | `dataset.md` §2 (y D10) |
| Tamaño de ventana, stride y densidades por buffer | `dataset.md` §6 (y D17) |
| Pipeline de construcción (scripts y orden) | `dataset.md` §11 |
| Reglas de construcción D1-D22 | `dataset.md` §7 |
| Decisiones cerradas B1-B13 | `decisiones.md` §1.1 |
| Pendientes de evaluación P1-P8 | `decisiones.md` §1.2 |
| Decisiones que bloquean la descarga BL1-BL7 | `decisiones.md` §1.3 |
| Alternativas descartadas | `decisiones.md` §1.4 |
| Decisiones de ensamblaje A1-A13 | `decisiones.md` §1.5 |
| Riesgos R1-R22 | `decisiones.md` §2 |
| Baselines BASE-0/BASE-1 + trivial y criterio de aceptación | `dataset.md` §9 |
| Protocolo de evaluación (métrica, incertidumbre, binarización, estratos) | `dataset.md` §10 |
| Entradas del modelo en operación | `operacion.md` §1 |
| Latencias por fuente | `datos.md` §1 y §6 |
| Estado del arte, benchmarks, por qué 300 m | `investigacion.md` §2 y §3 |
| Composición canónica del dataset y exclusiones | `operacion.md` §9 |
| Resultados canónicos y soporte por banda de horizonte | `operacion.md` §9 |
| Comparativa M-A vs M-B (diagnóstico, no canónica) | `operacion.md` §9 |
| Paneles de predicción en PNG (figuras del informe) | `experiment-tracking.md` §6 |
| Flujo de preentrenamiento MCD64A1 | `operacion.md` §10 |
| Sumas de verificación de los artefactos | `artefactos/canonicos.json` |
| Resultados publicables en JSON (baselines, auditorías, reproducción local) | `artefactos/resultados/` |

> Esta tabla **no repite los valores**: solo dice dónde vive cada uno. Los valores están en
> el fichero canónico; si un número cambia, cambia ahí y en ningún otro sitio.

## Mapa de identificadores

Cada cosa que se numera en la documentación tiene **un prefijo propio**, para que un
código nunca pueda significar dos cosas. `B1` es siempre una decisión y `BASE-1` es
siempre un baseline.

| Prefijo | Qué identifica | Rango | Dónde se declara |
|---|---|---|---|
| `D` | Reglas de construcción del dataset | D1–D22 | `dataset.md` §7 |
| `R` | Riesgos y trampas | R1–R22 | `decisiones.md` §2 |
| `B` | Decisiones de diseño cerradas | B1–B13 | `decisiones.md` §1.1 |
| `P` | Pendientes de evaluación | P1–P8 | `decisiones.md` §1.2 |
| `BL` | Decisiones que bloquean la descarga | BL1–BL7 | `decisiones.md` §1.3 |
| `A` | Decisiones de ensamblaje del dataset | A1–A13 | `decisiones.md` §1.5 |
| `BASE` | Baselines de referencia | BASE-0–BASE-2 | `dataset.md` §9 |
| `M` | Modelos del pipeline (preentrenamiento y ajuste) | M-A, M-B | `dataset.md` §8 |

> **Ojo con `B`.** Las decisiones de diseño se numeran `B1`–`B13` y los baselines
> `BASE-0`–`BASE-2`: son series distintas y no comparten ningún token. Antes los
> baselines se numeraban con el mismo prefijo que las decisiones, así que "B1" podía
> significar dos cosas distintas en el mismo repo; por eso los baselines llevan ahora
> prefijo propio.

## Contrato con el código y con las specs

- `tests/test_documentation_consistency.py` **contrasta la documentación contra el índice
  canónico y contra el código**: si el dataset cambia, el test falla. Las cifras no son
  constantes escritas a mano.
- `openspec/specs/` es la especificación de comportamiento, y `openspec/changes/archive/`
  el historial de cómo se llegó a ella. Capacidades: `propagation-dataset`,
  `propagation-evaluation`, `mcd64a1-pretraining`, `prediction-visualization`,
  `execution-environment`, `nix-development-environment`, `shared-experiment-tracking`.

## Reglas de mantenimiento

1. **No duplicar.** Antes de escribir un dato en un documento, comprobar en la tabla de
   arriba dónde vive y enlazar en su lugar.
2. **Conservar los identificadores.** `D1-D22`, `R1-R22`, `B1-B13`, `P1-P8`, `BL1-BL7`,
   `M-A/M-B` se citan en el código, en las specs de OpenSpec y en el informe: no se
   renumeran.
3. **Las referencias son cualificadas.** Cuando una referencia cruza de documento, se
   escribe con el fichero (`dataset.md §7`); dentro del mismo documento, basta `§7`.
4. **Actualizar las referencias con el contenido.** Si una sección cambia de sitio, hay que
   buscar quién la citaba (`grep -rn "§X"`).
5. **Sin marcadores de conflicto.** Un `<<<<<<<` sin resolver en un documento es un
   documento roto: revisa `git diff` antes de commitear.
