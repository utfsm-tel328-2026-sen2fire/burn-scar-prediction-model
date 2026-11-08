# Índice de la documentación

> **Estado: Fase B completada.** La documentación está reorganizada: cada documento tiene
> una audiencia y un trabajo, y **cada dato vive en un solo fichero**.
>
> **En una frase:** qué se ha decidido, qué hay que construir, qué necesita el modelo en
> producción y de dónde sale cada dato.
>
> Empieza por el [README de la raíz](../README.md). ¿No sabes qué es una AOI, una
> delineación o el NBR? → [Glosario](dataset.md#1-glosario).

## Los documentos

| Documento | Qué contiene | Naturaleza |
|---|---|---|
| `README.md` (raíz) | Qué es el proyecto, estado, decisiones clave, próximos pasos | punto de entrada |
| [`decisiones.md`](decisiones.md) | Decisiones cerradas (B), pendientes (P, BL), alternativas descartadas, riesgos (R), preguntas abiertas | **cambia** |
| [`dataset.md`](dataset.md) | Glosario, qué es una muestra, el target, el censo, las reglas (D1-D18), baselines y protocolo de evaluación | **cerrado** |
| [`operacion.md`](operacion.md) | Qué necesita el modelo en inferencia: entradas, latencias, ciclo, TMB-S3 como generador de cicatriz | **cerrado** |
| [`investigacion.md`](investigacion.md) | Estado del arte, por qué 300 m, cómo verificar los números | referencia |
| [`datos.md`](datos.md) | Catálogo de fuentes: dónde está cada dato, IDs, endpoints, licencias | referencia |
| [`experiment-tracking.md`](experiment-tracking.md) | TensorBoard compartido, checkpoints, NFS | guía |

El corte es **lo que puede cambiar vs. lo que ya está cerrado**: `decisiones.md` es el único
sitio donde se toca lo que está abierto; `dataset.md` y `datos.md` son especificación.

## Casa canónica de cada dato

Regla: **cada dato vive en un solo fichero**. El resto lo enlaza, no lo repite. Esta tabla
es el contrato: si un número cambia, se cambia en su fila, y en ningún otro sitio.

| Dato | Casa canónica |
|---|---|
| Censo CEMS (activaciones, series, rondas, transiciones) | `dataset.md` §3 |
| Series limpias tras la auditoría y su reparto TMB-S3 | `decisiones.md` §3 (y D13) |
| Nº de muestras con la regla "una por frame OLCI" | `dataset.md` §2 (pendiente P7) |
| Definición del target | `dataset.md` §2 |
| Máscara de evaluación | `dataset.md` §10 (y D3) |
| Rejilla de horizontes de preentrenamiento | `dataset.md` §2 (y D10) |
| Tamaño de ventana, stride y densidades por buffer | `dataset.md` §6 (y D17) |
| Reglas de construcción D1-D18 | `dataset.md` §7 |
| Decisiones cerradas B1-B13 | `decisiones.md` §1.1 |
| Pendientes de evaluación P1-P7 | `decisiones.md` §1.2 |
| Decisiones que bloquean la descarga BL1-BL7 | `decisiones.md` §1.3 |
| Alternativas descartadas | `decisiones.md` §1.4 |
| Riesgos R1-R22 | `decisiones.md` §2 |
| Baselines B0/B1/B2 y criterio de aceptación | `dataset.md` §9 |
| Protocolo de evaluación (métrica, incertidumbre, binarización, estratos) | `dataset.md` §10 |
| Entradas del modelo en operación | `operacion.md` §1 |
| Latencias por fuente | `datos.md` §1 y §6 |
| Estado del arte, benchmarks, por qué 300 m | `investigacion.md` §2 y §3 |

> Esta tabla **no repite los valores**: solo dice dónde vive cada uno. Los valores están en
> el fichero canónico; si un número cambia, cambia ahí y en ningún otro sitio.

## Reglas de mantenimiento

1. **No duplicar.** Antes de escribir un dato en un documento, comprobar en la tabla de
   arriba dónde vive y enlazar en su lugar.
2. **Conservar los identificadores.** `D1-D18`, `R1-R22`, `B1-B13`, `P1-P7`, `BL1-BL7`,
   `M-A/M-B` se citan en el código, en las specs de OpenSpec y en el informe: no se
   renumeran.
3. **Las referencias son cualificadas.** Cuando una referencia cruza de documento, se
   escribe con el fichero (`dataset.md §7`); dentro del mismo documento, basta `§7`.
4. **Actualizar las referencias con el contenido.** Si una sección cambia de sitio, hay que
   buscar quién la citaba (`grep -rn "§X"`).

## Mapa de la migración (research.md → documentos actuales)

| Antes (`research.md`) | Ahora |
|---|---|
| §1 Resumen ejecutivo | `investigacion.md` §1 |
| §2 Glosario | `dataset.md` §1 |
| §3 Estado del arte | `investigacion.md` §2 |
| §4 Por qué 300 m | `investigacion.md` §3 |
| §5 Diseño del dataset | `dataset.md` §2-§8 |
| §6 Qué necesita el modelo en inferencia | `operacion.md` §1-§7 |
| §7 Riesgos y trampas | `decisiones.md` §2 |
| §8 Log de decisiones | `decisiones.md` §1.4 |
| §8 Baselines y protocolo | `dataset.md` §9-§10 |
| §9 Preguntas abiertas y siguientes pasos | `decisiones.md` §3-§7 |
| §10 Cómo verificar los números | `investigacion.md` §4 |
| §11 Decisiones tomadas y pendientes | `decisiones.md` §1.1-§1.3 |
