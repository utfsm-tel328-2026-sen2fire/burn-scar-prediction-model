# Design

## Context

Ver `proposal.md` para la motivación. Estado actual relevante:

- `scripts/build_pytorch_dataset.py` calcula la fracción de dato ausente como `float((~valid).mean())` sobre un tensor de forma `(5, 64, 64)`: promedia sobre las celdas de banda, no sobre los píxeles. El spec exige que se mida sobre píxeles, y lo dice de forma explícita ("no sobre las celdas de banda").
- El paquete de muestra se escribe con `mask` en `{1, 255}` y `mask_cems` en `{0, 1}` cuando se generó con la versión anterior del generador. `src/burn_scar/dataset.py` normaliza al leer (`_canonical_mask`), de modo que la suite de tests pasa y el incumplimiento del spec queda invisible.
- El índice `dataset_index.json` registra `generated_at` y `target_mode`, pero nada que permita saber con qué versión del generador se construyó. Por eso un dataset obsoleto no es detectable a simple vista.

Restricciones: el cambio debe ser compatible con los paquetes antiguos ya existentes en disco (hay copias y datasets derivados en `data/`); no se puede obligar a regenerar todo el histórico de inmediato; la suite de tests debe seguir siendo ejecutable sin dataset presente.

## Goals / Non-Goals

**Goals:**
- Que el criterio de descarte de adquisiciones sin dato sea el documentado, medido sobre píxeles.
- Que el paquete almacenado cumpla la codificación canónica, sin depender del cargador para que se cumpla.
- Que la conformidad del paquete almacenado sea verificable por la suite de tests.
- Que un dataset producido por un generador distinto del vigente sea detectable antes de usarlo.

**Non-Goals:**
- No se cambia el criterio de cobertura por banda del tensor de entrada: la sustitución de celdas sin dato por la media de su banda se mantiene tal cual.
- No se cambian los umbrales numéricos del filtro (20 %) ni el criterio de adquisición nocturna.
- No se reformatea ni se versiona el paquete de muestra más allá de las dos máscaras.
- No se resuelve aquí la alineación de la ventana temporal: es el change `fix-propagation-window-alignment`.

## Decisions

### D1. La unidad de medida del filtro es el píxel

**Elegido:** una adquisición se descarta cuando más del 20 % de sus **píxeles** no tienen dato válido en ninguna de sus bandas, es decir cuando el píxel carece por completo de observación. La señal es: `fill_frac = (~valid.any(axis=0)).mean()`.

**Por qué:** es lo que dice el spec y lo que tiene sentido físico. El dato ausente en Sentinel-3 OLCI es espacialmente coherente (borde del swath, zonas no iluminadas), así que en la práctica las bandas fallan juntas y ambas medidas coinciden en 2120 de las 2121 adquisiciones disponibles. Pero cuando no coinciden, la medida por celda queda sin significado: depende del número de bandas elegidas (5) y trata un píxel con cuatro bandas válidas como si estuviera casi vacío. La regla de D3 del proyecto ("nadie miró aquí") es una afirmación sobre la cobertura del píxel, no sobre la cobertura de cada banda.

**Alternativa descartada:** cambiar el spec para que la medida sea por celda. Se descartaría entonces la semántica documentada y la medida quedaría atada al número de bandas, que es un detalle de implementación.

**Consecuencia asumida:** cambia el conjunto de adquisiciones utilizables en, al menos, una adquisición del inventario actual (`20260904T092328`: 5,4 % por celda frente a 26,7 % por píxel). Ese frame pasa a descartarse, con lo que pueden aparecer muestras nuevas o cambiar la ventana de muestras existentes.

### D2. La conformidad del paquete se verifica sobre el fichero, no sobre el cargador

**Elegido:** el generador escribe el paquete con la codificación canónica y, antes de escribirlo, comprueba que sus dos máscaras contienen únicamente `1` y `255`; si no fuera así, aborta. En paralelo, la suite de tests añade un caso que abre el `.npz` almacenado directamente y comprueba los conjuntos de valores de `mask` y `mask_cems`.

**Por qué:** el test existente (`tests/test_dataset.py`) pasa porque el cargador normaliza; por eso la suite no detectó el incumplimiento. Un test que mira el fichero crudo es el que hace visible el requisito "SHALL NOT almacenar ningún otro valor". La comprobación en el generador evita además escribir paquetes no conformes.

**Alternativa descartada:** fiarse solo del test. No protege contra una regresión que se introduzca después de generar el dataset.

### D3. El cargador mantiene la normalización, ahora como comportamiento declarado

**Elegido:** `_canonical_mask` se conserva y su propósito queda documentado explícitamente como compatibilidad con paquetes antiguos.

**Por qué:** hay paquetes en disco generados antes del cambio y el dataset en modo `cems` de una corrida anterior puede seguir usándose. Quitar la normalización obligaría a regenerar todo de golpe y rompería la lectura de cualquier copia existente. El coste es que el cargador sigue siendo capaz de enmascarar un paquete no conforme; por eso D2 añade la verificación en el punto de escritura y en el fichero.

### D4. La versión del generador es una huella del contenido de sus fuentes

**Elegido:** el índice registra `generator_version`: una huella corta (sha256 truncado) del contenido de los ficheros que determinan el dataset (`scripts/build_pytorch_dataset.py` y `src/burn_scar/dataset.py`), junto con la lista explícita de esos ficheros para que el valor sea autodescriptivo.

**Por qué:** un número de versión mantenido a mano se olvida subir y produce un falso "compatible". El hash de contenido cambia exactamente cuando cambia el comportamiento del generador, y es reproducible sin acceso al historial de Git. El hash de Git se descartó porque no captura modifications sin confirmar, que es el estado habitual durante el desarrollo.

**Alternativa descartada:** registrar solo el hash del commit de Git. No distingue un dataset generado con el árbol de trabajo sucio, que es el caso normal mientras se develops.

### D5. La detección de obsolescencia se hace al cargar el dataset

**Elegido:** el cargador de muestras compara la `generator_version` del índice con la versión vigente y emite un aviso cuando difieren. El aviso aparece en el punto en que el dataset se va a usar para entrenar o evaluar, que es donde la diferencia importa.

**Por qué:** es el punto de uso real. Comprobarlo solo al regenerar dejaría pasar el caso habitual, que es llegar a un dataset viejo y usarlo sin haber regenerado.

**Alternativa descartada:** un script aparte de comprobación. Se puede no ejecutar, y el aviso en el punto de uso no depende de la disciplina de quien lo usa.

### D6. Orden de aplicación y coste de regeneración

**Elegido:** aplicar `fix-propagation-window-alignment` antes que este change. Ambos tocan el generador y ambos obligan a regenerar el dataset; el primero cambia qué muestras existen, el segundo cambia cómo se filtran los frames y cómo se escriben las máscaras. Regenerando una sola vez al final, se evita un ciclo completo de ensamblaje, entrenamiento y republicación de cifras.

**Consecuencia asumida:** entre ambos changes hay un dataset transitorio que no cumple ninguno de los dos requisitos. No se publica ni se entrena con él; es un artefacto de trabajo.

## Risks / Trade-offs

- **Cambia la selección de frames** → D1 puede descartar una adquisición que antes se usaba, lo que altera la ventana de algunas muestras y el tamaño del dataset. Mitigación: los contadores de control del ensamblaje (ya previstos en el change de la ventana) permiten ver el efecto, y la comparación de composición de los datasets queda documentada.
- **El aviso de obsolescencia puede resultar ruidoso** → todo dataset generado antes de este change carece de `generator_version` y disparará el aviso. Mitigación: el aviso distingue explícitamente "dataset sin versión registrada" de "dataset generado con otra versión", y no es un error.
- **El hash depende de qué ficheros se enumeran** → olvidar un fichero relevante dejaría dos generadores distintos con la misma huella. Mitigación: la lista de ficheros se escribe en el índice y el test verifica que la lista sigue siendo la esperada.
- **El test que abre el `.npz` requiere dataset presente** → en entornos sin dataset el test se salta, igual que el resto de los tests que dependen de `data/processed`. Mitigación: se mantiene el patrón `skipif` ya usado en `tests/test_dataset.py`.
- **Dos regeneraciones si se aplican los changes en orden inverso** → trabajo de GPU y de entrenamiento evitable. Mitigación: D6 fija el orden.

## Migration Plan

1. Aplicar el cambio del filtro (D1) y el de la escritura canónica (D2) en el generador.
2. Añadir la huella del generador al índice (D4) y el aviso de obsolescencia al cargador (D5).
3. Regenerar el dataset por defecto en modo `scar` y el dataset en modo `cems`.
4. Verificar que la suite de tests pasa y que el test sobre el fichero crudo encuentra las dos máscaras canónicas.
5. Reentrenar y reevaluar junto con el change de la ventana temporal, y republicar las cifras una sola vez.
6. Rollback: el generador vuelve a su versión anterior y el dataset se regenera; no hay estado que migrar.

## Open Questions

Ninguna. La granularidad del filtro, la verificación sobre el fichero, la tolerancia del cargador y el mecanismo de versión quedan fijados por D1 a D5.
