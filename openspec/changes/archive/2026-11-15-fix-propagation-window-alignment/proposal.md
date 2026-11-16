# Proposal

## Why

La ventana temporal de cada muestra **no termina en el frame actual `f`** en 251 de las 390 muestras del dataset (64 %): la marca de tiempo del frame en el manifiesto CEMS se trunca a segundos y el nombre del fichero descargado viene de otro catálogo con otro redondeo, así que la adquisición de `f` queda fuera de su propia ventana. En consecuencia la cicatriz conocida `x_scar` es la de una adquisición anterior (mediana ~1 día antes, hasta 4) y el target `clip(CEMS(r) − scar(f), 0, 1)` cuenta como "quemado nuevo" quemado que ya había ocurrido. Los números publicados miden esa equivocación y no son reproducibles: el spec ya exige que la ventana termine en `f`, pero el código no lo cumple.

## What Changes

- **Emparejamiento temporal con tolerancia**: la adquisición de `f` se localiza por closeness temporal (no por comparación de cadenas de segundos truncadas), de modo que se acepte el desfase de sub-segundo entre el catálogo CDSE (manifiesto) y el catálogo GEE (descarga).
- **Muestra sin adquisición de `f` no se construye**: si no existe una adquisición utilizable que corresponda a `f`, la muestra no se ensambla con una imagen anterior; se descarta de forma explícita y se informa.
- **Ventana cronológica**: los siete frames de la ventana se garantizan en orden temporal no decreciente, incluido el caso en que no hay frame post-incendio utilizable (hoy se produce una secuencia que va hacia atrás en el 9 % de las muestras).
- **Inventario completo de adquisiciones**: la consulta de imágenes pasa a usar la extensión del área de interés (la que ya viaja en el manifiesto) en lugar del punto del centroide. Hoy el downloader filtra por un punto y el catálogo de referencia por el polígono, y como OLCI se distribuye en tiles fijos con S3A y S3B de geometría distinta, el centroide cae en un hueco de tile de S3B: 125 de las 128 adquisiciones ausentes son de S3B. Además se levanta el tope de resultados de la consulta, que es un recorte real aunque explica solo una de esas 128 ausencias.
- **Cicatriz y target anclados a `f`**: el spec deja de ser ambiguo sobre qué cicatriz se resta, para que la regresión sea detectable.
- **BREAKING**: el dataset se regenera y los números publicados quedan obsoletos. Se reentrena, se reevalúa y se republica el informe.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `propagation-dataset`: se precisan los requisitos "Temporal window with a pre-fire acquisition" y "Non-degenerate propagation target" (emparejamiento tolerante de `f`, descarte explícito de la muestra sin adquisición de `f`, orden cronológico garantizado, cicatriz sustraída tomada en `f`) y se añade el requisito "Complete acquisition inventory per series" (la consulta de imágenes usa la cobertura del área de interés, no un punto, y no se trunca por un límite de resultados).

## Impact

- **Código**: `scripts/build_pytorch_dataset.py` (emparejamiento de `f`, construcción de la ventana, guarda de seguridad), `scripts/download_gee_olci.py` (tope de la consulta de escenas).
- **Datos**: `data/processed/samples/` y `data/processed/dataset_index.json` se regeneran; el dataset por defecto pasa a ser coherente con la cicatriz y el target documentados.
- **Entrenamiento y evaluación**: se reentrenan los 5 folds y se reevalúa el conjunto de test con el umbral operativo congelado.
- **Documentación y resultados**: `docs/operacion.md`, `docs/decisiones.md` y las cifras de referencia quedan actualizadas; los checkpoints publicables se regeneran.
- **Dependencias**: ninguna nueva. Requiere acceso a la API de Earth Engine y a los catálogos CDSE, ya presentes en el flujo actual.
