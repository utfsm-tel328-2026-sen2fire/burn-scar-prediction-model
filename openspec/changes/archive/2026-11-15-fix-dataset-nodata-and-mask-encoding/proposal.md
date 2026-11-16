# Proposal

## Why

Dos requisitos del spec de `propagation-dataset` no se cumplen en el código ni en el dataset almacenado. El filtrado de adquisiciones sin dato se computa sobre celdas de banda cuando el spec exige hacerlo sobre píxeles, de modo que el criterio real de descarte no es el documentado. Y el dataset por defecto (`data/processed/`, modo `scar`) almacena la validez común entre rondas con el valor `0` en lugar de `255`, incumpliendo la codificación canónica que el propio spec declara obligatoria; el cargador lo enmascara al leer, así que la suite de tests pasa y nadie lo detecta.

## What Changes

- **Filtrado de nodata por píxel**: una adquisición se descarta cuando más del 20 % de sus **píxeles** carecen de dato válido en todas sus bandas, no cuando ese porcentaje se cumple sobre las celdas de banda. Un píxel parcialmente observado cuenta como observado.
- **Codificación canónica en el paquete almacenado**: el paquete `.npz` de cada muestra se escribe con `1` = evaluable y `255` = ignorado en las dos máscaras, sin ningún otro valor, y esa condición se verifica sobre el fichero, no solo sobre lo que devuelve el cargador.
- **Tolerancia explícita del cargador**: el cargador mantiene la normalización de los paquetes antiguos (`0` → `255`), que queda documentada como comportamiento intencionado y no como efecto colateral.
- **Dataset trazable**: el índice del dataset registra la versión del generador con la que se construyó, de modo que un conjunto producido por un generador distinto del actual se detecte antes de usarlo para entrenar o evaluar.
- **BREAKING**: el dataset por defecto se regenera y el dataset en modo `cems` se regenera; el dataset almacenado deja de contener la codificación antigua.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `propagation-dataset`: se precisan los requisitos "Filtering of unusable acquisitions" (el umbral del 20 % se mide sobre píxeles) y "Canonical encoding of the validity mask" (la codificación se cumple en el fichero almacenado, el cargador tolera paquetes antiguos) y se añade el requisito "Traceable dataset version" (el índice identifica la versión del generador).

## Impact

- **Código**: `scripts/build_pytorch_dataset.py` (cálculo del porcentaje de píxeles sin dato), `src/burn_scar/dataset.py` (documentación de la normalización en lectura).
- **Datos**: `data/processed/` y `data/processed_target_cems/` se regeneran; el dataset por defecto pasa a almacenar `mask_cems` con la codificación canónica.
- **Tests**: se añade cobertura que inspecciona el `.npz` almacenado y no solo el resultado del cargador.
- **Documentación**: `docs/dataset.md` (regla D3 y el criterio de descarte de frames) y `docs/operacion.md` (tamaño y composición del dataset tras la regeneración).
- **Modelos y resultados**: el dataset cambia, por lo que el cambio queda encadenado con la regeneración y republicación de cifras del change `fix-propagation-window-alignment`; conviene aplicarlos en ese orden para no regenerar dos veces.
