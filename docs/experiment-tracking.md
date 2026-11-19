# Guía de Experiment Tracking y TensorBoard Compartido

> **En una frase:** cómo registrar métricas y pesos en un TensorBoard compartido por NFS
> para que los dos desarrolladores vean lo mismo.

Esta guía explica cómo registrar métricas, guardar checkpoints y visualizar entrenamientos
colaborativamente entre los dos integrantes del proyecto mediante el servidor NFS compartido
(`TK_Stratego/Shared`).

---

## 1. Configuración Inicial (Una sola vez por máquina)

Cada desarrollador debe tener su archivo `.env` local en la raíz del repositorio (este archivo está ignorado por Git, por lo que cada uno puede tener sus propias rutas sin pisarse).

1. Copia la plantilla:
   ```bash
   cp .env.example .env
   ```

2. Configura tus variables en `.env`:
   * `BURNSCAR_SHARED_STORAGE` (obligatoria para compartir): si la ruta no existe o no está
     montada, el sistema avisa y usa el almacenamiento local de respaldo.
   * `BURNSCAR_USER` (recomendada): aísla tus experimentos con tu nombre. Si no la pones,
     el sistema usa el usuario del sistema operativo, que puede coincidir con el de tu
     compañero.
   * `BURNSCAR_TENSORBOARD_PORT` (opcional, por defecto 6006).
   * **Para Antonio:**
     ```ini
     BURNSCAR_SHARED_STORAGE=/home/antonio/TK_Stratego/Shared
     BURNSCAR_USER=antonio
     BURNSCAR_TENSORBOARD_PORT=6006
     ```
   * **Para tu compañero:**
     ```ini
     BURNSCAR_SHARED_STORAGE=/home/ruta_de_tu_companero/TK_Stratego/Shared
     BURNSCAR_USER=nombre_companero
     BURNSCAR_TENSORBOARD_PORT=6006
     ```

> [!NOTE]
> Si el servidor NFS no está montado o no configuras `BURNSCAR_SHARED_STORAGE`, el sistema emitirá un aviso y usará automáticamente almacenamiento local de respaldo (`./runs` y `./checkpoints`).

---

## 2. Cómo Levantar TensorBoard

Para abrir la interfaz web y ver las curvas de ambos desarrolladores:

```bash
uv run --extra cuda python scripts/launch_tensorboard.py
```

* El script detecta automáticamente la ruta del NFS configurada en tu `.env` (o el fallback local) y levanta el servidor.
* Verás en la consola la URL activa:
  ```text
  [TensorBoard] Modo: NFS compartido
  [TensorBoard] Directorio de logs: /home/.../TK_Stratego/Shared/burn_scar/runs
  [TensorBoard] URL: http://localhost:6006
  ```
* Abre tu navegador en **[http://localhost:6006](http://localhost:6006)**.

> [!NOTE]
> En los ejemplos, `--extra cuda` es solo un ejemplo: el extra depende de tu GPU
> (`rocm` en AMD, `cuda` en NVIDIA, `cpu` si no hay acelerador). En NixOS, el intérprete
> y el venv son otros: el venv de devenv vive en `.devenv/state/venv` en vez de `./.venv`,
> y el servidor se lanza desde dentro del shell. Ver [`nixos.md`](nixos.md) §3 y §5.

### Parámetros opcionales del launcher:
* Cambiar de puerto: `uv run --extra cuda python scripts/launch_tensorboard.py --port 6007`
* Exponer a la red local: `uv run --extra cuda python scripts/launch_tensorboard.py --host 0.0.0.0`
* Solo ver la ruta resuelta sin lanzar: `uv run --extra cuda python scripts/launch_tensorboard.py --dry-run`

---

## 3. Cómo Usar el Tracker en Scripts de Entrenamiento

En lugar de usar `SummaryWriter` a mano, utiliza la clase `ExperimentTracker` provista en `src/burn_scar/tracking.py`. 

Esta clase se encarga automáticamente de:
1. Crear el namespace único (`<usuario>_<tag>_<timestamp>`) para no pisar las carpetas del otro.
2. Forzar el vaciado a disco (`flush()`) para que el otro vea los avances en tiempo real por NFS.
3. Asegurar permisos de grupo en Linux (`0o775` / `0o664`) para que ambos puedan leer y escribir sin bloqueos.
4. Guardar los pesos (`checkpoints`) en la carpeta correspondiente.

### Ejemplo Completo de Entrenamiento:

```python
import torch
import torch.nn as nn
from burn_scar.tracking import ExperimentTracker

# 1. Tu modelo y optimizador
model = nn.Linear(10, 1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Inicias el tracker con el nombre/tag del experimento
#    Crea automáticamente: runs/<tu_usuario>_baseline_b0_<timestamp>/
with ExperimentTracker("baseline_b0") as tracker:
    print(f"Logueando en: {tracker.run_dir}")
    print(f"Checkpoints en: {tracker.checkpoint_dir}")

    for epoch in range(10):
        # Simulación de pérdida y métricas
        train_loss = 1.0 / (epoch + 1)
        val_iou = 0.5 + 0.04 * epoch

        # Registrar un escalar individual:
        tracker.log_scalar("Loss/train", train_loss, global_step=epoch)

        # O registrar múltiples métricas en el mismo paso:
        tracker.log_metrics({
            "Loss/val": train_loss * 1.1,
            "Metrics/IoU": val_iou
        }, global_step=epoch)

    # 3. Guardar el mejor modelo en el NFS compartido
    best_weights_path = tracker.save_checkpoint({
        "epoch": 10,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_iou": val_iou
    }, filename="best_model.pt")

    print(f"Pesos guardados exitosamente en: {best_weights_path}")
```

---

## 4. Estructura de Carpetas en el Almacenamiento Compartido

Dentro del servidor NFS (`TK_Stratego/Shared/burn_scar/`), los archivos quedan ordenados de la siguiente forma:

```
TK_Stratego/Shared/burn_scar/
├── runs/
│   ├── antonio_baseline_b0_20261107_180000/
│   │   └── events.out.tfevents...
│   └── companero_unet_atten_20261107_193000/
│       └── events.out.tfevents...
│
└── checkpoints/
    ├── antonio_baseline_b0_20261107_180000/
    │   └── best_model.pt
    └── companero_unet_atten_20261107_193000/
        └── best_model.pt
```

---

## 5. Visualización y Comparación en TensorBoard

Cuando ambos entrenen, al abrir **http://localhost:6006**:
* **Filtros por usuario**: En la esquina superior izquierda del panel de TensorBoard, puedes escribir `antonio` para ver solo tus experimentos, `companero` para ver los de tu colega, o dejarlos ambos para comparar cómo evolucionan las curvas superpuestas.
* **Descarga de checkpoints**: Como los checkpoints se guardan en el NFS bajo la misma estructura, si tu compañero entrenó un modelo, puedes cargarlo directamente desde tu máquina leyendo su ruta de checkpoint.

---

## 6. Inspección Visual de Predicciones por Checkpoint (Pestaña IMAGES)

Para evaluar un checkpoint ya entrenado (`best_fold_*.pt`) sobre su split de validación correspondiente y visualizar los resultados espaciales en el apartado **IMAGES** de TensorBoard, utiliza `scripts/visualize_predictions.py`:

```bash
# Ver las N mejores y N peores predicciones del fold (modo por defecto: --mode both)
uv run --extra cuda python scripts/visualize_predictions.py \
  --checkpoint checkpoints/<run_namespace>/best_fold_0.pt \
  --mode both \
  --n-samples 6
```

### Modos de selección disponibles (`--mode`)
* `--mode both` *(por defecto)*: registra las `N` mejores y las `N` peores muestras según su F1 por muestra.
* `--mode best` / `--mode worst`: registra únicamente las `N` mejores o las `N` peores.
* `--mode series --series-id <ID>`: filtra todas las muestras de un incendio/evento específico en ese fold.
* `--mode indices --indices 0 5 12`: inspecciona índices puntuales del split de validación.
* `--mode random --n-samples 8`: muestreo aleatorio reproducible (`--seed`).

### Qué muestra cada panel horizontal (5 bloques)
1. **OLCI pseudo-RGB**: último frame espectral de entrada (`x_olci[-1]`).
2. **`x_scar`**: cicatriz previa conocida en el instante $f$ (azul acero).
3. **`target`**: nueva área quemada real en $(f, f+h]$ (verde brillante; gris atenuado donde `mask == 255`).
4. **Probabilidad**: mapa continuo $\sigma(\text{logits}) \in [0, 1]$ (mapa de calor).
5. **Overlay TP/FP/FN**: azul = `x_scar`, verde = acierto (`TP`), rojo = falsa alarma (`FP`), amarillo = omisión (`FN`), gris = ignorado (`mask == 255`).

### Cómo guardar/exportar las imágenes
1. **Desde la propia página de TensorBoard**: en la pestaña **IMAGES**, haz clic derecho sobre cualquier panel y selecciona *"Guardar imagen como..."* (o activa *"Show actual image size"* en el panel superior para descargarlo a resolución completa escalada por `--scale`).
2. **Exportación directa en lote a PNG**: agrega `--save-dir <carpeta>` al comando para guardar automáticamente todos los paneles generados como archivos `.png` listos para usar como figuras:
   ```bash
   uv run --extra cuda python scripts/visualize_predictions.py \
     --checkpoint checkpoints/<run_namespace>/best_fold_0.pt \
     --mode both \
     --n-samples 6 \
     --save-dir runs/exported_panels_fold0
   ```

### Los paneles del repositorio: la carpeta `outputs/`

La carpeta `outputs/` del repo guarda **40 paneles exportados a PNG** de los runs de
diagnóstico, para poder mirar el comportamiento espacial del modelo sin levantar
TensorBoard. Es material de figura: la lectura visual de un frente.

| Carpeta | Qué contiene |
|---|---|
| `outputs/vis_m_b_fold0/` | M-B (ajuste con CEMS), fold 0: las 4 mejores y las 4 peores muestras |
| `outputs/vis_m_b_fold0_random/` | M-B, fold 0: 2 muestras elegidas al azar |
| `outputs/vis_tversky_mpw5_fold0/` | M-B + Tversky (`w_max = 5`), fold 0: las 6 mejores y las 6 peores |
| `outputs/vis_tversky_mpw5_fold0_random/` | M-B + Tversky, fold 0: 6 muestras al azar |
| `outputs/vis_tversky_mpw5_fold0_thr03/` | M-B + Tversky, fold 0, binarizado a umbral 0,30 (y no a 0,50) |

El nombre de cada fichero dice de dónde sale: `best_01_idx016_EMSR899_AOI01_h0d_f1_070.png`
es la mejor muestra (por F1) del índice 016 de la serie `EMSR899_AOI01`, con horizonte de
0 días.

**Cómo se leen**: cada PNG es una tira de cinco bloques (ver §6): imagen de entrada, cicatriz
conocida, **verdad** (verde = quemado nuevo), **probabilidad** (mapa de calor) y overlay de
aciertos: **verde = acierto, rojo = falsa alarma, amarillo = omisión**.

**Cómo se regeneran** (el comando que produjo cada carpeta):

```bash
uv run --extra rocm python scripts/visualize_predictions.py \
  --checkpoint checkpoints/<run_m_b>/best_fold_0.pt \
  --mode both --n-samples 6 \
  --save-dir outputs/vis_m_b_fold0
```

> Son **figuras de diagnóstico de los runs M-B y M-B + Tversky**, no del modelo de
> producción: las cifras publicables son las de [`operacion.md`](operacion.md) §9, y estas
> imágenes son la lectura cualitativa de esas mismas etiquetas. Se generan sobre el split de
> **validación** de su fold, no sobre el test.

