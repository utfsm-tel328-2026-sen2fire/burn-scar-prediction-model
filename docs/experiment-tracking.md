# Guía de Experiment Tracking y TensorBoard Compartido

Esta guía explica cómo registrar métricas, guardar checkpoints y visualizar entrenamientos colaborativamente entre los dos integrantes del proyecto mediante el servidor NFS compartido (`TK_Stratego/Shared`).

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
