"""Tests de consistencia entre la documentación y los artefactos canónicos.

Las cifras de la documentación no se mantienen como constantes manuales: se
contrastan contra el índice canónico y contra el código que define las bandas
de horizonte. Si el dataset cambia, el test falla.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from burn_scar.quality import HORIZON_BAND_SPECS

CANONICAL_INDEX = Path("data/processed/dataset_index.json")
CANONICAL_INDEX_CEMS = Path("data/processed_target_cems/dataset_index.json")
OPERACION = Path("docs/operacion.md")
DATASET_DOC = Path("docs/dataset.md")
DECISIONES = Path("docs/decisiones.md")
DOCS_DIR = Path("docs")
README = Path("README.md")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _band_intervals() -> list[str]:
    intervals = []
    for _, lower, upper in HORIZON_BAND_SPECS:
        low = f"{lower:g}"
        if upper == float("inf"):
            intervals.append(f"({low},+inf)")
        else:
            intervals.append(f"({low},{upper:g}]")
    return intervals


def _canonical_inventory() -> dict:
    if not CANONICAL_INDEX.exists():
        pytest.skip("Índice canónico no generado")
    payload = json.loads(CANONICAL_INDEX.read_text(encoding="utf-8"))
    return payload.get("inventory", {})


def test_las_bandas_de_la_documentacion_coinciden_con_el_codigo():
    for document in (OPERACION, DATASET_DOC):
        if not document.exists():
            pytest.skip(f"{document} no encontrado")
        text = _normalized(document.read_text(encoding="utf-8"))
        for interval in _band_intervals():
            assert interval in text, f"{document} no documenta la banda {interval}"


def test_la_composicion_documentada_cuadra_con_el_indice_canonico():
    inventory = _canonical_inventory()
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    muestras = inventory["total_unique_samples"]
    series = inventory["total_series"]
    assert f"{muestras} muestras" in text
    assert f"{series} series" in text


def test_la_distribucion_de_horizontes_documentada_cuadra_con_el_indice():
    inventory = _canonical_inventory()
    bands = inventory["horizon_distribution"]["bands"]
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    expected = (
        f"{bands['0-1d']} muestras en `(0,1]`, "
        f"{bands['1-3d']} en `(1,3]`, "
        f"{bands['3-7d']} en `(3,7]`, "
        f"{bands['7-14d']} en `(7,14]`"
    )
    assert expected in text


def test_la_cobertura_de_arranque_en_frio_documentada_cuadra_con_el_indice():
    inventory = _canonical_inventory()
    coverage = inventory["scar_coverage"]
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    assert coverage["cold_start_samples"] == 0
    assert coverage["cold_start_represented"] is False
    assert "arranque en frío" in text
    assert "cold_start_represented: false" in text


def test_la_exclusion_de_series_documentada_cuadra_con_el_indice():
    inventory = _canonical_inventory()
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    for exclusion in inventory["excluded_series"]:
        assert exclusion["series_id"] in text
        assert exclusion["reason"] in text


def test_ambos_indices_canonicos_comparten_composicion():
    if not (CANONICAL_INDEX.exists() and CANONICAL_INDEX_CEMS.exists()):
        pytest.skip("Índices canónicos no generados")
    produccion = json.loads(CANONICAL_INDEX.read_text(encoding="utf-8"))
    cems = json.loads(CANONICAL_INDEX_CEMS.read_text(encoding="utf-8"))

    assert produccion["inventory"]["total_unique_samples"] == (
        cems["inventory"]["total_unique_samples"]
    )
    assert produccion["inventory"]["total_series"] == (
        cems["inventory"]["total_series"]
    )
    assert produccion["inventory"]["horizon_distribution"] == (
        cems["inventory"]["horizon_distribution"]
    )
    assert produccion["target_mode"] != cems["target_mode"]


def test_la_documentacion_no_declara_arranque_en_frio_como_representado():
    for document in (OPERACION, DATASET_DOC, DECISIONES, README):
        if not document.exists():
            continue
        text = _normalized(document.read_text(encoding="utf-8")).lower()
        assert "arranque en frío incluido" not in text


def test_el_run_historico_prop_v4_esta_marcado_como_no_publicable():
    text = _normalized(OPERACION.read_text(encoding="utf-8")).lower()
    assert "prop_v4_integrity" in text
    assert "histórico" in text
    assert "no publicable" in text
    # El run canónico es el único que puede Appecer como resultado publicable.
    assert "prop_v4` queda marcado" in text


def test_la_documentacion_deja_claro_que_no_hay_maximo_de_horizonte():
    negaciones = ("sin máximo", "no hay máximo", "no hay horizonte máximo",
                  "ningún horizonte máximo", "ningun horizonte maximo",
                  "no imponen ningún horizonte máximo",
                  "no fija un horizonte máximo", "no impone un horizonte máximo")
    for document in (OPERACION, DATASET_DOC):
        text = _normalized(document.read_text(encoding="utf-8")).lower()
        assert any(negacion in text for negacion in negaciones), (
            f"{document} no deja claro que no hay horizonte máximo"
        )


def test_las_huellas_documentadas_coinciden_con_el_manifiesto_de_artefactos():
    manifest_path = Path("artefactos/canonicos.json")
    if not manifest_path.exists():
        pytest.skip("Manifiesto de artefactos no generado")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text = _normalized(OPERACION.read_text(encoding="utf-8"))

    for label, entry in manifest["datasets"].items():
        if not entry.get("available"):
            continue
        assert entry["content_sha256"] in text, (
            f"la huella de contenido de {label} no aparece en operacion.md"
        )


def test_el_manifiesto_de_artefactos_declara_la_procedencia_de_los_runs():
    manifest_path = Path("artefactos/canonicos.json")
    if not manifest_path.exists():
        pytest.skip("Manifiesto de artefactos no generado")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for label, entry in manifest["runs"].items():
        if not entry.get("available"):
            continue
        assert entry["checkpoint_dir"], f"{label} no declara su directorio"
        assert entry["operational_threshold_sha256"], f"{label} no declara el hash del umbral"
        assert entry["dataset_index_sha256"], f"{label} no declara el hash del índice"
        assert entry["dataset_content_sha256"], f"{label} no declara la huella de contenido"
        assert entry["dataset_generator_version"], f"{label} no declara el generador"
        assert entry["seed"] is not None, f"{label} no declara la semilla"
        assert len(entry["checkpoints"]) == 5, f"{label} no tiene cinco checkpoints"
        for checkpoint in entry["checkpoints"]:
            assert len(checkpoint["file_sha256"]) == 64


def _documentos() -> list[Path]:
    return [README, *sorted(DOCS_DIR.glob("*.md"))]


def _sin_seccion(texto: str, numero: str) -> str:
    """Devuelve el documento sin la sección cuyo encabezado empieza por `numero`."""
    patron = re.compile(rf"^#{{2,3}} {re.escape(numero)}\.?\s", re.M)
    cabecera = patron.search(texto)
    assert cabecera, f"no se encuentra la sección {numero}"
    siguiente = re.compile(r"^#{2,3} ", re.M).search(texto, cabecera.end())
    fin = siguiente.start() if siguiente else len(texto)
    return texto[: cabecera.start()] + texto[fin:]


def test_el_estado_de_los_baselines_coincide_con_el_codigo():
    """La documentación no puede afirmar que un baseline está implementado si no lo está."""
    source = Path("src/burn_scar/baselines.py")
    if not source.exists():
        pytest.skip("src/burn_scar/baselines.py no encontrado")
    codigo = source.read_text(encoding="utf-8")
    texto = DATASET_DOC.read_text(encoding="utf-8")

    # ¿Está implementado el baseline de advección por viento?
    b2_implementado = "Advection" in codigo or "advection" in codigo

    # El estado que declara la tabla de baselines de dataset.md §9, en SU fila.
    fila = re.search(r"^\|\s*\*{0,2}BASE-2\*{0,2}\s*\|[^\n]*$", texto, re.M)
    assert fila, "dataset.md §9 no declara la fila de BASE-2 (advección por viento)"
    celdas = [c.strip() for c in fila.group(0).strip("|").split("|")]
    estado = _normalized(celdas[-1]).lower()

    if b2_implementado:
        assert "implementado" in estado and "fuera de alcance" not in estado, (
            f"BASE-2 está implementado en el código pero la documentación dice: {estado!r}"
        )
    else:
        assert "fuera de alcance" in estado or "pendiente" in estado, (
            f"BASE-2 no está implementado, así que la fila debe decirlo; dice: {estado!r}"
        )


def _codigos_b_sueltos(texto: str) -> list[str]:
    """Tokens B<digito> que no forman parte de una referencia a un rango (B1-B13)."""
    sueltos = []
    for match in re.finditer(r"\bB\d+\b", texto):
        if re.match(r"\W{0,3}[-–—]\W{0,3}B\d+\b", texto[match.end() :]):
            continue  # "B1-B13": este token es el inicio del rango
        if re.search(r"\bB\d+\W{0,3}[-–—]\W{0,3}$", texto[: match.start()]):
            continue  # "B1-B13": este token es el final del rango
        sueltos.append(match.group(0))
    return sueltos


def test_los_baselines_no_comparten_prefijo_con_las_decisiones():
    """
    B1-B13 son decisiones de diseño; BASE-0..BASE-2 son los baselines.

    El token "B1" no puede significar dos cosas distintas en el mismo repo. Por eso
    los baselines llevan prefijo propio y los códigos B<digito> quedan reservados
    a la serie de decisiones.
    """
    if not (DECISIONES.exists() and DATASET_DOC.exists()):
        pytest.skip("documentación no encontrada")

    # 1) Las dos series, tal y como se declaran.
    decisiones = set(
        re.findall(r"^\|\s*\*{0,2}(B\d+)\*{0,2}\s*\|", DECISIONES.read_text(encoding="utf-8"), re.M)
    )
    baselines = set(
        re.findall(r"^\|\s*\*{0,2}(BASE-\d+)\*{0,2}\s*\|", DATASET_DOC.read_text(encoding="utf-8"), re.M)
    )
    assert decisiones, "no se encuentra la serie de decisiones B* en decisiones.md §1.1"
    assert baselines == {"BASE-0", "BASE-1", "BASE-2"}, (
        f"la serie de baselines debería ser BASE-0..BASE-2; es: {sorted(baselines)}"
    )
    assert not (decisiones & baselines), (
        f"hay identificadores en las dos series: {sorted(decisiones & baselines)}"
    )

    # 2) En las tablas, un código B<digito> suelto solo puede aparecer en
    #    decisiones.md, que es donde vive la serie de decisiones. En el resto, los
    #    baselines van como BASE-<n>. (En prosa, citar "B3" es correcto.)
    for document in _documentos():
        if document in (DECISIONES, DOCS_DIR / "README.md"):
            continue  # el índice y decisiones.md declaran y citan la serie
        texto = document.read_text(encoding="utf-8")
        filas = [fila for fila in re.findall(r"^\|.*$", texto, re.M) if _codigos_b_sueltos(fila)]
        assert not filas, (
            f"{document} tiene tablas con códigos B<digito> sueltos: {filas[:2]}. "
            "Los baselines se nombran BASE-<n>; los B<digito> son decisiones"
        )

    # 3) "B0" solo existió como baseline: la serie de decisiones va de B1 a B13.
    for document in _documentos():
        assert not re.search(r"\bB0\b", document.read_text(encoding="utf-8")), (
            f"{document} usa B0, que era un baseline y ya no existe"
        )

    # 4) Y el código tampoco numera los baselines: las clases no llevan código.
    source = Path("src/burn_scar/baselines.py")
    if source.exists():
        sueltos = _codigos_b_sueltos(source.read_text(encoding="utf-8"))
        assert not sueltos, (
            f"el código numera los baselines con {sueltos}: sus clases son "
            "BaselinePersistence, BaselineRadialDilation y BaselineAllBurned"
        )


def test_la_documentacion_no_tiene_marcadores_de_conflicto():
    """Un conflicto sin resolver deja el documento roto (y ya ha pasado dos veces)."""
    for document in _documentos():
        if not document.exists():
            continue
        for numero, linea in enumerate(document.read_text(encoding="utf-8").split("\n"), 1):
            if linea.startswith(("<<<<<<<", ">>>>>>>")) or linea.strip() == "=======":
                raise AssertionError(
                    f"{document}:{numero} tiene un marcador de conflicto sin resolver: {linea!r}"
                )


def test_las_reglas_del_dataset_estan_documentadas_sin_huecos():
    """Las reglas D1..Dn son el contrato del ensamblador: si hay una más, está documentada."""
    text = DATASET_DOC.read_text(encoding="utf-8")
    numeros = sorted({int(n) for n in re.findall(r"\|\s*\*\*D(\d+)\*\*\s*\|", text)})
    assert numeros, "no se ha encontrado ninguna regla D en dataset.md"
    assert numeros == list(range(1, max(numeros) + 1)), (
        f"las reglas D tienen huecos: {numeros}"
    )


def test_los_scripts_mencionados_en_la_documentacion_existen():
    for document in _documentos():
        if not document.exists():
            continue
        for script in re.findall(r"scripts/([a-z0-9_]+\.py)", document.read_text(encoding="utf-8")):
            assert Path("scripts", script).exists(), (
                f"{document} menciona scripts/{script}, que no existe"
            )


def test_la_comparativa_de_preentrenamiento_esta_documentada():
    """La comparación M-A vs M-B es un resultado del proyecto: tiene que estar escrita."""
    runs = ("mcd_pretrained", "prop_m_b_finetuned", "prop_m_b_tversky")
    texto = _normalized(OPERACION.read_text(encoding="utf-8"))
    assert "Comparativa M-A vs M-B" in texto
    for run in runs:
        assert run in texto, f"la comparativa no menciona el run {run}"
    # El resumen del README debe nombrar los mismos runs que la tabla de §9.
    resumen = _normalized(README.read_text(encoding="utf-8"))
    for run in runs:
        assert run in resumen, f"el README no menciona el run {run}"
