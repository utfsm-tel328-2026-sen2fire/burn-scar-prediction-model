"""Tests de la consulta de la colección OLCI, sin necesidad de Earth Engine."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path("scripts/download_gee_olci.py")

BBOX_WKT = "POLYGON((16.3142 43.1867, 16.6720 43.1867, 16.6720 43.4652, 16.3142 43.4652, 16.3142 43.1867))"


class FakeGeometry:
    """Doble de `ee.Geometry` que registra el valor con el que se construye."""

    def __init__(self, value=None):
        self.value = value

    def buffer(self, meters):
        return FakeGeometry(("buffer", self.value, meters))

    def bounds(self):
        return FakeGeometry(("bounds", self.value))

    def __repr__(self) -> str:
        return f"Geometry({self.value!r})"


FakeGeometry.Point = staticmethod(lambda coords: FakeGeometry(("point", tuple(coords))))


class FakeImageCollection:
    def __init__(self, recorder):
        self.recorder = recorder

    def filterBounds(self, geom):
        self.recorder["filter_bounds"] = geom
        return self

    def filterDate(self, start, end):
        self.recorder["filter_date"] = (start, end)
        return self

    def select(self, bands):
        return self

    def mosaic(self):
        return self

    def toList(self, limit):
        self.recorder["to_list"] = limit
        return self

    def getInfo(self):
        return self.recorder["images"]


@pytest.fixture
def downloader(monkeypatch):
    """Carga el script con dobles de `ee` y `rasterio`."""
    recorder: dict = {"images": []}

    fake_ee = types.ModuleType("ee")
    fake_ee.Geometry = FakeGeometry
    fake_ee.ImageCollection = lambda name: FakeImageCollection(recorder)
    fake_ee.Image = lambda image_id: types.SimpleNamespace(
        select=lambda bands: types.SimpleNamespace()
    )
    fake_ee.Initialize = lambda **kwargs: None
    fake_ee.ServiceAccountCredentials = lambda key_file: f"credenciales:{key_file}"
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setitem(sys.modules, "rasterio", types.ModuleType("rasterio"))

    spec = importlib.util.spec_from_file_location("download_gee_olci", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # La descarga del parche se sustituye por un array sintético.
    monkeypatch.setattr(
        module,
        "download_gee_patch",
        lambda image, region, crs, dimensions=(64, 64): np.zeros((5, 64, 64), dtype=np.uint16),
    )
    module.recorder = recorder
    return module


def _series_meta() -> dict:
    return {
        "series_id": "TEST_AOI01",
        "center": [16.49, 43.32],
        "bbox_wkt": BBOX_WKT,
        "rounds": [{"acquisition_time": "2026-09-12T15:47:00"}],
    }


def test_la_consulta_usa_el_bbox_del_aoi_y_no_el_centroide(downloader, tmp_path):
    """La cobertura de la consulta debe ser la del AOI, no un punto."""
    downloader.recorder["images"] = []

    downloader.process_series_gee(
        _series_meta(),
        str(tmp_path / "olci"),
        str(tmp_path / "static"),
    )

    geom = downloader.recorder["filter_bounds"]
    assert isinstance(geom.value, dict), "la geometría debe ser un GeoJSON (dict)"
    assert geom.value["type"] == "Polygon"
    assert geom.value["coordinates"][0][0] == [16.3142, 43.1867]


def test_la_consulta_no_se_trunca_en_60_imagenes(downloader, tmp_path):
    """El límite de resultados deja de ser 60, que era un recorte silencioso."""
    downloader.recorder["images"] = []

    downloader.process_series_gee(
        _series_meta(),
        str(tmp_path / "olci"),
        str(tmp_path / "static"),
    )

    assert downloader.recorder["to_list"] == downloader.IMAGE_QUERY_LIMIT
    assert downloader.IMAGE_QUERY_LIMIT > 60


def test_se_avisa_cuando_la_consulta_alcanza_el_limite(downloader, tmp_path, monkeypatch, capsys):
    """Alcanzar el límite de resultados deja rastro en la salida."""
    monkeypatch.setattr(downloader, "IMAGE_QUERY_LIMIT", 2)
    downloader.recorder["images"] = [
        {"id": "a", "properties": {"system:time_start": 1757000000000}},
        {"id": "b", "properties": {"system:time_start": 1757000100000}},
    ]

    downloader.process_series_gee(
        _series_meta(),
        str(tmp_path / "olci"),
        str(tmp_path / "static"),
    )

    assert "AVISO" in capsys.readouterr().out


def test_sin_bbox_wkt_se_usa_el_punto_del_centroide(downloader, tmp_path):
    """Si el manifiesto no trae bbox, se degrada al punto sin romperse."""
    downloader.recorder["images"] = []
    meta = _series_meta()
    meta.pop("bbox_wkt")

    downloader.process_series_gee(meta, str(tmp_path / "olci"), str(tmp_path / "static"))

    geom = downloader.recorder["filter_bounds"]
    assert geom.value[0] == "point"


def test_imas_del_mismo_segundo_se_colapsan_en_un_unico_raster(downloader):
    timestamp_ms = 1_784_000_000_000
    images = [
        {
            "id": "z-image",
            "properties": {
                "system:time_start": timestamp_ms,
                "system:platform": "Sentinel-3B",
            },
        },
        {
            "id": "a-image",
            "properties": {
                "system:time_start": timestamp_ms + 125,
                "system:platform": "Sentinel-3B",
            },
        },
    ]

    canonical = downloader.group_gee_images_by_timestamp(images)

    assert len(canonical) == 1
    assert canonical[0]["id"] == "a-image"
    assert canonical[0]["duplicate_image_ids"] == ["z-image"]


def test_colision_de_satelites_en_el_mismo_segundo_es_explicita(downloader):
    timestamp_ms = 1_784_000_000_000
    images = [
        {
            "id": "s3a",
            "properties": {
                "system:time_start": timestamp_ms,
                "system:platform": "Sentinel-3A",
            },
        },
        {
            "id": "s3b",
            "properties": {
                "system:time_start": timestamp_ms,
                "system:platform": "Sentinel-3B",
            },
        },
    ]

    with pytest.raises(ValueError, match="colisión de ruta no resoluble"):
        downloader.group_gee_images_by_timestamp(images)


def test_el_bbox_wkt_se_convierte_a_geojson(downloader):
    """`ee.Geometry` necesita un diccionario GeoJSON, no WKT ni una cadena."""
    geojson = downloader.wkt_to_geojson(BBOX_WKT)

    assert isinstance(geojson, dict)
    assert geojson["type"] == "Polygon"
    assert len(geojson["coordinates"][0]) == 5


def test_una_consulta_que_falla_se_reintenta(downloader, tmp_path, monkeypatch):
    """Un error transitorio de permisos no debe tumbar la descarga."""
    intentos = {"n": 0}
    real_to_list = FakeImageCollection.toList

    def to_list_fallido(self, limit):
        intentos["n"] += 1
        if intentos["n"] < 3:
            raise RuntimeError("Caller does not have required permission")
        return real_to_list(self, limit)

    monkeypatch.setattr(FakeImageCollection, "toList", to_list_fallido)
    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)

    downloader.process_series_gee(
        _series_meta(),
        str(tmp_path / "olci"),
        str(tmp_path / "static"),
    )

    assert intentos["n"] == 3


def test_una_consulta_que_falla_siempre_propaga_el_error(downloader, tmp_path, monkeypatch):
    intentos = {"n": 0}

    def to_list_fallido(self, limit):
        intentos["n"] += 1
        raise RuntimeError("sin permiso")

    monkeypatch.setattr(FakeImageCollection, "toList", to_list_fallido)
    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        downloader.process_series_gee(
            _series_meta(),
            str(tmp_path / "olci"),
            str(tmp_path / "static"),
        )

    assert intentos["n"] == downloader.QUERY_RETRIES


# ── Credenciales de Earth Engine ───────────────────────────────────────────

def test_sin_variables_de_entorno_no_hay_proyecto_ni_cuenta_de_servicio(downloader):
    """Sin variables, el comportamiento es el de siempre: `ee.Initialize()` a secas."""
    config = downloader.resolve_ee_credentials({})

    assert config["project"] == ""
    assert config["service_account"] == ""


def test_el_proyecto_se_toma_de_earthengine_project(downloader):
    config = downloader.resolve_ee_credentials({"EARTHENGINE_PROJECT": "mi-proyecto"})

    assert config["project"] == "mi-proyecto"


def test_google_cloud_project_es_el_plan_b(downloader):
    config = downloader.resolve_ee_credentials({"GOOGLE_CLOUD_PROJECT": "otro-proyecto"})

    assert config["project"] == "otro-proyecto"


def test_earthengine_project_tiene_prioridad_sobre_google_cloud_project(downloader):
    config = downloader.resolve_ee_credentials(
        {"EARTHENGINE_PROJECT": "a", "GOOGLE_CLOUD_PROJECT": "b"}
    )

    assert config["project"] == "a"


def test_la_cuenta_de_servicio_se_lee_del_entorno(downloader):
    config = downloader.resolve_ee_credentials(
        {"EARTHENGINE_SERVICE_ACCOUNT": "/tmp/clave.json"}
    )

    assert config["service_account"] == "/tmp/clave.json"


def test_sin_variables_se_llama_a_initialize_sin_proyecto(downloader, monkeypatch):
    """Retrocompatible: sin variables, `ee.Initialize(project=None)`."""
    llamadas = []
    monkeypatch.setattr(
        downloader.ee, "Initialize", lambda **kw: llamadas.append(kw)
    )

    downloader.init_earth_engine({})

    assert llamadas == [{"project": None}]


def test_con_cuenta_de_servicio_se_pasa_a_initialize(downloader, monkeypatch, tmp_path):
    clave = tmp_path / "clave.json"
    clave.write_text("{}", encoding="utf-8")

    llamadas = []
    monkeypatch.setattr(
        downloader.ee, "Initialize", lambda **kw: llamadas.append(kw)
    )
    monkeypatch.setattr(
        downloader.ee,
        "ServiceAccountCredentials",
        lambda key_file: f"credenciales:{key_file}",
    )

    downloader.init_earth_engine(
        {
            "EARTHENGINE_PROJECT": "mi-proyecto",
            "EARTHENGINE_SERVICE_ACCOUNT": str(clave),
        }
    )

    assert llamadas[0]["project"] == "mi-proyecto"
    assert llamadas[0]["credentials"] == f"credenciales:{clave}"


def test_una_cuenta_de_servicio_inexistente_da_error_explicito(downloader, monkeypatch):
    monkeypatch.setattr(downloader.ee, "Initialize", lambda **kw: None)

    with pytest.raises(SystemExit) as exc:
        downloader.init_earth_engine(
            {"EARTHENGINE_SERVICE_ACCOUNT": "/no/existe/clave.json"}
        )

    assert "no existe" in str(exc.value)


def test_un_fichero_estatico_valido_se_preserva(downloader, tmp_path, monkeypatch):
    from burn_scar.quality import validate_static_layers

    static_dir = tmp_path / "static"
    series_dir = static_dir / "TEST_AOI01"
    series_dir.mkdir(parents=True)
    static_file = series_dir / "static_layers.npz"
    np.savez_compressed(
        static_file,
        dem=np.full((64, 64), 100.0, dtype=np.float32),
        worldcover=np.full((64, 64), 40, dtype=np.uint8),
    )
    before = static_file.read_bytes()
    calls = []
    monkeypatch.setattr(
        downloader,
        "download_gee_patch",
        lambda *args, **kwargs: calls.append(args) or np.zeros((5, 64, 64), dtype=np.uint16),
    )

    downloader.process_series_gee(
        _series_meta(), str(tmp_path / "olci"), str(static_dir)
    )

    assert static_file.read_bytes() == before
    assert calls == []
    with np.load(static_file) as data:
        validate_static_layers(data["dem"], data["worldcover"])


def test_un_fichero_estatico_invalido_se_reemplaza_por_uno_valido(downloader, tmp_path, monkeypatch):
    from burn_scar.quality import validate_static_layers

    static_dir = tmp_path / "static"
    series_dir = static_dir / "TEST_AOI01"
    series_dir.mkdir(parents=True)
    static_file = series_dir / "static_layers.npz"
    np.savez_compressed(
        static_file,
        dem=np.full((64, 64), 100.0, dtype=np.float32),
        worldcover=np.zeros((64, 64), dtype=np.uint8),
    )
    respuestas = iter(
        [
            np.full((1, 64, 64), 100.0, dtype=np.float32),
            np.full((1, 64, 64), 40, dtype=np.uint8),
        ]
    )
    monkeypatch.setattr(
        downloader,
        "download_gee_patch",
        lambda *args, **kwargs: next(respuestas),
    )

    downloader.process_series_gee(
        _series_meta(), str(tmp_path / "olci"), str(static_dir)
    )

    with np.load(static_file) as data:
        validate_static_layers(data["dem"], data["worldcover"])


def test_un_fallo_estatico_no_deja_un_destino_invalido(downloader, tmp_path, monkeypatch):
    static_dir = tmp_path / "static"
    series_dir = static_dir / "TEST_AOI01"
    series_dir.mkdir(parents=True)
    static_file = series_dir / "static_layers.npz"
    np.savez_compressed(
        static_file,
        dem=np.full((64, 64), 100.0, dtype=np.float32),
        worldcover=np.zeros((64, 64), dtype=np.uint8),
    )

    def fallar(*args, **kwargs):
        raise RuntimeError("fallo de descarga")

    monkeypatch.setattr(downloader, "download_gee_patch", fallar)

    downloader.process_series_gee(
        _series_meta(), str(tmp_path / "olci"), str(static_dir)
    )

    assert not static_file.exists()
    assert (series_dir / "static_layers.npz.invalid").exists()


def test_el_dotenv_no_pisa_a_las_variables_del_entorno(downloader, tmp_path, monkeypatch):
    """Las variables ya exportadas tienen prioridad sobre el `.env`."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("EARTHENGINE_PROJECT=del-dotenv\n", encoding="utf-8")

    env = {**downloader.load_dotenv_file(dotenv), "EARTHENGINE_PROJECT": "del-entorno"}

    assert downloader.resolve_ee_credentials(env)["project"] == "del-entorno"


def test_una_clave_duplicada_avisa_y_gana_el_ultimo_valor(downloader, tmp_path, capsys):
    """Una clave repetida no se resuelve en silencio: gana el último y se avisa."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "EARTHENGINE_PROJECT=primero\nEARTHENGINE_PROJECT=segundo\n", encoding="utf-8"
    )

    values = downloader.load_dotenv_file(dotenv)

    assert values["EARTHENGINE_PROJECT"] == "segundo"
    assert "AVISO" in capsys.readouterr().err


def test_una_clave_duplicada_con_el_mismo_valor_no_avisa(downloader, tmp_path, capsys):
    dotenv = tmp_path / ".env"
    dotenv.write_text("EARTHENGINE_PROJECT=igual\nEARTHENGINE_PROJECT=igual\n", encoding="utf-8")

    downloader.load_dotenv_file(dotenv)

    assert "AVISO" not in capsys.readouterr().err
