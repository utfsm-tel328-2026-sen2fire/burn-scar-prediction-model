"""Tests de la composición de la ventana temporal y del emparejamiento de `f`."""
from __future__ import annotations

import datetime as dt

from burn_scar.dataset import (
    FRAME_MATCH_TOLERANCE,
    build_temporal_window,
    frame_key_to_datetime,
    is_chronological,
    match_frame_key,
    resolve_sample_window,
)

WINDOW_LEN = 7


def _key(hh: str, mm: str, ss: str = "00", day: int = 10) -> str:
    return f"202609{day:02d}T{hh}{mm}{ss}"


# ── 2.1 Emparejamiento por cercanía temporal ──────────────────────────────

def test_match_frame_key_acepta_un_desfase_de_un_segundo():
    """La marca del manifiesto y la del fichero pueden diferir en sub-segundo."""
    # Caso real: el manifiesto trunca 08:49:48.959375 a "084948" y el
    # fichero se nombra con el tiempo redondeado "084949".
    keys = [_key("08", "49", "49")]
    instant = dt.datetime(2026, 9, 10, 8, 49, 48, 959375)

    assert match_frame_key(keys, instant) == _key("08", "49", "49")


def test_match_frame_key_devuelve_none_fuera_de_tolerancia():
    """Una adquisición a 5 minutos no es el frame f."""
    keys = [_key("09", "00", "00"), _key("09", "10", "00")]
    instant = dt.datetime(2026, 9, 10, 9, 5, 0)

    assert match_frame_key(keys, instant) is None


def test_match_frame_key_elige_la_mas_cercana():
    keys = [_key("09", "00", "00"), _key("09", "00", "05"), _key("09", "01", "00")]
    instant = dt.datetime(2026, 9, 10, 9, 0, 3)

    assert match_frame_key(keys, instant) == _key("09", "00", "05")


def test_match_frame_key_sin_candidatos_devuelve_none():
    assert match_frame_key([], dt.datetime(2026, 9, 10)) is None


def test_tolerancia_por_defecto_de_dos_minutos():
    assert FRAME_MATCH_TOLERANCE == dt.timedelta(minutes=2)


def test_frame_key_to_datetime_interpreta_el_formato():
    assert frame_key_to_datetime("20260912T091600") == dt.datetime(2026, 9, 12, 9, 16, 0)


# ── 2.2 Composición de la ventana ─────────────────────────────────────────

def test_ventana_termina_en_f_y_tiene_siete_frames():
    keys = [_key("08", "00"), _key("09", "00"), _key("10", "00"), _key("11", "00")]
    f = _key("11", "00")

    window = build_temporal_window(keys, f, pre_fire_cut_key=_key("09", "00"))

    assert len(window) == WINDOW_LEN
    assert window[-1] == f
    assert is_chronological(window)


def test_ventana_incluye_el_frame_pre_incendio_en_primera_posicion():
    keys = [_key("08", "00"), _key("09", "00"), _key("10", "00"), _key("11", "00")]
    f = _key("11", "00")

    window = build_temporal_window(keys, f, pre_fire_cut_key=_key("09", "00"))

    # El pre-incendio más reciente es el de las 08:00 (anterior al corte) y
    # ocupa la primera posición; los post-incendios cierran la ventana.
    assert window[0] == _key("08", "00")
    assert window[-3:] == [_key("09", "00"), _key("10", "00"), _key("11", "00")]
    assert is_chronological(window)


def test_ventana_repite_el_frame_pre_incendio_cuando_faltan_post_incendios():
    """El relleno repite el frame más antiguo, que es el pre-incendio."""
    keys = [_key("08", "00"), _key("09", "00"), _key("10", "00"), _key("11", "00")]
    f = _key("11", "00")

    window = build_temporal_window(keys, f, pre_fire_cut_key=_key("09", "00"))

    assert len(window) == WINDOW_LEN
    assert window.count(_key("08", "00")) == 4


def test_ventana_repite_el_frame_mas_antiguo_cuando_faltan_frames():
    keys = [_key("09", "00"), _key("10", "00")]
    f = _key("10", "00")

    window = build_temporal_window(keys, f, pre_fire_cut_key=_key("09", "00"))

    assert len(window) == WINDOW_LEN
    assert window[-1] == f
    assert window[0] == _key("09", "00")
    assert is_chronological(window)


def test_ventana_ordenada_cuando_f_es_precedido_por_frames_no_post_incendio():
    """Si no hay frame post-incendio, la ventana son las últimas adquisiciones."""
    keys = [_key("08", "00"), _key("08", "20"), _key("09", "10")]
    f = _key("09", "10")

    window = build_temporal_window(keys, f, pre_fire_cut_key=_key("10", "00"))

    assert window[-1] == f
    assert is_chronological(window)


def test_ventana_sin_pre_incendio_usa_las_ultimas_adquisiciones():
    """Sin adquisición anterior al corte, no se inventa una posición pre-incendio."""
    keys = [_key("09", "00"), _key("10", "00"), _key("11", "00")]
    f = _key("11", "00")

    window = build_temporal_window(keys, f, pre_fire_cut_key=_key("09", "00"))

    assert len(window) == WINDOW_LEN
    assert window[-1] == f
    assert is_chronological(window)


def test_ventana_vacia_sin_adquisiciones():
    assert build_temporal_window([], _key("11", "00"), pre_fire_cut_key=_key("09", "00")) == []


def test_is_chronological_detecta_una_secuencia_invertida():
    assert not is_chronological([_key("10", "00"), _key("09", "00")])
    assert is_chronological([_key("09", "00"), _key("09", "00"), _key("10", "00")])


# ── 3.1 / 3.2 Resolución de la ventana de una muestra ─────────────────────

def test_resolve_sample_window_descarta_la_muestra_sin_adquisicion_de_f():
    """Sin adquisición de f no se sustituye por una adquisición anterior."""
    keys = [_key("08", "00"), _key("09", "00"), _key("10", "00")]
    frame_iso = "2026-09-10T15:00:00"  # ninguna adquisición cerca

    f_key, window = resolve_sample_window(keys, frame_iso, pre_fire_cut_key=_key("09", "00"))

    assert f_key is None
    assert window == []


def test_resolve_sample_window_conserva_la_muestra_sin_frame_pre_incendio():
    """La falta de frame pre-incendio no descarta la muestra."""
    keys = [_key("09", "00"), _key("10", "00"), _key("11", "00")]
    frame_iso = "2026-09-10T11:00:00"

    f_key, window = resolve_sample_window(keys, frame_iso, pre_fire_cut_key=_key("09", "00"))

    assert f_key == _key("11", "00")
    assert len(window) == WINDOW_LEN
    assert window[-1] == f_key
    assert is_chronological(window)


def test_resolve_sample_window_empareja_el_frame_con_desfase_de_sub_segundo():
    keys = [_key("08", "49", "49")]
    frame_iso = "2026-09-10T08:49:48.959375"

    f_key, window = resolve_sample_window(keys, frame_iso, pre_fire_cut_key=_key("07", "00"))

    assert f_key == _key("08", "49", "49")
    assert window[-1] == f_key
