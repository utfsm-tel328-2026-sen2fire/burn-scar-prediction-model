"""Criterio de filtrado de adquisiciones sin dato (D19 / spec propagation-dataset)."""
from __future__ import annotations

import numpy as np
import pytest

from burn_scar.quality import MAX_NODATA_FRAC, nodata_pixel_fraction


def test_un_pixel_sin_ninguna_banda_valida_cuenta_como_sin_dato():
    """Dos píxeles, uno observado y otro sin ninguna banda válida."""
    valid = np.ones((5, 1, 2), dtype=bool)
    valid[:, 0, 1] = False

    assert nodata_pixel_fraction(valid) == pytest.approx(0.5)


def test_un_pixel_parcialmente_observado_cuenta_como_observado():
    """El criterio es por píxel: le faltan 4 de 5 bandas pero tiene dato."""
    valid = np.ones((5, 1, 2), dtype=bool)
    valid[4, 0, 0] = False  # al primer píxel le falta una banda

    assert nodata_pixel_fraction(valid) == pytest.approx(0.0)


def test_la_medida_no_depende_del_numero_de_bandas():
    """Con 5 bandas y con 12 bandas, el mismo píxel parcial sigue siendo válido."""
    parcial_5 = np.ones((5, 1, 1), dtype=bool)
    parcial_5[4, 0, 0] = False
    parcial_12 = np.ones((12, 1, 1), dtype=bool)
    parcial_12[11, 0, 0] = False

    assert nodata_pixel_fraction(parcial_5) == pytest.approx(0.0)
    assert nodata_pixel_fraction(parcial_12) == pytest.approx(0.0)


def test_un_frame_usable_no_se_descarta_por_pixeles_parcialmente_observados():
    """El caso real que motivó la corrección: 30 % de píxeles con 4/5 bandas."""
    valid = np.ones((5, 10, 10), dtype=bool)
    valid[4, :3, :] = False  # 30 de 100 píxeles les falta una banda

    # Ningún píxel está completamente muerto: el frame se conserva.
    assert nodata_pixel_fraction(valid) == pytest.approx(0.0)
    assert nodata_pixel_fraction(valid) <= MAX_NODATA_FRAC


def test_un_frame_con_muchos_pixeles_muertos_si_se_descarta():
    valid = np.ones((5, 10, 10), dtype=bool)
    valid[:, :5, :] = False  # 50 de 100 píxeles sin ninguna banda

    assert nodata_pixel_fraction(valid) == pytest.approx(0.5)
    assert nodata_pixel_fraction(valid) > MAX_NODATA_FRAC


def test_la_forma_de_entrada_se_valida():
    with pytest.raises(ValueError):
        nodata_pixel_fraction(np.ones((64, 64), dtype=bool))
