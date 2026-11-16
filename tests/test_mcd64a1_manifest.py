from __future__ import annotations

import datetime as dt

import numpy as np

from scripts.build_mcd64a1_dataset import compute_mcd_target_and_mask
from scripts.build_mcd64a1_manifest import (
    PRETRAIN_HORIZONS,
    build_event_window_and_horizons,
    dbscan_spatiotemporal,
    filter_and_cluster_candidates,
    is_overlapping_cems,
)


def test_is_overlapping_cems_spatial_and_temporal():
    cems_records = [
        {
            "id": "EMSR921_AOI01",
            "lon": 23.85,
            "lat": 38.15,
            "start_date": dt.date(2026, 8, 12),
            "end_date": dt.date(2026, 8, 20),
        }
    ]

    # Mismo lugar y fecha -> solapa
    overlap, matched_id = is_overlapping_cems(
        23.90,
        38.10,
        dt.date(2026, 8, 14),
        dt.date(2026, 8, 18),
        cems_records,
        spatial_radius_km=50.0,
        temporal_buffer_days=45,
    )
    assert overlap is True
    assert matched_id == "EMSR921_AOI01"

    # Mismo lugar pero 2 años antes -> NO solapa
    overlap_diff_year, _ = is_overlapping_cems(
        23.90,
        38.10,
        dt.date(2024, 8, 14),
        dt.date(2024, 8, 18),
        cems_records,
        spatial_radius_km=50.0,
        temporal_buffer_days=45,
    )
    assert overlap_diff_year is False

    # Misma fecha pero a >500 km -> NO solapa
    overlap_far, _ = is_overlapping_cems(
        -5.0,
        39.0,
        dt.date(2026, 8, 14),
        dt.date(2026, 8, 18),
        cems_records,
        spatial_radius_km=50.0,
        temporal_buffer_days=45,
    )
    assert overlap_far is False


def test_filter_and_cluster_excludes_entire_contaminated_dbscan_cluster():
    cems_records = [
        {
            "id": "EMSR_TEST",
            "lon": 15.0,
            "lat": 40.0,
            "start_date": dt.date(2025, 7, 10),
            "end_date": dt.date(2025, 7, 15),
        }
    ]

    candidates = [
        # Candidato A: coincide con CEMS
        {
            "year": 2025,
            "center": [15.05, 40.02],
            "epsg_utm": 32633,
            "start_date": dt.date(2025, 7, 11),
            "end_date": dt.date(2025, 7, 14),
            "overview_burned_cells": 25,
            "estimated_burned_ha": 10000.0,
        },
        # Candidato B: a ~80 km de A (mismo clúster DBSCAN de 150 km) -> también debe excluirse
        {
            "year": 2025,
            "center": [15.80, 40.20],
            "epsg_utm": 32633,
            "start_date": dt.date(2025, 7, 12),
            "end_date": dt.date(2025, 7, 16),
            "overview_burned_cells": 18,
            "estimated_burned_ha": 7200.0,
        },
        # Candidato C: en España (lejos de Italia), limpio
        {
            "year": 2025,
            "center": [-3.50, 40.10],
            "epsg_utm": 32630,
            "start_date": dt.date(2025, 7, 12),
            "end_date": dt.date(2025, 7, 18),
            "overview_burned_cells": 30,
            "estimated_burned_ha": 12000.0,
        },
    ]

    selected, stats = filter_and_cluster_candidates(
        candidates, cems_records, max_events=10, cluster_eps_km=150.0, cluster_eps_days=30
    )
    assert stats["total_candidates"] == 3
    assert stats["excluded_by_cems_cluster"] == 2
    assert len(selected) == 1
    assert selected[0]["center"] == [-3.50, 40.10]


def test_build_event_window_and_horizons_d18():
    start = dt.date(2025, 8, 10)
    end = dt.date(2025, 8, 15)
    win = build_event_window_and_horizons(start, end)

    assert win["burn_start_date"] == "2025-08-10"
    assert win["burn_end_date"] == "2025-08-15"
    assert win["duration_days"] == 6
    # D18: 14 días antes para garantizar frame pre-incendio
    assert win["olci_start_date"] == "2025-07-27"
    assert win["olci_end_date"] == "2025-08-16"
    assert win["candidate_horizons_days"] == PRETRAIN_HORIZONS


def test_dbscan_spatiotemporal_separates_distant_or_late_events():
    lons = np.array([10.0, 10.1, 25.0], dtype=np.float64)
    lats = np.array([40.0, 40.05, 38.0], dtype=np.float64)
    ords = np.array([100, 105, 102], dtype=np.int32)

    labels = dbscan_spatiotemporal(lons, lats, ords, eps_km=150.0, eps_days=30)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_compute_mcd_target_and_mask():
    f_date = dt.date(2025, 8, 10)
    f_ord = f_date.toordinal()

    burn_ord = np.zeros((4, 4), dtype=np.int32)
    qa = np.ones((4, 4), dtype=np.uint8)  # bit 0 = 1 (land), bits 5..7 = 0 (valid)

    burn_ord[0, 0] = f_ord        # quemado en f (no es crecimiento futuro) -> target 0
    burn_ord[0, 1] = f_ord + 2    # quemado en f+2 -> target 1 para h=3
    burn_ord[0, 2] = f_ord + 5    # quemado en f+5 -> target 0 para h=3, 1 para h=7
    burn_ord[1, 0] = -1           # sin dato -> mask 255
    burn_ord[1, 1] = -2           # agua -> mask 255
    qa[2, 0] = 0b00100001         # bits 5..7 != 0 (observaciones espaciadas) -> mask 255
    qa[2, 1] = 0b00000000         # bit 0 == 0 (agua en QA) -> mask 255

    target_h3, mask_h3 = compute_mcd_target_and_mask(burn_ord, qa, f_date=f_date, horizon_days=3)
    assert target_h3[0, 0] == 0.0
    assert target_h3[0, 1] == 1.0
    assert target_h3[0, 2] == 0.0
    assert mask_h3[0, 1] == 1
    assert mask_h3[1, 0] == 255
    assert mask_h3[1, 1] == 255
    assert mask_h3[2, 0] == 255
    assert mask_h3[2, 1] == 255

