"""Semana 10: costo por inferencia a partir de req/s y una tarifa horaria.

Las pruebas usan filas fijas. No leen las mediciones de Locust ni consultan precios.
"""

from decimal import Decimal

from scripts.week10.estimate_inference_cost import (
    build_cost_table,
    cost_per_successful_inference,
    is_recommended_operating_point,
    load_errors_are_zero,
)

HOURLY_USD = Decimal("0.145")


def test_costo_por_inferencia_divide_la_tarifa_entre_las_exitosas_de_una_hora():
    requests_per_second = Decimal("96.9")
    assert cost_per_successful_inference(HOURLY_USD, requests_per_second) == HOURLY_USD / (
        requests_per_second * 3600
    )


def test_req_por_segundo_en_cero_no_divide():
    assert cost_per_successful_inference(HOURLY_USD, Decimal(0)) is None
    assert cost_per_successful_inference(HOURLY_USD, Decimal(-1)) is None


def test_punto_recomendado_es_150_usuarios_con_p95_menor_a_80():
    assert is_recommended_operating_point("150", Decimal(76))
    assert not is_recommended_operating_point("150", Decimal(80))
    assert not is_recommended_operating_point("300", Decimal(740))
    assert not is_recommended_operating_point("100", Decimal(44))
    assert not is_recommended_operating_point("50-300", Decimal(50))


def _recurso(cpu: str, memoria: str) -> dict[str, str]:
    return {"cpu_media_pct": cpu, "mem_media_mib": memoria}


def test_tabla_marca_solo_el_escalon_de_150_y_deja_el_reposo_sin_costo_unitario():
    load_summary = [
        {"escenario": "S1", "usuarios": "5", "rps": "3.23", "error_pct": "0.0", "p95_ms": "50"},
        {"escenario": "S2", "usuarios": "20", "rps": "12.57", "error_pct": "0.0", "p95_ms": "48"},
        {"escenario": "S3", "usuarios": "50", "rps": "31.54", "error_pct": "0.0", "p95_ms": "36"},
        {"escenario": "S4", "usuarios": "100", "rps": "63.21", "error_pct": "0.0", "p95_ms": "44"},
        {"escenario": "S5", "usuarios": "50-300", "rps": "100.33", "error_pct": "0.0", "p95_ms": "700"},
    ]
    capacity_steps = [
        {"usuarios": "150", "rps": "96.9", "fallos_s": "0.0", "p95_ms": "76"},
        {"usuarios": "300", "rps": "147.1", "fallos_s": "0.0", "p95_ms": "740"},
    ]
    resource_summary = [
        {"escenario": "Base", "servicio": "api", **_recurso("3.45", "303.9")},
        {"escenario": "Base", "servicio": "mlflow", **_recurso("5.44", "2166.51")},
        {"escenario": "S1", "servicio": "api", **_recurso("14.68", "304.51")},
        {"escenario": "S1", "servicio": "mlflow", **_recurso("2.9", "2166.31")},
        {"escenario": "S2", "servicio": "api", **_recurso("38.8", "304.99")},
        {"escenario": "S2", "servicio": "mlflow", **_recurso("2.87", "2167.86")},
        {"escenario": "S3", "servicio": "api", **_recurso("71.01", "305.79")},
        {"escenario": "S3", "servicio": "mlflow", **_recurso("1.77", "2168.18")},
        {"escenario": "S4", "servicio": "api", **_recurso("120.76", "307.56")},
        {"escenario": "S4", "servicio": "mlflow", **_recurso("1.77", "2169.0")},
    ]
    resource_steps = [
        {"usuarios": "150", "servicio": "api", **_recurso("173.78", "308.75")},
        {"usuarios": "150", "servicio": "mlflow", **_recurso("4.51", "2176.0")},
        {"usuarios": "300", "servicio": "api", **_recurso("235.5", "316.54")},
        {"usuarios": "300", "servicio": "mlflow", **_recurso("2.15", "2176.58")},
    ]

    rows = build_cost_table(
        load_summary, capacity_steps, resource_summary, resource_steps, HOURLY_USD
    )
    by_name = {row["escenario"]: row for row in rows}

    assert list(by_name) == ["Base", "S1", "S2", "S3", "S4", "S5-150", "S5-300"]
    assert "S5" not in by_name
    assert by_name["Base"]["costo_por_inferencia_usd"] == ""
    assert by_name["Base"]["costo_horario_usd"] == "0.145"
    assert by_name["S5-150"]["rps"] == "96.9"
    assert by_name["S5-300"]["rps"] == "147.1"
    assert by_name["S5-150"]["punto_recomendado"] == "si"
    assert by_name["S5-300"]["punto_recomendado"] == "no"
    assert by_name["S5-150"]["costo_por_inferencia_usd"] == format(
        HOURLY_USD / (Decimal("96.9") * 3600), "f"
    )
    assert load_errors_are_zero(rows)
