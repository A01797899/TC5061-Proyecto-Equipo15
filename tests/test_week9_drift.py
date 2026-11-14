"""Pruebas del monitoreo de drift con Evidently de la semana 9 (Champion P1)."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import drift_report as dr
from mibici.monitoring.prepare_data import COLUMNAS_P1, CURRENT_P1, FEATURES_P1, REFERENCE_P1

ESTACIONES = (1.0, 2.0, 3.0)


def panel(inicio, fin, factor=1.0, semilla=0):
    rng = np.random.default_rng(semilla)
    horas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    df = pd.MultiIndex.from_product([ESTACIONES, horas], names=["station", "hour"]).to_frame(
        index=False
    )
    hora = df["hour"].dt.hour
    base = 1.0 + df["station"]
    perfil = np.exp(-((hora - 8) ** 2) / 8) + np.exp(-((hora - 18) ** 2) / 10) + 0.1
    lam = base * perfil * factor
    df["y"] = rng.poisson(lam).astype("float64")
    df["hour_of_day"] = hora.astype("float64")
    df["weekday"] = df["hour"].dt.dayofweek.astype("float64")
    df["fin_semana"] = (df["weekday"] >= 5).astype("float64")
    for rezago in ("1h", "24h", "168h"):
        df[f"salidas_lag_{rezago}"] = rng.poisson(lam).astype("float64")
    df["media_estacion"] = base.astype("float64")
    df["media_estacion_dia_hora"] = (base * perfil).astype("float64")
    return df[COLUMNAS_P1]


@pytest.fixture(scope="module")
def conjuntos():
    referencia = panel("2024-01-08", "2024-09-01", semilla=1)
    actual = panel("2025-01-01", "2025-04-01", factor=0.6, semilla=2)
    return referencia, actual


@pytest.fixture(scope="module")
def escenario_principal(conjuntos):
    referencia, actual = conjuntos
    return dr.Escenario("principal", "prueba", referencia, actual)


@pytest.fixture(scope="module")
def drift_principal(escenario_principal):
    return dr.resultados_drift(dr.ejecutar(escenario_principal).dict())


def test_columnas_monitoreo_son_las_8_variables_y_el_objetivo():
    assert dr.COLUMNAS_MONITOREO == [*FEATURES_P1, "y"]
    assert set(dr.CATEGORICAS) | set(dr.NUMERICAS) == set(FEATURES_P1)
    assert not set(dr.CATEGORICAS) & set(dr.NUMERICAS)


def test_para_evidently_convierte_categoricas_sin_tocar_el_original(conjuntos):
    referencia, _ = conjuntos
    salida = dr.para_evidently(referencia)
    assert list(salida.columns) == ["station", "hour", *dr.COLUMNAS_MONITOREO]
    for col in ["station", *dr.CATEGORICAS]:
        assert salida[col].dtype == "int64"
    for col in [*dr.NUMERICAS, "y"]:
        assert salida[col].dtype == "float64"
    assert (referencia[FEATURES_P1].dtypes == "float64").all()


def test_definicion_datos_separa_tipos_y_llaves():
    definicion = dr.definicion_datos()
    assert definicion.id_column == "station"
    assert definicion.timestamp == "hour"
    assert set(definicion.categorical_columns) == set(dr.CATEGORICAS)
    assert set(definicion.numerical_columns) == {*dr.NUMERICAS, "y"}


def test_validar_columnas_rechaza_faltantes_y_nulos(conjuntos):
    referencia, _ = conjuntos
    with pytest.raises(ValueError, match="faltan columnas"):
        dr.validar_columnas(referencia.drop(columns="salidas_lag_24h"), "referencia")
    con_nulo = referencia.copy()
    con_nulo.loc[0, "media_estacion"] = np.nan
    with pytest.raises(ValueError, match="nulos"):
        dr.validar_columnas(con_nulo, "referencia")


def test_cargar_conjuntos_sin_parquets_explica_como_generarlos(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        dr.cargar_conjuntos(tmp_path)


def test_cargar_conjuntos_lee_y_muestrea_con_semilla(tmp_path, conjuntos):
    referencia, actual = conjuntos
    referencia.to_parquet(tmp_path / REFERENCE_P1)
    actual.to_parquet(tmp_path / CURRENT_P1)
    completo_ref, completo_act = dr.cargar_conjuntos(tmp_path)
    assert len(completo_ref) == len(referencia) and len(completo_act) == len(actual)
    muestra_a, _ = dr.cargar_conjuntos(tmp_path, muestra=500)
    muestra_b, _ = dr.cargar_conjuntos(tmp_path, muestra=500)
    assert len(muestra_a) == 500
    pd.testing.assert_frame_equal(muestra_a, muestra_b)


def test_armar_escenarios_incluye_control_estacional_y_meses(conjuntos):
    referencia, actual = conjuntos
    escenarios = {e.nombre: e for e in dr.armar_escenarios(referencia, actual)}
    assert list(escenarios) == [
        "principal",
        "estacional",
        "mes_2025-01",
        "mes_2025-02",
        "mes_2025-03",
    ]
    horas = pd.to_datetime(escenarios["estacional"].referencia["hour"])
    assert horas.min() >= pd.Timestamp("2024-01-01")
    assert horas.max() < pd.Timestamp("2024-04-01")
    assert len(escenarios["principal"].referencia) == len(referencia)
    total_meses = sum(len(escenarios[f"mes_2025-0{m}"].actual) for m in (1, 2, 3))
    assert total_meses == len(actual)


@pytest.mark.parametrize(
    ("metodo", "puntaje", "esperado"),
    [
        ("wasserstein", 0.15, True),
        ("wasserstein", 0.10, True),
        ("jensenshannon", 0.05, False),
        ("K-S p_value", 0.01, True),
        ("chi-square p_value", 0.30, False),
    ],
)
def test_hay_drift_respeta_la_regla_de_cada_metodo(metodo, puntaje, esperado):
    assert dr.hay_drift(metodo, puntaje, 0.1 if "p_value" not in metodo else 0.05) is esperado


def test_resultados_drift_usa_metodos_fijos_aunque_la_muestra_sea_chica(drift_principal):
    assert list(drift_principal["variable"]) == dr.COLUMNAS_MONITOREO
    metodos = drift_principal.set_index("variable")["metodo"].str.lower()
    for col in dr.CATEGORICAS:
        assert "jensen" in metodos[col]
    for col in [*dr.NUMERICAS, "y"]:
        assert "wasserstein" in metodos[col]
    assert (drift_principal["umbral"] == dr.UMBRAL_DRIFT).all()


def test_detecta_el_cambio_de_demanda_y_no_las_variables_estables(drift_principal):
    drift = drift_principal.set_index("variable")["drift"]
    for col in ["y", "salidas_lag_1h", "salidas_lag_24h", "salidas_lag_168h"]:
        assert drift[col], f"{col} debía derivar con 40% menos demanda"
    for col in ["hour_of_day", "media_estacion", "media_estacion_dia_hora"]:
        assert not drift[col], f"{col} no cambia entre conjuntos"


def test_resultados_drift_falla_si_no_hay_metricas_de_drift():
    with pytest.raises(ValueError, match="ValueDrift"):
        dr.resultados_drift({"metrics": [{"config": {"type": "x:RowCount"}, "value": 1}]})


def test_resumen_pruebas_cuenta_estados_y_lista_fallas():
    snap = {
        "tests": [
            {"status": "TestStatus.SUCCESS", "name": "a"},
            {"status": "TestStatus.FAIL", "name": "Value Drift for column y"},
            {"status": "FAIL", "name": "Row count"},
        ]
    }
    resumen = dr.resumen_pruebas(snap)
    assert resumen["conteo"] == {"SUCCESS": 1, "FAIL": 2}
    assert resumen["no_pasaron"] == ["FAIL: Value Drift for column y", "FAIL: Row count"]


def test_reporte_html_de_evidently_se_genera(escenario_principal, tmp_path):
    ruta = tmp_path / "reporte.html"
    dr.ejecutar(escenario_principal).save_html(str(ruta))
    contenido = ruta.read_text(encoding="utf-8")
    assert ruta.stat().st_size > 10_000
    assert "salidas_lag_1h" in contenido


def test_calidad_detecta_duplicados_rangos_y_estaciones_sin_salidas(conjuntos):
    _, actual = conjuntos
    limpio = dr.calidad(actual)
    assert limpio["duplicados_estacion_hora"] == 0
    assert limpio["nulos"] == 0 and limpio["tipos_float64"]
    assert limpio["estaciones_sin_salidas"] == []

    sucio = pd.concat([actual, actual.head(2)], ignore_index=True)
    sucio.loc[0, "hour_of_day"] = 25.0
    sucio.loc[1, "salidas_lag_1h"] = -1.0
    sucio.loc[sucio["station"] == 3.0, "y"] = 0.0
    resultado = dr.calidad(sucio)
    assert resultado["duplicados_estacion_hora"] == 2
    assert resultado["fuera_de_rango"]["hour_of_day"] == 1
    assert resultado["negativos"]["salidas_lag_1h"] == 1
    assert resultado["estaciones_sin_salidas"] == [3]


def test_demanda_por_estacion_mide_la_brecha_con_la_media_congelada(conjuntos):
    referencia, actual = conjuntos
    tabla = dr.demanda_por_estacion(referencia, actual).set_index("station")
    assert set(tabla.index) == set(ESTACIONES)
    assert (tabla["cambio_pct"] < -20).all()
    assert (tabla["brecha_congelada"] < 0).all()
    esperado = tabla["demanda_act"] - tabla["media_congelada"]
    pd.testing.assert_series_equal(tabla["brecha_congelada"], esperado, check_names=False)


def test_generar_escribe_todos_los_artefactos(conjuntos, tmp_path):
    referencia, actual = conjuntos
    resumen = dr.generar(referencia, actual, tmp_path, html=False)
    for archivo in [
        "drift_por_variable.csv",
        "drift_escenarios.csv",
        "drift_por_mes.csv",
        "demanda_por_estacion.csv",
        "drift_resumen.json",
        "evidently_drift_principal.json",
        "evidently_drift_estacional.json",
        "figures/01_drift_por_variable.png",
        "figures/02_perfil_horario.png",
        "figures/03_distribuciones.png",
        "figures/04_drift_por_mes.png",
        "figures/05_estaciones_media_congelada.png",
    ]:
        assert (tmp_path / archivo).exists(), archivo
    guardado = json.loads((tmp_path / "drift_resumen.json").read_text(encoding="utf-8"))
    assert guardado["escenarios"] == resumen["escenarios"]
    assert set(guardado["escenarios"]) == {"principal", "estacional"}
    assert "y" in guardado["escenarios"]["principal"]["variables_con_drift"]
    assert set(guardado["por_mes"]) == {"2025-01", "2025-02", "2025-03"}
    assert guardado["configuracion"]["umbral"] == dr.UMBRAL_DRIFT
    por_variable = pd.read_csv(tmp_path / "drift_por_variable.csv")
    assert {"puntaje", "drift", "media_ref", "media_act"} <= set(por_variable.columns)


def test_main_corre_desde_la_linea_de_comandos(conjuntos, tmp_path, capsys):
    referencia, actual = conjuntos
    datos, salida = tmp_path / "datos", tmp_path / "salida"
    datos.mkdir()
    referencia.to_parquet(datos / REFERENCE_P1)
    actual.to_parquet(datos / CURRENT_P1)
    dr.main(["--datos", str(datos), "--salida", str(salida), "--muestra", "3000", "--sin-html"])
    assert (salida / "drift_resumen.json").exists()
    assert not list(salida.glob("*.html"))
    assert "[principal]" in capsys.readouterr().out
