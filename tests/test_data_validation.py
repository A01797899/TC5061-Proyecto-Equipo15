import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import mibici.data_validation as dv

FEATURES = ["hour_of_day", "weekday", "fin_semana", "salidas_lag_1h", "media_estacion"]


def panel_sintetico(estaciones=4, dias=40, cambio_test=0.0, semilla=0):
    # Panel estación-hora con la misma forma que datos_p1: train, val y test cronológicos.
    rng = np.random.default_rng(semilla)
    horas = pd.date_range("2024-01-01", periods=dias * 24, freq="h")
    panel = pd.MultiIndex.from_product(
        [[f"E{i}" for i in range(estaciones)], horas], names=["station", "hour"]
    ).to_frame(index=False)
    corte_val, corte_test = horas[int(len(horas) * 0.6)], horas[int(len(horas) * 0.8)]
    panel["split"] = np.where(
        panel["hour"] < corte_val, "train", np.where(panel["hour"] < corte_test, "val", "test")
    )
    media = np.where(panel["split"] == "test", 2 + cambio_test, 2)
    panel["y"] = rng.poisson(media)
    panel["hour_of_day"] = panel["hour"].dt.hour
    panel["weekday"] = panel["hour"].dt.weekday
    panel["fin_semana"] = (panel["weekday"] >= 5).astype("int8")
    panel["salidas_lag_1h"] = panel.groupby("station")["y"].shift(1).astype("float32")
    panel["media_estacion"] = rng.normal(2, 0.3, len(panel)).astype("float32")
    return panel


@pytest.fixture(scope="module")
def dc():
    # No uso importorskip: importaría deepchecks antes de los parches y fallaría.
    if importlib.util.find_spec("deepchecks") is None:
        pytest.skip("deepchecks no está instalado; viene en requirements-dev.txt")
    return dv.cargar_deepchecks()


def test_acepta_panel_valido():
    dv.validar_columnas(panel_sintetico(), FEATURES)


def test_rechaza_panel_vacio():
    with pytest.raises(ValueError, match="vacío"):
        dv.validar_columnas(panel_sintetico().iloc[0:0], FEATURES)


@pytest.mark.parametrize("columna", ["y", "split", "hour", "salidas_lag_1h"])
def test_rechaza_columnas_faltantes(columna):
    with pytest.raises(ValueError, match="Faltan columnas"):
        dv.validar_columnas(panel_sintetico().drop(columns=columna), FEATURES)


@pytest.mark.parametrize("columna", ["pressure", "salidas", "llegadas"])
def test_rechaza_columnas_de_la_hora_objetivo_como_predictor(columna):
    panel = panel_sintetico().assign(**{columna: 1.0})
    with pytest.raises(ValueError, match="hora objetivo"):
        dv.validar_columnas(panel, [*FEATURES, columna])


def test_exige_train_y_test():
    panel = panel_sintetico()
    with pytest.raises(ValueError, match="train y test"):
        dv.validar_columnas(panel[panel["split"] != "test"], FEATURES)


def test_resumen_cuenta_nulos_de_rezagos_y_llave_unica():
    panel = panel_sintetico(estaciones=4)
    resumen = dv.resumen_exacto(panel, FEATURES)
    # Cada estación pierde solo su primera hora en el rezago de 1 h.
    assert resumen["nulos_train_por_columna"] == {"salidas_lag_1h": 4}
    assert resumen["filas_train_descartadas_por_nulos"] == 4
    assert resumen["llave_estacion_hora_duplicada"] == 0
    assert resumen["nulos_test"] == 0


def test_resumen_detecta_llave_estacion_hora_duplicada():
    panel = panel_sintetico()
    panel = pd.concat([panel, panel.head(3)], ignore_index=True)
    assert dv.resumen_exacto(panel, FEATURES)["llave_estacion_hora_duplicada"] == 3


def test_dividir_respeta_muestra_semilla_y_orden():
    panel = panel_sintetico()
    a = dv.dividir(panel, FEATURES, muestra=300, semilla=7)
    b = dv.dividir(panel, FEATURES, muestra=300, semilla=7)
    assert len(a["train"]) == len(a["test"]) == 300
    pd.testing.assert_frame_equal(a["train"], b["train"])
    assert a["train"]["hour"].is_monotonic_increasing
    assert set(a) == {"train", "test"}


def test_dividir_no_quita_nulos():
    panel = panel_sintetico()
    partes = dv.dividir(panel, FEATURES, muestra=0)
    assert partes["train"]["salidas_lag_1h"].isna().sum() == 4


def test_compatibilidad_aplicada(dc):
    from sklearn.metrics import get_scorer

    assert np.Inf == np.inf
    assert get_scorer("max_error") is not None


def test_correlacion_sin_pares_altos_aprueba(dc):
    # Con pandas 3 la condición original de deepchecks marcaba todos estos pares.
    matriz = pd.DataFrame(
        [[1.0, 0.55, 0.11], [0.55, 1.0, 0.10], [0.11, 0.10, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    _, evaluar = dv.condicion_pares_correlacionados(dc)
    resultado = evaluar(matriz)
    assert resultado.category == dc.ConditionCategory.PASS
    assert "máximo 0.55" in resultado.details


def test_correlacion_marca_solo_el_par_alto(dc):
    matriz = pd.DataFrame(
        [[1.0, 0.95, 0.11], [0.95, 1.0, 0.10], [0.11, 0.10, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    _, evaluar = dv.condicion_pares_correlacionados(dc)
    resultado = evaluar(matriz)
    assert resultado.category == dc.ConditionCategory.FAIL
    assert "('a', 'b')" in resultado.details
    assert "'c'" not in resultado.details


@pytest.mark.parametrize(
    "proporciones, esperado",
    [({0: 0.95, 1: 0.05}, "WARN"), ({0: 0.6, 1: 0.4}, "PASS"), ({0: 0.9, 1: 0.1}, "PASS")],
)
def test_desbalance_avisa_solo_si_la_minoritaria_es_menor_al_umbral(dc, proporciones, esperado):
    _, evaluar = dv.condicion_desbalance(dc)
    assert evaluar(proporciones).category.value == esperado


@pytest.mark.parametrize("pregunta", ["p1", "p2"])
def test_suites_excluyen_incompatibles_y_ajustan_por_tipo(dc, pregunta):
    integridad, train_test = dv.preparar_suites(pregunta, dc)
    nombres = {type(c).__name__ for s in (integridad, train_test) for c in s.checks.values()}
    assert not nombres & (set(dv.CHECKS_INCOMPATIBLES) | set(dv.CHECKS_NO_APLICAN))
    assert "FeatureFeatureCorrelation" in nombres and "PercentOfNulls" in nombres
    es_clasificacion = pregunta == "p2"
    assert ("ClassImbalance" in nombres) is es_clasificacion
    assert ("ConflictingLabels" in nombres) is es_clasificacion


@pytest.mark.parametrize("cambio, detecta", [(0.0, False), (4.0, True)])
def test_label_drift_detecta_cambio_entre_train_y_test(dc, cambio, detecta):
    from deepchecks.tabular.checks import LabelDrift

    partes = dv.dividir(panel_sintetico(cambio_test=cambio), FEATURES, muestra=0)
    train, test = (dv.construir_dataset(partes[n], FEATURES, "p1", dc) for n in ("train", "test"))
    resultado = LabelDrift().add_condition_drift_score_less_than().run(train, test)
    assert (not resultado.passed_conditions()) is detecta


def test_validar_pregunta_genera_reportes_y_los_reemplaza(dc, tmp_path, monkeypatch):
    panel = panel_sintetico(estaciones=3, dias=30)
    monkeypatch.setattr(dv, "cargar_panel", lambda *a: (panel, FEATURES))
    args = SimpleNamespace(raw=tmp_path, chunk_size=1, muestra=0, salida=tmp_path / "reportes")
    for _ in range(2):
        resumen = dv.validar_pregunta("p1", args, dc)
    # La segunda corrida reemplaza los HTML en lugar de crear "nombre (1).html".
    archivos = sorted(p.name for p in args.salida.iterdir())
    assert archivos == [
        "p1_integridad_test.html",
        "p1_integridad_train.html",
        "p1_resumen.json",
        "p1_train_test.html",
    ]
    assert set(resumen["suites"]) == {"integridad_train", "integridad_test", "train_test"}
    # Una sola copia de plotly.js desde CDN: sin ella cada gráfica incrusta ~5 MB.
    reporte = (args.salida / "p1_train_test.html").read_text(encoding="utf-8")
    assert "plotly.js v" not in reporte
    assert reporte.count("cdn.jsdelivr.net/npm/plotly.js-dist-min") == 1
    assert "Plotly.newPlot" in reporte
    estados = {f["estado"] for filas in resumen["suites"].values() for f in filas}
    assert "no_ejecutado" not in estados
