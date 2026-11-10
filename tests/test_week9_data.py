"""Pruebas de la preparación de datos de la semana 9 (referencia 2024 y actual 2025)."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici.api.schemas import P1_FEATURES
from mibici.config import PROCESSED_DATA_DIR
from mibici.modeling import p1_baseline
from mibici.modeling.p1_p2_candidates import agregar_historicos, agregar_rezagos, asignar_split
from mibici.monitoring import data_quality as dq
from mibici.monitoring import prepare_data as pd9

ENCABEZADO = (
    '"Viaje_Id","Usuario_Id","Genero","Año_de_nacimiento",'
    '"Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"'
)


def escribir_csv(ruta, inicios, encoding="latin1", encabezado=ENCABEZADO, origen="5"):
    # Escribo un CSV mensual con el mismo formato que publica MiBici.
    filas = [encabezado]
    for i, inicio in enumerate(inicios, start=1):
        fin = (pd.Timestamp(inicio) + pd.Timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
        filas.append(f'"{i}","{100 + i}","M","1990","{inicio}","{fin}","{origen}","7"')
    ruta.write_bytes(("\n".join(filas) + "\n").encode(encoding))
    return ruta


def viajes(registros):
    # Armo viajes ya normalizados como los que devuelve leer_viajes().
    df = pd.DataFrame(registros, columns=["origen_id", "inicio_viaje"])
    n = len(df)
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"])
    return pd.DataFrame(
        {
            "viaje_id": pd.array(range(1, n + 1), dtype="Int64"),
            "usuario_id": pd.array(range(1, n + 1), dtype="Int64"),
            "genero": pd.Categorical(["M"] * n),
            "anio_nacimiento": pd.array([1990] * n, dtype="Int16"),
            "inicio_viaje": df["inicio_viaje"],
            "fin_viaje": df["inicio_viaje"] + pd.Timedelta(minutes=10),
            "origen_id": pd.array(df["origen_id"], dtype="Int32"),
            "destino_id": pd.array([1] * n, dtype="Int32"),
        }
    )


def viajes_sinteticos(semilla=0, extra_2025=0):
    # Genero viajes de dos estaciones en 2024 y 2025, y una estación nueva solo en 2025.
    rng = np.random.default_rng(semilla)
    horas_2024 = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")
    horas_2025 = pd.date_range("2025-01-01", "2025-03-31 23:00", freq="h")
    ref = [(e, h) for e in (1, 2) for h in rng.choice(horas_2024, 3000)]
    act = [(e, h) for e in (1, 2) for h in rng.choice(horas_2025, 800 + extra_2025)]
    act += [(9, h) for h in rng.choice(horas_2025, 50)]
    return viajes(ref), viajes(act)


@pytest.fixture(scope="module")
def panel():
    ref, act = viajes_sinteticos()
    return pd9.construir_panel_p1(ref, act)


@pytest.fixture(scope="module")
def datasets(panel):
    return pd9.separar_p1(panel)


def test_carga_csv_detecta_encoding_y_formato(tmp_path):
    ruta = escribir_csv(tmp_path / "enero.csv", ["2025-01-02 08:00:00", "2025-01-31 23:59:00"])
    viajes_mes, info = pd9.leer_mes(ruta, 1)
    assert len(viajes_mes) == 2
    assert info["encoding"] == "latin1"
    assert info["formato_fecha"] == "%Y-%m-%d %H:%M:%S"
    assert info["fechas_invalidas"] == 0 and info["inicios_fuera_del_mes"] == 0
    utf8 = escribir_csv(tmp_path / "marzo.csv", ["2025-03-05 10:00:00"], encoding="utf-8")
    assert pd9.leer_mes(utf8, 3)[1]["encoding"] == "utf-8"


def test_formato_de_fecha_con_dia_primero(tmp_path):
    ruta = tmp_path / "sep.csv"
    ruta.write_text(
        ENCABEZADO + '\n"1","2","F","1990","01/09/2024 0:05","01/09/2024 0:20","5","7"\n',
        encoding="latin1",
    )
    assert pd9.detectar_formato_fecha(ruta, "latin1") == "%d/%m/%Y %H:%M"


def test_esquema_igual_a_2024_y_rechazo_de_columnas(tmp_path):
    ruta = escribir_csv(tmp_path / "ok.csv", ["2025-02-03 07:00:00"])
    assert pd9.leer_mes(ruta, 2)[1]["coincide_esquema_2024"]
    malo = escribir_csv(
        tmp_path / "malo.csv",
        ["2025-02-03 07:00:00"],
        encabezado=ENCABEZADO.replace("Genero", "Sexo"),
    )
    with pytest.raises(ValueError, match="Columna inesperada"):
        pd9.leer_mes(malo, 2)


def test_rechaza_archivo_de_otro_mes(tmp_path):
    ruta = escribir_csv(tmp_path / "febrero.csv", ["2025-01-10 09:00:00", "2025-01-11 09:00:00"])
    with pytest.raises(ValueError, match="no parece de 2025-02"):
        pd9.leer_mes(ruta, 2)


def test_cargar_actual_une_meses_en_orden_y_exige_archivos(tmp_path):
    escribir_csv(tmp_path / "datos_abiertos_2025_02.csv", ["2025-02-01 00:10:00"])
    escribir_csv(tmp_path / "datos_abiertos_2025_01.csv", ["2025-01-31 23:50:00"])
    actual, fuentes = pd9.cargar_actual(tmp_path, meses=(1, 2))
    assert actual["inicio_viaje"].is_monotonic_increasing
    assert [f["registros"] for f in fuentes] == [1, 1]
    with pytest.raises(FileNotFoundError, match="datos_abiertos_2025_03"):
        pd9.cargar_actual(tmp_path, meses=(3,))


def test_nulos_se_descartan_o_se_conservan_segun_la_regla():
    df = viajes([(1, "2025-01-05 10:00"), (2, "2025-01-06 11:00"), (1, "2024-12-31 23:00")])
    df.loc[1, "origen_id"] = pd.NA
    df["genero"] = pd.Categorical([None, "M", "F"])
    r = dq.evaluar_periodo(df, "2025-01-01", "2025-04-01", estaciones=[1, 2])
    assert r["descartados"] == {"sin_inicio_u_origen": 1, "fuera_del_periodo": 1, "total": 2}
    assert r["registros_validos"] == 1
    assert r["anomalias_conservadas"]["genero_vacio"] == 1
    assert r["estaciones"]["del_modelo_sin_viajes"] == [2]


def test_duplicados_de_viaje_y_de_llave(datasets):
    df = viajes([(1, "2025-01-05 10:00"), (1, "2025-01-05 10:30")])
    df.loc[1, "viaje_id"] = 1
    assert dq.evaluar_periodo(df, "2025-01-01", "2025-04-01", [1])["duplicados"]["viaje_id"] == 1
    _, actual = datasets
    assert not actual.duplicated(["station", "hour"]).any()
    with pytest.raises(ValueError, match="repetidas"):
        pd9.validar_p1(pd.concat([actual.head(2), actual.head(1)]), "duplicado")


def test_rezagos_solo_usan_horas_anteriores_incluso_al_iniciar_2025(panel):
    serie = panel[panel["station"] == 1].set_index("hour")["y"]
    for hora in (pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-02-10 15:00")):
        fila = panel[(panel["station"] == 1) & (panel["hour"] == hora)].iloc[0]
        for h in (1, 24, 168):
            assert fila[f"salidas_lag_{h}h"] == serie[hora - pd.Timedelta(hours=h)]


def test_medias_historicas_no_usan_datos_futuros():
    ref, act = viajes_sinteticos()
    _, mas_viajes = viajes_sinteticos(extra_2025=500)
    medias = ["media_estacion", "media_estacion_dia_hora"]
    base = pd9.separar_p1(pd9.construir_panel_p1(ref, act))[1][medias]
    otra = pd9.separar_p1(pd9.construir_panel_p1(ref, mas_viajes))[1][medias]
    pd.testing.assert_frame_equal(base, otra)


def test_caracteristicas_coinciden_con_p1_y_estaciones_del_modelo(datasets):
    referencia, actual = datasets
    assert pd9.FEATURES_P1 == list(P1_FEATURES)
    for df in datasets:
        assert list(df.columns) == ["station", "hour", "y", *P1_FEATURES]
    assert set(actual["station"]) == set(referencia["station"]) == {1, 2}


def test_las_ocho_caracteristicas_son_float64_sin_nulos(datasets):
    for df in datasets:
        assert (df[list(P1_FEATURES)].dtypes == "float64").all()
        assert df[list(P1_FEATURES)].notna().all().all()
    roto = datasets[1].copy()
    roto.loc[0, "salidas_lag_1h"] = np.nan
    with pytest.raises(ValueError, match="nulas o infinitas"):
        pd9.validar_p1(roto, "roto")


def test_particion_temporal_sin_superposicion(datasets):
    entrenamiento, evaluacion = pd9.particionar(datasets[1])
    assert entrenamiento["hour"].max() < evaluacion["hour"].min()
    assert evaluacion["hour"].min() == pd.Timestamp(pd9.INICIO_EVALUACION)
    assert len(entrenamiento) + len(evaluacion) == len(datasets[1])
    with pytest.raises(ValueError, match="vacía"):
        pd9.particionar(entrenamiento)


def test_reproducibilidad(tmp_path):
    ref, act = viajes_sinteticos(semilla=3)
    uno = pd9.separar_p1(pd9.construir_panel_p1(ref, act))
    dos = pd9.separar_p1(pd9.construir_panel_p1(ref, act))
    for a, b in zip(uno, dos):
        pd.testing.assert_frame_equal(a, b)
    rutas = [tmp_path / "a.json", tmp_path / "b.json"]
    for ruta in rutas:
        dq.generar_reporte(ref, act, [], ruta)
    assert rutas[0].read_text(encoding="utf-8") == rutas[1].read_text(encoding="utf-8")


def test_no_cambia_el_comportamiento_de_2024():
    ref, act = viajes_sinteticos()
    panel = pd9.construir_panel_p1(ref, act)
    # Reproduzco el pipeline original solo con 2024 para compararlo con el panel continuo.
    observados = pd9.salidas_por_hora(ref)
    estaciones = pd.Index(sorted(observados["station"].unique()), name="station")
    original = asignar_split(p1_baseline.panel(observados, estaciones, "2024-01-01", "2025-01-01"))
    original = agregar_rezagos(original.rename(columns={"y": "salidas"}), ["salidas"])
    original = agregar_historicos(original.rename(columns={"salidas": "y"}), "media")
    columnas = ["station", "hour", "y", "split", *P1_FEATURES]
    nuevo = panel[panel["hour"] < pd.Timestamp("2025-01-01")][columnas].reset_index(drop=True)
    original[list(P1_FEATURES)] = original[list(P1_FEATURES)].astype("float64")
    pd.testing.assert_frame_equal(nuevo, original[columnas].reset_index(drop=True))


def test_main_no_modifica_la_ingesta_de_2024(tmp_path, monkeypatch):
    ref, act = viajes_sinteticos()
    manifiesto_2024 = tmp_path / "manifiesto.json"
    manifiesto_2024.write_text('{"version_datos": "fija"}', encoding="utf-8")
    monkeypatch.setattr(pd9, "cargar_referencia", lambda: ref)
    monkeypatch.setattr(pd9, "cargar_actual", lambda directorio: (act, []))
    monkeypatch.setattr(pd9, "version_datos", lambda: "fija")
    manifiesto = tmp_path / "data_manifest.json"
    pd9.main(["--carpeta", str(tmp_path / "week9"), "--manifiesto", str(manifiesto)])
    salidas = {p.name for p in (tmp_path / "week9").iterdir()}
    assert salidas == {
        pd9.VIAJES_2025,
        pd9.REFERENCE_P1,
        pd9.CURRENT_P1,
        pd9.TRAIN_ALTERNATIVE,
        pd9.EVALUATION,
    }
    assert manifiesto_2024.read_text(encoding="utf-8") == '{"version_datos": "fija"}'
    datos = json.loads(manifiesto.read_text(encoding="utf-8"))
    assert datos["estaciones"]["nuevas_2025"] == [9]
    assert datos["champion"]["run_id"] == "6e3429f222a0407aa159dd9f46699365"


REALES = PROCESSED_DATA_DIR / "week9"


@pytest.mark.skipif(not (REALES / pd9.EVALUATION).exists(), reason="faltan los datasets reales")
def test_datasets_reales_cumplen_el_contrato():
    filas = {
        pd9.REFERENCE_P1: 2_059_056,
        pd9.CURRENT_P1: 781_920,
        pd9.TRAIN_ALTERNATIVE: 512_592,
        pd9.EVALUATION: 269_328,
    }
    for nombre, esperado in filas.items():
        df = pd.read_parquet(REALES / nombre)
        pd9.validar_p1(df, nombre)
        assert len(df) == esperado and df["station"].nunique() == 362
