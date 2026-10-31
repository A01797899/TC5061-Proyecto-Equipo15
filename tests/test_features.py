import pandas as pd
import pytest

from mibici.features import (
    calendar_features,
    complete_hourly_panel,
    hourly_arrivals,
    hourly_departures,
    hourly_flow_balance,
    hourly_usage_profiles,
    lag_features,
    proxy_risk_labels,
    trip_quality_flags,
)


@pytest.fixture
def viajes():
    # Lunes 1 de enero de 2024 y sabado 6 de enero.
    return pd.DataFrame(
        {
            "origen_id": pd.array([1, 1, 2, 1], dtype="Int32"),
            "destino_id": pd.array([2, 2, 1, 2], dtype="Int32"),
            "inicio_viaje": pd.to_datetime(
                ["2024-01-01 08:05", "2024-01-01 08:50", "2024-01-01 09:10", "2024-01-06 08:00"]
            ),
            "fin_viaje": pd.to_datetime(
                [
                    "2024-01-01 08:20:00",
                    "2024-01-01 09:05:00",
                    "2024-01-01 09:10:30",
                    "2024-01-06 12:00:00",
                ]
            ),
            "anio_nacimiento": pd.array([1990, 200, None, 2000], dtype="Int16"),
        }
    )


def fila(df, estacion, hora):
    """Fila unica de una estacion-hora."""
    return df[(df["estacion_id"] == estacion) & (df["hora"] == pd.Timestamp(hora))].squeeze()


def test_salidas_y_llegadas_por_hora(viajes):
    salidas = hourly_departures(viajes)
    assert salidas["salidas"].sum() == 4
    assert fila(salidas, 1, "2024-01-01 08:00")["salidas"] == 2

    llegadas = hourly_arrivals(viajes)
    # El segundo viaje llega a las 9:05, asi que cuenta en la hora de las 9.
    assert fila(llegadas, 2, "2024-01-01 09:00")["llegadas"] == 1


def test_calendario(viajes):
    cal = calendar_features(hourly_departures(viajes))
    lunes = cal[cal["hora"] == pd.Timestamp("2024-01-01 08:00")].iloc[0]
    sabado = cal[cal["hora"] == pd.Timestamp("2024-01-06 08:00")].iloc[0]
    assert (lunes["dia_semana"], lunes["hora_dia"], lunes["fin_semana"]) == (0, 8, 0)
    assert (sabado["dia_semana"], sabado["fin_semana"]) == (5, 1)


def test_balance_rellena_con_cero(viajes):
    flujos = hourly_flow_balance(hourly_departures(viajes), hourly_arrivals(viajes))
    # Estacion 2 a las 8: no sale nada y llega un viaje.
    f = fila(flujos, 2, "2024-01-01 08:00")
    assert (f["salidas"], f["llegadas"], f["balance"]) == (0, 1, 1)


def test_etiqueta_proxy_con_claves_de_distinto_tipo():
    flujos = pd.DataFrame(
        {
            "estacion_id": ["1", "2", "3"],
            "hora": pd.to_datetime(["2024-01-01 08:00"] * 3),
            "balance": [-5, 1, 4],
        }
    )
    capacidades = pd.DataFrame(
        {"estacion_id": pd.array([1, 2, 3], dtype="Int32"), "capacidad": [20, 20, 0]}
    )
    out = proxy_risk_labels(flujos, capacidades, threshold=0.25)
    # La estacion 3 sale porque su capacidad es cero.
    assert out["estacion_id"].tolist() == ["1", "2"]
    assert out["riesgo_proxy"].tolist() == [1, 0]


def test_umbral_debe_ser_positivo():
    with pytest.raises(ValueError):
        proxy_risk_labels(pd.DataFrame(), pd.DataFrame(), threshold=0)


def test_perfiles_por_tipo_de_dia(viajes):
    perfiles = hourly_usage_profiles(viajes)
    fila = perfiles.query("estacion_id == 1 and tipo_dia == 'laboral' and hora_dia == 8")
    assert fila["salidas"].item() == 2
    assert set(perfiles["tipo_dia"]) == {"laboral", "fin_semana"}


def test_perfiles_rechazan_conteos_negativos():
    agregados = pd.DataFrame(
        {"origen_id": [1], "inicio_viaje": pd.to_datetime(["2024-01-01 08:00"]), "n": [-1]}
    )
    with pytest.raises(ValueError):
        hourly_usage_profiles(agregados, count_column="n")


def test_panel_completo(viajes):
    estaciones = pd.Index([1, 2, 3])
    panel = complete_hourly_panel(
        hourly_departures(viajes), estaciones, "2024-01-01 08:00", "2024-01-01 10:00"
    )
    assert len(panel) == 3 * 2
    assert panel["salidas"].sum() == 3
    # La estacion 3 nunca tuvo viajes y aun asi aparece con cero.
    assert panel.query("estacion_id == 3")["salidas"].tolist() == [0, 0]
    # El viaje del sabado queda fuera del intervalo [start, end).
    assert panel["hora"].max() == pd.Timestamp("2024-01-01 09:00")


def test_rezagos_solo_usan_el_pasado():
    panel = pd.DataFrame(
        {
            "estacion_id": [1, 1, 1],
            "hora": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00", "2024-01-01 11:00"]),
            "salidas": [5, 7, 9],
        }
    )
    out = lag_features(panel, lags=(1,))
    # La primera hora no tiene historia.
    assert pd.isna(out["salidas_lag_1h"][0])
    assert out["salidas_lag_1h"][1] == 5
    # A las 11 falta la hora de las 10: queda NaN en lugar de tomar el valor de las 9.
    assert pd.isna(out["salidas_lag_1h"][2])


def test_rezago_no_positivo_falla():
    with pytest.raises(ValueError):
        lag_features(pd.DataFrame(columns=["estacion_id", "hora", "salidas"]), lags=(0,))


def test_marcas_de_calidad(viajes):
    out = trip_quality_flags(viajes)
    assert len(out) == len(viajes)
    assert out["duracion_min"].tolist() == [15, 15, 0.5, 240]
    assert out["duracion_atipica"].tolist() == [0, 0, 1, 1]
    # Anio 200 da una edad imposible; el anio vacio no se marca.
    assert out["edad_atipica"].tolist() == [0, 1, 0, 0]

def test_sin_viajes_devuelve_tablas_vacias():
    vacio = pd.DataFrame(
        {
            "origen_id": pd.array([], dtype="Int32"),
            "destino_id": pd.array([], dtype="Int32"),
            "inicio_viaje": pd.to_datetime([]),
            "fin_viaje": pd.to_datetime([]),
        }
    )
    salidas = hourly_departures(vacio)
    assert salidas.empty
    assert list(salidas.columns) == ["estacion_id", "hora", "salidas"]
    flujos = hourly_flow_balance(salidas, hourly_arrivals(vacio))
    assert flujos.empty
    assert hourly_usage_profiles(vacio).empty


def test_conteos_ignoran_nulos(viajes):
    con_nulos = pd.concat(
        [
            viajes,
            pd.DataFrame(
                {
                    "origen_id": pd.array([None, 1], dtype="Int32"),
                    "destino_id": pd.array([2, None], dtype="Int32"),
                    "inicio_viaje": pd.to_datetime(["2024-01-01 08:30", None]),
                    "fin_viaje": pd.to_datetime(["2024-01-01 08:40", "2024-01-01 08:45"]),
                }
            ),
        ],
        ignore_index=True,
    )
    assert hourly_departures(con_nulos)["salidas"].sum() == 4
    assert hourly_arrivals(con_nulos)["llegadas"].sum() == 5
    assert hourly_usage_profiles(con_nulos)["salidas"].sum() == 4


def test_ids_como_texto_de_los_csv_crudos(viajes):
    crudos = viajes.assign(
        origen_id=viajes["origen_id"].astype(str),
        inicio_viaje=viajes["inicio_viaje"].astype(str),
    )
    salidas = hourly_departures(crudos)
    assert fila(salidas, "1", "2024-01-01 08:00")["salidas"] == 2


@pytest.mark.parametrize(
    "funcion, columna",
    [
        (hourly_departures, "origen_id"),
        (hourly_arrivals, "fin_viaje"),
        (hourly_usage_profiles, "inicio_viaje"),
        (trip_quality_flags, "anio_nacimiento"),
    ],
)
def test_columnas_faltantes_fallan(viajes, funcion, columna):
    with pytest.raises(KeyError):
        funcion(viajes.drop(columns=columna))


def test_calendario_no_modifica_la_entrada_y_acepta_texto():
    horas = pd.DataFrame({"hora": ["2024-01-07 23:00:00", "2024-01-08 00:00:00"]})
    original = horas.copy()
    cal = calendar_features(horas)
    pd.testing.assert_frame_equal(horas, original)
    # Domingo 23 h y lunes 0 h.
    assert cal["dia_semana"].tolist() == [6, 0]
    assert cal["hora_dia"].tolist() == [23, 0]
    assert cal["fin_semana"].tolist() == [1, 0]


def test_balance_negativo_cuando_solo_hay_salidas():
    salidas = pd.DataFrame(
        {"estacion_id": [1], "hora": pd.to_datetime(["2024-01-01 08:00"]), "salidas": [3]}
    )
    llegadas = pd.DataFrame(columns=["estacion_id", "hora", "llegadas"]).astype(
        {"estacion_id": "int64", "hora": "datetime64[ns]", "llegadas": "int64"}
    )
    f = hourly_flow_balance(salidas, llegadas).iloc[0]
    assert (f["salidas"], f["llegadas"], f["balance"]) == (3, 0, -3)


@pytest.fixture
def flujos():
    return pd.DataFrame(
        {
            "estacion_id": pd.array([1, 2, 3, 4, 5], dtype="Int32"),
            "hora": pd.to_datetime(["2024-01-01 08:00"] * 5),
            "balance": [5, -4, 3, 2, 9],
        }
    )


def test_etiqueta_en_el_umbral_exacto_es_riesgo(flujos):
    capacidades = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [20, 20]})
    out = proxy_risk_labels(flujos, capacidades, threshold=0.25)
    # 5/20 = 0.25 cuenta como riesgo; 4/20 = 0.20 no.
    assert out["presion"].tolist() == [0.25, 0.2]
    assert out["riesgo_proxy"].tolist() == [1, 0]


def test_capacidades_invalidas_quedan_fuera(flujos):
    capacidades = pd.DataFrame(
        {"estacion_id": [1, 2, 3, 4], "capacidad": ["20", "no aplica", None, -5]}
    )
    out = proxy_risk_labels(flujos, capacidades)
    # Solo la estacion 1 tiene capacidad valida; la 5 no esta en la nomenclatura.
    assert out["estacion_id"].tolist() == [1]


def test_capacidad_repetida_usa_la_primera(flujos):
    capacidades = pd.DataFrame({"estacion_id": [1, 1], "capacidad": [20, 10]})
    out = proxy_risk_labels(flujos, capacidades)
    assert len(out) == 1
    assert out["capacidad"].item() == 20


def test_perfiles_con_conteos_agregados_y_texto_invalido():
    agregados = pd.DataFrame(
        {
            "origen_id": [1, 1, 1],
            "inicio_viaje": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-08 08:00", "2024-01-06 08:00"]
            ),
            "n": [3, 4, 2],
        }
    )
    perfiles = hourly_usage_profiles(agregados, count_column="n")
    assert perfiles.query("tipo_dia == 'laboral'")["salidas"].item() == 7
    assert perfiles.query("tipo_dia == 'fin_semana'")["salidas"].item() == 2
    with pytest.raises(ValueError):
        hourly_usage_profiles(agregados.assign(n=["3", "x", "2"]), count_column="n")


def test_panel_sin_salidas_y_con_intervalo_vacio():
    sin_salidas = pd.DataFrame(columns=["estacion_id", "hora", "salidas"]).astype(
        {"estacion_id": "int64", "hora": "datetime64[ns]", "salidas": "int64"}
    )
    panel = complete_hourly_panel(
        sin_salidas, pd.Index([1, 2]), "2024-01-01 00:00", "2024-01-02 00:00"
    )
    assert len(panel) == 2 * 24
    assert (panel["salidas"] == 0).all()
    vacio = complete_hourly_panel(
        sin_salidas, pd.Index([1]), "2024-01-01 00:00", "2024-01-01 00:00"
    )
    assert vacio.empty


def test_panel_excluye_estaciones_no_listadas(viajes):
    panel = complete_hourly_panel(
        hourly_departures(viajes), pd.Index([1]), "2024-01-01 08:00", "2024-01-01 10:00"
    )
    assert set(panel["estacion_id"]) == {1}
    assert panel["salidas"].sum() == 2


def test_rezagos_no_se_mezclan_entre_estaciones():
    horas = pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"])
    panel = pd.DataFrame(
        {"estacion_id": [1, 1, 2, 2], "hora": horas.append(horas), "salidas": [5, 7, 100, 200]}
    )
    out = lag_features(panel, lags=(1,)).set_index(["estacion_id", "hora"])["salidas_lag_1h"]
    assert out[(1, horas[1])] == 5
    assert out[(2, horas[1])] == 100
    assert pd.isna(out[(2, horas[0])])


def test_rezagos_diario_y_semanal_sin_fuga():
    horas = pd.date_range("2024-01-01 00:00", periods=24 * 8 + 1, freq="h")
    panel = pd.DataFrame({"estacion_id": 1, "hora": horas, "salidas": range(len(horas))})
    original = panel.copy()
    out = lag_features(panel)
    pd.testing.assert_frame_equal(panel, original)
    ultima = out.iloc[-1]
    assert ultima["salidas_lag_1h"] == ultima["salidas"] - 1
    assert ultima["salidas_lag_24h"] == ultima["salidas"] - 24
    assert ultima["salidas_lag_168h"] == ultima["salidas"] - 168
    # Ningun rezago es igual o posterior al valor de la hora objetivo.
    for h in (1, 24, 168):
        validos = out[f"salidas_lag_{h}h"].dropna()
        assert (validos < out.loc[validos.index, "salidas"]).all()


def test_rezagos_de_otra_columna():
    panel = pd.DataFrame(
        {
            "estacion_id": [1, 1],
            "hora": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"]),
            "llegadas": [2, 4],
        }
    )
    out = lag_features(panel, column="llegadas", lags=(1,))
    assert out["llegadas_lag_1h"].tolist()[1] == 2


@pytest.mark.parametrize("lags", [(-1,), (1, 0)])
def test_rezago_negativo_o_cero_falla(lags):
    with pytest.raises(ValueError):
        lag_features(pd.DataFrame(columns=["estacion_id", "hora", "salidas"]), lags=lags)


def test_marcas_de_calidad_en_los_limites():
    inicio = pd.Timestamp("2024-01-01 08:00")
    viajes = pd.DataFrame(
        {
            "inicio_viaje": [inicio] * 4,
            "fin_viaje": [
                inicio + pd.Timedelta(minutes=1),
                inicio + pd.Timedelta(minutes=180),
                inicio + pd.Timedelta(minutes=181),
                inicio - pd.Timedelta(minutes=5),
            ],
            # Edades de 10, 90, 9 y 91 anios en 2024.
            "anio_nacimiento": pd.array([2014, 1934, 2015, 1933], dtype="Int16"),
        }
    )
    original = viajes.copy()
    out = trip_quality_flags(viajes)
    pd.testing.assert_frame_equal(viajes, original)
    # Los umbrales son inclusivos; un fin antes del inicio es atipico.
    assert out["duracion_atipica"].tolist() == [0, 0, 1, 1]
    assert out["edad"].tolist() == [10, 90, 9, 91]
    assert out["edad_atipica"].tolist() == [0, 0, 1, 1]
