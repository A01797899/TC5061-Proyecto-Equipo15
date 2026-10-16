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
