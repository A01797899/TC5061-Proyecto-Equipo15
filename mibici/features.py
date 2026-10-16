"""Feature engineering reutilizable para P1, P2 y P3.

Todas las funciones reciben y devuelven DataFrames; no leen ni escriben archivos,
asi que funcionan igual con el Parquet de ``dataset.py`` (ids ``Int32``) o con los
CSV crudos que leen los baselines (ids como texto).

Regla general contra fuga de informacion: lo que se calcula en la hora objetivo
(salidas, llegadas, balance, presion) solo sirve como etiqueta. Como predictores
solo se usan el calendario de la hora objetivo y valores de horas anteriores
(``lag_features``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mibici.config import (
    ANIO_DATOS,
    DURACION_MAX_MINUTOS,
    DURACION_MIN_MINUTOS,
    EDAD_MAX,
    EDAD_MIN,
    UMBRAL_RIESGO_CAPACIDAD,
)

CLAVE = ["estacion_id", "hora"]


def _conteo_por_hora(
    trips: pd.DataFrame, columna_estacion: str, columna_fecha: str, nombre: str
) -> pd.DataFrame:
    """Cuenta viajes por estacion y hora truncada; base de salidas y llegadas."""
    x = trips[[columna_estacion, columna_fecha]].dropna()
    horas = pd.to_datetime(x[columna_fecha]).dt.floor("h")
    return (
        x.assign(hora=horas)
        .groupby([columna_estacion, "hora"], observed=True)
        .size()
        .rename(nombre)
        .reset_index()
        .rename(columns={columna_estacion: "estacion_id"})
    )


def hourly_departures(trips: pd.DataFrame) -> pd.DataFrame:
    """Salidas por estacion-hora (``estacion_id``, ``hora``, ``salidas``).

    Es la variable objetivo de P1: cuantos viajes inician en una estacion en la
    siguiente hora. Solo devuelve las horas con al menos un viaje; para tener
    tambien las horas en cero se usa ``complete_hourly_panel``.
    """
    return _conteo_por_hora(trips, "origen_id", "inicio_viaje", "salidas")


def hourly_arrivals(trips: pd.DataFrame) -> pd.DataFrame:
    """Llegadas por estacion-hora (``estacion_id``, ``hora``, ``llegadas``).

    P2 las necesita porque el riesgo de quedarse sin bicis o sin anclajes depende
    de lo que entra y lo que sale. Se usa la hora real de fin del viaje, que es
    cuando la bici ocupa un anclaje.
    """
    return _conteo_por_hora(trips, "destino_id", "fin_viaje", "llegadas")


def calendar_features(station_hours: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``dia_semana`` (0 = lunes), ``hora_dia`` y ``fin_semana`` a partir de ``hora``.

    Son las variables que siempre se conocen antes de la hora objetivo, y en el
    EDA explican la mayor parte del patron de uso (picos de entrada y salida del
    trabajo entre semana). Sirven para P1 y P2.
    """
    out = station_hours.copy()
    hora = pd.to_datetime(out["hora"])
    out["dia_semana"] = hora.dt.dayofweek.astype("int8")
    out["hora_dia"] = hora.dt.hour.astype("int8")
    out["fin_semana"] = (out["dia_semana"] >= 5).astype("int8")
    return out


def hourly_flow_balance(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """Une salidas y llegadas por estacion-hora y calcula ``balance = llegadas - salidas``.

    Un balance negativo vacia la estacion y uno positivo la llena; es la base de la
    etiqueta proxy de P2. Las horas que solo aparecen en uno de los dos lados
    cuentan cero en el otro. Es informacion de la hora objetivo: solo para etiquetas.
    """
    flujos = departures[CLAVE + ["salidas"]].merge(
        arrivals[CLAVE + ["llegadas"]], on=CLAVE, how="outer"
    )
    flujos[["salidas", "llegadas"]] = flujos[["salidas", "llegadas"]].fillna(0).astype("int64")
    flujos["balance"] = flujos["llegadas"] - flujos["salidas"]
    return flujos


def proxy_risk_labels(
    flows: pd.DataFrame,
    capacities: pd.DataFrame,
    threshold: float = UMBRAL_RIESGO_CAPACIDAD,
) -> pd.DataFrame:
    """Etiqueta de riesgo de P2: ``|balance| / capacidad >= threshold``.

    MiBici no publica inventario por estacion, asi que el riesgo se aproxima con la
    presion que el flujo neto de una hora pone sobre la capacidad: si en una hora
    se mueve una cuarta parte de los anclajes, la estacion se acerca a quedar vacia
    o llena. Agrega ``capacidad``, ``presion`` y ``riesgo_proxy``.

    Las capacidades vienen de la nomenclatura de 2026, no de 2024, y las estaciones
    sin capacidad valida (vacia o cero) quedan fuera porque la presion no se define.
    """
    if threshold <= 0:
        raise ValueError("threshold debe ser positivo")

    caps = capacities[["estacion_id", "capacidad"]].copy()
    caps["capacidad"] = pd.to_numeric(caps["capacidad"], errors="coerce")
    caps = caps[caps["capacidad"] > 0].drop_duplicates("estacion_id")
    # Se iguala el tipo de la clave: estaciones.parquet trae Int32 y los baselines texto.
    caps["estacion_id"] = caps["estacion_id"].astype(flows["estacion_id"].dtype)

    out = flows.merge(caps, on="estacion_id", how="inner", validate="many_to_one")
    out["presion"] = out["balance"].abs() / out["capacidad"]
    out["riesgo_proxy"] = (out["presion"] >= threshold).astype("int8")
    return out


def hourly_usage_profiles(trips: pd.DataFrame, count_column: str | None = None) -> pd.DataFrame:
    """Salidas por estacion, tipo de dia (``laboral``/``fin_semana``) y hora del dia.

    Es la materia prima de P3: dos estaciones con perfiles parecidos (por ejemplo,
    pico a las 8 entre semana) cumplen un papel parecido en la ciudad aunque no
    tengan el mismo volumen. Sin ``count_column`` cuenta cada viaje; con ella suma
    conteos que ya vienen agregados.
    """
    if count_column is None:
        x = trips[["origen_id", "inicio_viaje"]].dropna().assign(conteo=1)
    else:
        x = trips[["origen_id", "inicio_viaje", count_column]].dropna()
        x = x.assign(conteo=pd.to_numeric(x[count_column], errors="raise"))
        if (x["conteo"] < 0).any():
            raise ValueError("Los conteos no pueden ser negativos")

    fecha = pd.to_datetime(x["inicio_viaje"])
    x = x.assign(
        tipo_dia=np.where(fecha.dt.dayofweek >= 5, "fin_semana", "laboral"),
        hora_dia=fecha.dt.hour,
    )
    return (
        x.groupby(["origen_id", "tipo_dia", "hora_dia"], observed=True)["conteo"]
        .sum()
        .reset_index()
        .rename(columns={"origen_id": "estacion_id", "conteo": "salidas"})
    )


def complete_hourly_panel(
    departures: pd.DataFrame, stations: pd.Index, start: str, end: str
) -> pd.DataFrame:
    """Panel estacion-hora completo en ``[start, end)``, con cero en las horas sin salidas.

    P1 se evalua por estacion-hora, y una hora sin viajes tambien es una
    observacion (demanda cero); sin ellas el error del modelo se subestima. Ojo:
    cero no prueba que la estacion estuviera operando. Agrega el calendario con
    ``calendar_features``.
    """
    horas = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(hours=1), freq="h")
    malla = pd.MultiIndex.from_product([stations, horas], names=CLAVE)
    panel = (
        departures.set_index(CLAVE)["salidas"]
        .reindex(malla, fill_value=0)
        .astype("int32")
        .reset_index()
    )
    return calendar_features(panel)


def lag_features(
    panel: pd.DataFrame, column: str = "salidas", lags: tuple[int, ...] = (1, 24, 168)
) -> pd.DataFrame:
    """Agrega ``{column}_lag_{h}h``: el valor de la misma estacion ``h`` horas antes.

    Con ``(1, 24, 168)`` se tiene la hora previa, la misma hora de ayer y la misma
    hora de la semana pasada, que es lo que un modelo candidato de P1 o P2 puede
    usar sin ver el futuro. Busca por fecha, no por posicion, asi que un hueco en
    el panel deja NaN en lugar de mezclar horas. Las primeras horas del periodo
    quedan en NaN porque no hay historia.
    """
    if any(h <= 0 for h in lags):
        raise ValueError("Los rezagos deben ser positivos para no usar el futuro")

    out = panel.copy()
    base = panel[CLAVE + [column]]
    for h in lags:
        pasado = base.assign(hora=base["hora"] + pd.Timedelta(hours=h))
        pasado = pasado.rename(columns={column: f"{column}_lag_{h}h"})
        out = out.merge(pasado, on=CLAVE, how="left")
    return out


def trip_quality_flags(trips: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``duracion_min``, ``duracion_atipica``, ``edad`` y ``edad_atipica``.

    Marca, sin borrar, los viajes fuera de los umbrales de ``config``: menos de
    ``DURACION_MIN_MINUTOS`` (falsos arranques), mas de ``DURACION_MAX_MINUTOS``
    (bicis no devueltas a tiempo) y edades fuera de ``[EDAD_MIN, EDAD_MAX]``. Cada
    pregunta decide si filtra: P1 cuenta inicios de viaje y los conserva.

    Septiembre y octubre traen la hora sin segundos, asi que ahi la duracion tiene
    resolucion de un minuto. El anio de nacimiento vacio deja la edad en NA y no
    se marca como atipico.
    """
    out = trips.copy()
    duracion = (out["fin_viaje"] - out["inicio_viaje"]).dt.total_seconds() / 60
    out["duracion_min"] = duracion
    out["duracion_atipica"] = (
        (duracion < DURACION_MIN_MINUTOS) | (duracion > DURACION_MAX_MINUTOS)
    ).astype("int8")

    edad = ANIO_DATOS - out["anio_nacimiento"].astype("Int16")
    out["edad"] = edad
    out["edad_atipica"] = ((edad < EDAD_MIN) | (edad > EDAD_MAX)).fillna(False).astype("int8")
    return out
