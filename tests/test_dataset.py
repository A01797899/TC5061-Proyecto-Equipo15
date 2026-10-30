import pandas as pd
import pytest

from mibici.config import FORMATO_FECHA_POR_MES
from mibici.dataset import (
    calcular_version,
    descargar,
    detectar_encoding,
    leer_viajes,
    normalizar_encabezados,
    parsear_fechas,
)

ENCABEZADO = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]


def test_normaliza_encabezado_con_anio_corrupto():
    corrupto = ENCABEZADO.copy()
    corrupto[3] = "A}äe_nacimiento"
    assert normalizar_encabezados(corrupto)[3] == "anio_nacimiento"
    assert normalizar_encabezados(ENCABEZADO)[0] == "viaje_id"


def test_rechaza_columnas_inesperadas():
    with pytest.raises(ValueError):
        normalizar_encabezados(ENCABEZADO[:-1] + ["Otra"])


def test_parsea_formatos_de_config():
    normal = parsear_fechas(pd.Series(["2024-01-02 03:04:05", None]), FORMATO_FECHA_POR_MES[1])
    assert normal[0] == pd.Timestamp("2024-01-02 03:04:05")
    assert pd.isna(normal[1])
    # Dia primero: 1 de septiembre, no 9 de enero.
    septiembre = parsear_fechas(pd.Series(["01/09/2024 0:07"]), FORMATO_FECHA_POR_MES[9])
    assert septiembre[0] == pd.Timestamp("2024-09-01 00:07")


def test_fecha_con_otro_formato_queda_nat():
    fechas = parsear_fechas(pd.Series(["01/09/2024 0:07"]), FORMATO_FECHA_POR_MES[1])
    assert pd.isna(fechas[0])


def test_detecta_encoding(tmp_path):
    utf8 = tmp_path / "utf8.csv"
    latin = tmp_path / "latin.csv"
    utf8.write_text("Año\n", encoding="utf-8")
    latin.write_text("Año\n", encoding="latin1")
    assert detectar_encoding(utf8) == "utf-8"
    assert detectar_encoding(latin) == "latin1"


def test_lee_viajes_formato_septiembre(tmp_path):
    ruta = tmp_path / "datos_abiertos_2024_09.csv"
    ruta.write_bytes(
        (
            ",".join(ENCABEZADO) + "\r\n" + "1,2,,1990,01/09/2024 0:00,01/09/2024 0:07,154,161\r\n"
        ).encode("latin1")
    )
    df = leer_viajes(ruta, "latin1", FORMATO_FECHA_POR_MES[9])
    assert list(df.columns)[3] == "anio_nacimiento"
    assert df.loc[0, "inicio_viaje"] == pd.Timestamp("2024-09-01 00:00")
    assert df.loc[0, "origen_id"] == 154
    assert pd.isna(df.loc[0, "genero"])


def test_version_depende_solo_de_los_hashes():
    fuentes = [{"archivo": "a.csv", "sha256": "1"}, {"archivo": "b.csv", "sha256": "2"}]
    assert calcular_version(fuentes) == calcular_version([dict(f) for f in fuentes])
    assert calcular_version(fuentes) != calcular_version(fuentes[:1])


def test_descarga_rechaza_esquema_no_permitido(tmp_path):
    destino = tmp_path / "raw" / "copia.csv"
    with pytest.raises(ValueError, match="Esquema de descarga no permitido"):
        descargar("ftp://ejemplo/datos.csv", destino)
    assert not destino.exists()


def test_descarga_no_deja_temporales(tmp_path):
    origen = tmp_path / "origen.csv"
    origen.write_text("a,b\n1,2\n")
    destino = tmp_path / "raw" / "copia.csv"
    descargar(origen.as_uri(), destino)
    assert destino.read_text() == "a,b\n1,2\n"
    assert list(destino.parent.iterdir()) == [destino]


@pytest.mark.parametrize("contenido", [None, ""])
def test_descarga_fallida_limpia_el_temporal(tmp_path, contenido):
    origen = tmp_path / "origen.csv"
    if contenido is not None:
        origen.write_text(contenido)  # Archivo vacio
    destino = tmp_path / "raw" / "copia.csv"
    with pytest.raises((OSError, ValueError)):
        descargar(origen.as_uri(), destino)
    assert list(destino.parent.iterdir()) == []
