import hashlib
import json

from loguru import logger
import pandas as pd
import pytest

from mibici import dataset
from mibici.config import FORMATO_FECHA_POR_MES, MANIFIESTO_DATOS
from mibici.dataset import (
    TIPOS_ESTACIONES,
    TIPOS_VIAJES,
    advertir_cambios,
    calcular_version,
    cargar_manifiesto,
    descargar,
    detectar_encoding,
    leer_estaciones,
    leer_viajes,
    normalizar_encabezados,
    parsear_fechas,
    ruta_relativa,
    sha256_archivo,
    version_datos,
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

@pytest.fixture
def avisos():
    """Mensajes de advertencia emitidos con loguru durante la prueba."""
    mensajes = []
    sink = logger.add(lambda m: mensajes.append(m.record["message"]), level="WARNING")
    yield mensajes
    logger.remove(sink)


def escribir_csv(ruta, filas, encabezado=ENCABEZADO, encoding="latin1"):
    texto = ",".join(encabezado) + "\n" + "".join(",".join(f) + "\n" for f in filas)
    ruta.write_bytes(texto.encode(encoding))
    return ruta


def test_normaliza_encabezados_con_comillas_y_espacios():
    sucios = [f' "{c}" ' for c in ENCABEZADO]
    assert normalizar_encabezados(sucios) == normalizar_encabezados(ENCABEZADO)


def test_rechaza_encabezado_vacio():
    with pytest.raises(ValueError):
        normalizar_encabezados([])


def test_fecha_imposible_y_serie_vacia():
    fechas = parsear_fechas(pd.Series(["2024-02-30 10:00:00"]), FORMATO_FECHA_POR_MES[2])
    assert pd.isna(fechas[0])
    assert parsear_fechas(pd.Series([], dtype=str), FORMATO_FECHA_POR_MES[1]).empty


def test_detecta_latin1_con_bytes_invalidos_en_utf8(tmp_path):
    ruta = tmp_path / "x.csv"
    ruta.write_bytes(b"Ni\xf1o\n")
    assert detectar_encoding(ruta) == "latin1"


def test_archivo_vacio_se_lee_como_utf8(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_bytes(b"")
    assert detectar_encoding(ruta) == "utf-8"


def test_sha256_no_depende_del_tamano_de_bloque(tmp_path):
    ruta = tmp_path / "datos.bin"
    contenido = bytes(range(256)) * 10
    ruta.write_bytes(contenido)
    esperado = hashlib.sha256(contenido).hexdigest()
    assert sha256_archivo(ruta) == esperado
    assert sha256_archivo(ruta, bloque=7) == esperado


def test_sha256_de_archivo_vacio(tmp_path):
    ruta = tmp_path / "vacio.bin"
    ruta.write_bytes(b"")
    assert sha256_archivo(ruta) == hashlib.sha256(b"").hexdigest()


def test_lee_viajes_tipos_y_valores_invalidos(tmp_path):
    ruta = escribir_csv(
        tmp_path / "enero.csv",
        [
            ["1", "10", "M", "1990", "2024-01-01 08:00:00", "2024-01-01 08:10:00", "5", "6"],
            ["2", "abc", "F", "", "fecha mala", "2024-01-01 09:10:00", "", "6"],
        ],
    )
    df = leer_viajes(ruta, "latin1", FORMATO_FECHA_POR_MES[1])
    for columna, tipo in TIPOS_VIAJES.items():
        assert str(df[columna].dtype) == tipo
    assert df["genero"].dtype == "category"
    # Texto no numerico y vacios quedan como NA; la fecha mala como NaT.
    assert pd.isna(df.loc[1, "usuario_id"])
    assert pd.isna(df.loc[1, "anio_nacimiento"])
    assert pd.isna(df.loc[1, "origen_id"])
    assert pd.isna(df.loc[1, "inicio_viaje"])


def test_lee_viajes_sin_filas(tmp_path):
    ruta = escribir_csv(tmp_path / "vacio.csv", [])
    df = leer_viajes(ruta, "latin1", FORMATO_FECHA_POR_MES[1])
    assert df.empty
    assert list(df.columns) == list(dataset.RENOMBRAR_COLUMNAS.values())


def test_lee_viajes_rechaza_columnas_faltantes(tmp_path):
    ruta = escribir_csv(tmp_path / "corto.csv", [["1"] * 7], encabezado=ENCABEZADO[:-1])
    with pytest.raises(ValueError):
        leer_viajes(ruta, "latin1", FORMATO_FECHA_POR_MES[1])


def test_lee_estaciones_normaliza_nombres_y_tipos(tmp_path):
    ruta = tmp_path / "nomenclatura.csv"
    ruta.write_bytes(
        (
            '"id","name","latitude","longitude","dpcapacity","extra"\n'
            '"2","(GDL-001) Niño","20.66","-103.34","15","x"\n'
            '"3","(GDL-002) Sin capacidad","20.67","no aplica","","y"\n'
        ).encode("latin1")
    )
    est = leer_estaciones(ruta)
    assert list(est.columns) == list(dataset.RENOMBRAR_ESTACIONES.values())
    for columna, tipo in TIPOS_ESTACIONES.items():
        assert str(est[columna].dtype) == tipo
    assert est.loc[0, "nombre"] == "(GDL-001) Niño"
    assert est.loc[0, "capacidad"] == 15
    assert pd.isna(est.loc[1, "capacidad"])
    assert pd.isna(est.loc[1, "longitud"])


def test_lee_estaciones_rechaza_columna_faltante(tmp_path):
    ruta = tmp_path / "nomenclatura.csv"
    ruta.write_text('"id","name","latitude","longitude"\n"2","A","20.6","-103.3"\n')
    with pytest.raises(KeyError):
        leer_estaciones(ruta)


def test_ruta_relativa_dentro_y_fuera_del_proyecto(tmp_path):
    assert ruta_relativa(MANIFIESTO_DATOS) == "data/manifiesto.json"
    with pytest.raises(ValueError):
        ruta_relativa(tmp_path / "otro.csv")


def test_version_cambia_con_un_hash_y_tiene_12_caracteres():
    fuentes = [{"archivo": "a.csv", "sha256": "1"}]
    version = calcular_version(fuentes)
    assert len(version) == 12
    assert int(version, 16) >= 0
    assert version != calcular_version([{"archivo": "a.csv", "sha256": "2"}])


def test_manifiesto_versionado_es_consistente():
    manifiesto = cargar_manifiesto()
    assert version_datos() == manifiesto["version_datos"]
    assert calcular_version(manifiesto["fuentes"]) == manifiesto["version_datos"]
    assert len(manifiesto["fuentes"]) == 13


def test_version_datos_de_un_manifiesto_propio(tmp_path):
    ruta = tmp_path / "manifiesto.json"
    ruta.write_text(json.dumps({"version_datos": "abc123", "fuentes": []}), encoding="utf-8")
    assert version_datos(ruta) == "abc123"


def test_advierte_solo_archivos_que_cambiaron(tmp_path, avisos):
    ruta = tmp_path / "manifiesto.json"
    ruta.write_text(
        json.dumps(
            {"fuentes": [{"archivo": "a.csv", "sha256": "1"}, {"archivo": "b.csv", "sha256": "2"}]}
        ),
        encoding="utf-8",
    )
    nuevas = [
        {"archivo": "a.csv", "sha256": "1"},
        {"archivo": "b.csv", "sha256": "9"},
        {"archivo": "c.csv", "sha256": "3"},
    ]
    advertir_cambios(nuevas, ruta)
    assert len(avisos) == 1
    assert "b.csv" in avisos[0]


def test_sin_manifiesto_no_advierte(tmp_path, avisos):
    advertir_cambios([{"archivo": "a.csv", "sha256": "1"}], tmp_path / "no_existe.json")
    assert avisos == []


def test_descarga_existente_no_se_sobrescribe_sin_forzar(tmp_path):
    origen = tmp_path / "origen.csv"
    origen.write_text("nuevo\n")
    destino = tmp_path / "copia.csv"
    destino.write_text("anterior\n")
    descargar(origen.as_uri(), destino)
    assert destino.read_text() == "anterior\n"
    descargar(origen.as_uri(), destino, forzar=True)
    assert destino.read_text() == "nuevo\n"


@pytest.fixture
def ingesta(tmp_path, monkeypatch):
    """Redirige rutas y URLs de ``dataset`` a CSV locales en ``tmp_path``."""
    fuentes = tmp_path / "fuentes"
    fuentes.mkdir()
    enero = escribir_csv(
        fuentes / "datos_abiertos_2024_01.csv",
        [
            ["3", "1", "M", "1990", "2024-01-02 09:00:00", "2024-01-02 09:10:00", "5", "6"],
            ["1", "1", "F", "1985", "2024-01-01 08:00:00", "2024-01-01 08:10:00", "5", "6"],
            # Repetido: se descarta por viaje_id.
            ["1", "1", "F", "1985", "2024-01-01 08:00:00", "2024-01-01 08:10:00", "5", "6"],
            # Sin origen: se descarta.
            ["4", "2", "M", "1970", "2024-01-03 10:00:00", "2024-01-03 10:20:00", "", "6"],
        ],
    )
    septiembre = escribir_csv(
        fuentes / "datos_abiertos_2024_09.csv",
        [["7", "3", "", "", "01/09/2024 0:00", "01/09/2024 0:07", "6", "5"]],
        encoding="utf-8",
    )
    estaciones = fuentes / "nomenclatura.csv"
    estaciones.write_text(
        '"id","name","latitude","longitude","dpcapacity"\n'
        '"5","A","20.6","-103.3","15"\n"6","B","20.7","-103.4","20"\n'
    )
    salida = tmp_path / "data"
    monkeypatch.setattr(dataset, "PROJ_ROOT", tmp_path)
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", salida / "raw")
    monkeypatch.setattr(dataset, "URLS_VIAJES", {1: enero.as_uri(), 9: septiembre.as_uri()})
    monkeypatch.setattr(dataset, "URL_NOMENCLATURA", estaciones.as_uri())
    monkeypatch.setattr(dataset, "VIAJES_PARQUET", salida / "processed" / "viajes.parquet")
    monkeypatch.setattr(dataset, "ESTACIONES_PARQUET", salida / "processed" / "estaciones.parquet")
    monkeypatch.setattr(dataset, "MANIFIESTO_DATOS", salida / "manifiesto.json")
    # advertir_cambios usa el manifiesto real como valor por defecto.
    monkeypatch.setattr(dataset, "advertir_cambios", lambda fuentes: None)
    return salida


def test_ingesta_completa_con_csv_locales(ingesta):
    dataset.main(forzar_descarga=False)

    viajes = pd.read_parquet(ingesta / "processed" / "viajes.parquet")
    assert viajes["viaje_id"].tolist() == [1, 3, 7]
    assert viajes["inicio_viaje"].is_monotonic_increasing
    assert pd.read_parquet(ingesta / "processed" / "estaciones.parquet")["capacidad"].tolist() == [
        15,
        20,
    ]

    manifiesto = json.loads((ingesta / "manifiesto.json").read_text(encoding="utf-8"))
    assert manifiesto["descartados"] == {"sin_inicio_u_origen": 1, "viaje_id_duplicado": 1}
    assert [s["registros"] for s in manifiesto["salidas"]] == [3, 2]
    assert manifiesto["fuentes"][0]["archivo"] == "data/raw/datos_abiertos_2024_01.csv"
    assert manifiesto["fuentes"][1]["encoding"] == "utf-8"
    assert manifiesto["version_datos"] == calcular_version(manifiesto["fuentes"])


def test_ingesta_repetida_da_la_misma_version(ingesta):
    dataset.main(forzar_descarga=False)
    primera = json.loads((ingesta / "manifiesto.json").read_text(encoding="utf-8"))
    dataset.main(forzar_descarga=True)
    segunda = json.loads((ingesta / "manifiesto.json").read_text(encoding="utf-8"))
    assert primera == segunda


def test_ingesta_falla_si_el_formato_de_fecha_no_coincide(ingesta, monkeypatch):
    # Enero leido con el formato de septiembre: todas las fechas quedan NaT.
    monkeypatch.setattr(
        dataset, "FORMATO_FECHA_POR_MES", {1: "%d/%m/%Y %H:%M", 9: "%d/%m/%Y %H:%M"}
    )
    with pytest.raises(ValueError, match="fechas no coinciden"):
        dataset.main(forzar_descarga=False)
