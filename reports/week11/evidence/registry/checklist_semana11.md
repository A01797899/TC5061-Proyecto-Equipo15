# Checklist final de José Luis - Registry y Model Card

Copia del checklist de la sección 24 de `ActividadesJoseLuis_Semana_11.pdf`, marcada con lo ejecutado el 26 de noviembre de 2026 sobre la copia del Registry servida en `http://127.0.0.1:5052`. Las referencias son archivos de `reports/week11/evidence/registry/`.

## GitHub y entorno (7/7)

- [x] Clonar el repositorio (`~/Projects/TC5061-Proyecto-Equipo15-Week11-JoseLuis`, commit `1230cff4`)
- [x] Crear rama con matrícula A01732182 (`feature/week11-registry-model-card-A01732182`, publicada en origin)
- [x] Crear directorios de documentación y evidencias
- [x] Preparar entorno Python y dependencias (`00_environment_packages.txt`, `00_pip_check.txt`)
- [x] Obtener una copia consistente del Registry histórico (`mlflow-runtime-week10.tar.gz` → `../mibici-week11-joseluis-registry/`)
- [x] Validar integridad SQLite y disponibilidad de artefactos (`integrity_check = ok`, 162 archivos; los artefactos de modelo de P2 y P3 no están en la copia, ver hallazgos)
- [x] Iniciar MLflow y validar conexión (`00_registry_connection.txt`)

## Model Registry (8/8)

- [x] Verificar modelos P1, P2 y P3 (`01_registered_models.txt`)
- [x] Verificar Champion v1 (`02b_alias_validation.txt`)
- [x] Verificar Challenger v2 (`02b_alias_validation.txt`)
- [x] Comprobar estados READY (4 versiones `READY`)
- [x] Guardar snapshot inicial de alias (`02_registry_snapshot_before.json`)
- [x] Validar Run IDs y Logged Model IDs (`03_model_lineage.json`)
- [x] Documentar linaje de ambas versiones (`03_model_lineage.txt`; commit del Challenger `ddedf4be…`)
- [x] Verificar parámetros y métricas (`05_model_parameters_metrics.txt`)

## Contrato e inferencia (7/8)

- [x] Validar firma de los tres modelos (`04c_signature_comparison.txt`: P1 confirmada con 8 entradas; P2 y P3 no confirmables porque la copia no tiene sus artefactos de modelo)
- [x] Verificar contrato de ocho variables de P1 (`04_registry_inspection.txt`)
- [x] Comparar firmas Champion y Challenger (idénticas)
- [x] Cargar los tres modelos Champion (`06a_models_loaded.txt`: P1 carga; P2 y P3 fallan por falta de `MLmodel`, documentado)
- [x] Validar predicción de referencia del Champion real (`06_pyfunc_inference.txt`: `3.694163560787005`)
- [x] Ejecutar inferencia del Challenger (`06c_champion_challenger_inference.txt`: `3.962376668317585`)
- [x] Ejecutar pruebas del Registry y ModelService (`06_registry_tests.txt`: 43 aprobadas, exit 0)
- [ ] Coordinar verificación del modelo desplegado con José Manuel (Compose no estaba en `:8000`; `06d_api_check.txt` registra el intento y la consulta complementaria al contenedor local de la semana 8)

## Model Card y gobernanza (12/12)

- [x] Documentar propósito y alcance del modelo (`docs/week11/model_card.md`, secciones 1, 3 y 4)
- [x] Documentar datos e hiperparámetros (secciones 5 y 7)
- [x] Documentar linaje y versionamiento (sección 6)
- [x] Documentar desempeño histórico (sección 9)
- [x] Incorporar resultados A/B (sección 10)
- [x] Incorporar resultados Evidently (sección 11)
- [x] Incorporar resultados Locust y FinOps (sección 17; `07_ab_drift_finops_checks.txt`)
- [x] Justificar conservación del Champion (sección 12)
- [x] Documentar riesgos y limitaciones (sección 13)
- [x] Documentar uso responsable (sección 14)
- [x] Proponer criterios de promoción y rollback (sección 18)
- [x] Validar enlaces del Model Card (37 enlaces relativos, 0 faltantes)

## Documentación y entrega (4/10)

- [x] Completar `model_registry_validation.md`
- [x] Completar `model_card.md`
- [ ] Generar las seis evidencias iniciales (capturas `01` a `06`)
- [x] Verificar conservación de los alias (`08_registry_snapshot_after.json`, `08_registry_unchanged.txt`)
- [ ] Realizar commit y push
- [ ] Capturar Model Card renderizado en GitHub (`../model_card/07_model_card.png`)
- [ ] Subir evidencia final del Model Card
- [ ] Crear Pull Request
- [ ] Preparar láminas 3 y 4
- [ ] Ensayar speech de 65 segundos
- [x] Copiar checklist (este archivo)

Total: 38/45. Los siete puntos abiertos dependen de las capturas, de la publicación en GitHub, de la coordinación con José Manuel y de la presentación.
