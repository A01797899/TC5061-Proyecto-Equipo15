# Evidencias Semana 9 — comparación A/B final

Estas imágenes son copias **sin edición** de capturas compartidas durante la ejecución de la actividad. La numeración deja espacios intencionales para preservar referencias a otras capturas.

| Captura | Verificación | Imagen fuente |
|---|---|---|
| `25_challenger_inferencia_validada.png` | Verificación de carga e inferencia del Challenger | `image(20261115-153744).png` |
| `26_challenger_registro_version2.png` | Registro MLflow de Challenger versión 2 | `image(20261115-154129).png` |
| `27_champion_challenger_aliases.png` | Validación de alias champion v1 y challenger v2 | `image(20261115-154247).png` |
| `29_champion_vs_challenger_comparison.png` | Métricas globales A/B del comparador | `image(20261115-155057).png` |
| `30_ab_comparison_files_validation.png` | Archivos CSV/JSON generados por la comparación | `image(20261115-155143).png` |
| `31_ab_daily_comparison_analysis.png` | Resultados de comparación diaria | `image(20261115-155226).png` |
| `32_ab_station_comparison_analysis.png` | Resultados del análisis por estación | `image(20261115-173337).png` |
| `37_ab_drift_integrated_analysis.png` | Integración de A/B y cambios de demanda | `image(20261115-173717).png` |
| `38_ab_final_unit_tests_4_passed.png` | Cuatro pruebas nuevas exitosas | `image(20261115-175619).png` |
| `39_ab_all_tests_27_passed.png` | 27 pruebas A/B exitosas | `image(20261115-175704).png` |
| `40_full_suite_311_passed.png` | 311 pruebas, 24 advertencias | `image(20261115-175824).png` |
| `41_readme_menu_navigation.png` | Navegación de sección 20 en el README | `image(20261115-182823).png` |

## Pendiente de incorporación manual

- `28_mlflow_registry_champion_challenger.png`: captura de la **UI de MLflow** mostrando versiones 1 y 2 y sus alias. Fue revisada en la conversación, pero no está disponible como archivo PNG verificable para empaquetar. Guardar desde el equipo y colocarla aquí con ese nombre.
- Otras capturas anunciadas en la conversación (por ejemplo, 33–36 de ejecución de estaciones) no se agregan si no se puede verificar un archivo original correspondiente. Son opcionales y pueden colocarse aquí desde el equipo, sin alterar el contenido.
- Las cuatro figuras finales (`ab_final_*.png`) **ya existen en** `reports/week9/ab_testing/` en el repositorio local: conservarlas en esa ubicación y no duplicarlas como capturas.
- La evidencia histórica 01–13 del Champion se conserva en `reports/week9/evidence/ab_testing/`; no debe sobrescribirse. Las evidencias de Jorge y Oscar permanecen en sus propias carpetas.

## Resultados de referencia verificados

Champion v1 MAE 0.840679, RMSE 1.436953. Challenger v2 MAE 0.824333, RMSE 1.404644. 269 de 362 estaciones y 28 de 31 días con mejora. 311 pruebas aprobadas, 24 advertencias.

**No ejecutar** `git add .` ni mover archivos `.db`, `mlartifacts` o `champion_predictions.parquet` al commit.
