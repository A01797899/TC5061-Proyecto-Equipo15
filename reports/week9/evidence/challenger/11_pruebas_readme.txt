Actividad 3.9. Pruebas e instrucciones de reproducción
Fecha: 2026-11-15
Rama: feature/week9-challenger-JLMM

python -m pytest tests/test_week9_ab_testing.py -v -k challenger
3 passed, 20 deselected in 0.98s
  test_comparacion_challenger_mejor PASSED
  test_challenger_rechaza_ruta_de_evaluacion PASSED
  test_challenger_hiperparametros_del_champion PASSED

ruff check scripts/week9/train_challenger.py tests/test_week9_ab_testing.py
All checks passed.

README sección 19 cita los tres comandos:
  python -m scripts.week9.train_challenger
  python -m mibici.modeling.registry
  client.get_model_version_by_alias

Validacion: pytest en verde, Ruff limpio y la sección 19 documenta el entrenamiento, el registro y la comprobación del alias champion.
