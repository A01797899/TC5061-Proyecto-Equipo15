# Diagnóstico de seguridad — Semana 7

Revisión de credenciales y configuración en la rama `feature/A01732182-week7-Security`, commit `7b56729` (PR #21). No se pegaron valores secretos: no había ninguno en el código revisado.

Los comandos de esta revisión usan el intérprete `TC5061-Proyecto-Equipo15/.venv/bin/python` (Python 3.12.13), el mismo entorno donde están instalados el proyecto, Bandit y pip-audit.

## .gitignore

Ya excluía lo que usa el proyecto:

- `.env` y `.env.*`, con la excepción `!.env.example`
- entornos virtuales (`.venv/`, `venv/`, `env/`)
- `mlruns/`, `mlartifacts/` y `mlflow.db`
- modelos serializados (`*.pkl`, `*.pickle`, `*.joblib`, y formatos similares)
- datasets bajo `data/raw`, `data/external`, `data/interim` y `data/processed`

Faltaban los nombres de claves y certificados que pide la actividad. Se añadieron `*.pem`, `*.key`, `*.p12` y `*.pfx`. El proyecto no genera esos archivos; la regla evita versionarlos si aparecen.

## Archivos versionados

```bash
git ls-files | grep -E '(^|/)(\.env(\.|$)|mlflow\.db|.*\.(pem|key|p12|pfx)$)'
```

Sin coincidencias: Git no sigue `.env`, `mlflow.db` ni certificados o claves con esos nombres.

No existía `.env` ni `.env.example`. Se agregó `.env.example` solo con `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. Un `.env` real sigue ignorado.

## Credenciales en Python

```bash
git grep -ilE 'password|secret|api[_-]?key|access[_-]?token|private[_-]?key' -- '*.py'
```

Sin archivos. No hay asignaciones de contraseñas, secretos, API keys ni tokens en el código Python.

Las únicas lecturas de entorno son la URI de MLflow:

| Archivo | Uso |
|---|---|
| `mibici/config.py` | `load_dotenv()` y `os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")` |
| `mibici/modeling/train.py` | misma variable, con el valor de `config.py` como respaldo |
| `mibici/modeling/search.py` | argumento, luego la variable de entorno, luego el valor de `config.py` |

El valor por defecto es el servidor local. No hay tokens de registro ni credenciales de bases remotas en el código.

## Conclusión

La configuración ya separa secretos del repositorio. El ajuste de esta revisión es de cobertura: ignorar claves y certificados, y dejar una plantilla de entorno sin secretos. El análisis estático (Bandit) y el de dependencias (pip-audit) están en los reportes de esta misma carpeta.
