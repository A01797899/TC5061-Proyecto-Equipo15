# Semana 10 - Costo por inferencia y propuestas FinOps

Responsable: José Luis Mendoza Mora (A01732182)

## 1. Objetivo

Estimar el costo de una inferencia exitosa de `POST /predict` con el Champion P1 y señalar dónde puede bajar ese costo. Las mediciones de carga y de recursos son las de Oscar en [`load_testing_analysis.md`](load_testing_analysis.md). El pipeline que se menciona aparte es el de [`cicd.md`](cicd.md). Este análisis no vuelve a correr Locust ni cambia Compose, el alias `champion` ni el workflow.

## 2. Fórmula y tarifa de referencia

En régimen estable:

```text
costo_por_inferencia = precio_hora / (req_s_exitosos * 3600)
```

El denominador cuenta solo inferencias que Locust marcó como exitosas: HTTP 200, modelo correcto y predicción finita y no negativa. En todas las filas de carga de `reports/week10/finops/cost_by_scenario.csv` el indicador de error es 0, así que el req/s observado es el req/s exitoso. Con req/s en cero el cociente no se calcula: la instancia sigue costando su tarifa horaria.

La referencia es una instancia Linux on-demand de 4 vCPU, la misma cantidad de CPU de la máquina virtual de las pruebas (4 CPU y 3.8 GiB útiles).

| Campo | Valor |
|---|---|
| Proveedor | Amazon Web Services, Amazon EC2 |
| SKU | `c7g.xlarge` (Graviton3, 4 vCPU, 8 GiB, EBS only) |
| Sistema | Linux, tenencia compartida, sin licencia incluida |
| Región | US East (N. Virginia), `us-east-1` |
| Tarifa | 0.145 USD por hora |
| Código de tarifa | `R367PJ3B3KCXQF5R.JRTCKXETXF.6YS6EN2CT7` |
| Lista | Publicación `2026-11-21T00:48:16Z` del feed on-demand de EC2 |
| Consulta | 22 de noviembre de 2026 |

Fuente: [feed público de precios on-demand de EC2 para Linux en US East (N. Virginia)](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json). La página general de tarifas está en [Amazon EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

La tarifa entra al cálculo como argumento. No está escrita dentro de `scripts/week10/estimate_inference_cost.py`. El Mac donde corrió Locust no es esta instancia: la VM tenía unos 4 GiB y el `c7g.xlarge` publica 8 GiB. El precio compara escenarios entre sí. No es la factura del equipo.

## 3. Resultados

La tabla sale de `reports/week10/finops/cost_by_scenario.csv`, generado con `--hourly-usd 0.145`. S1–S4 vienen de `load_summary.csv` y `resource_summary.csv`. Cada escalón de S5 viene de `load_capacity_steps.csv` y `resource_capacity_steps.csv`. El agregado de S5 (100.33 req/s durante los seis minutos) mezcla escalones distintos, así que no es una fila de costo. Los USD por millón de inferencias son `costo_por_mil_usd * 1000`, redondeados a 6 decimales.

| Escenario | Usuarios | req/s | P95 (ms) | Inferencias por hora | USD por millón | API CPU (%) | API MiB | MLflow CPU (%) | MLflow MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | — | 0 | — | 3.45 | 303.9 | 5.44 | 2166.51 |
| S1 | 5 | 3.23 | 50 | 11,628 | 12.469900 | 14.68 | 304.51 | 2.9 | 2166.31 |
| S2 | 20 | 12.57 | 48 | 45,252 | 3.204278 | 38.8 | 304.99 | 2.87 | 2167.86 |
| S3 | 50 | 31.54 | 36 | 113,544 | 1.277038 | 71.01 | 305.79 | 1.77 | 2168.18 |
| S4 | 100 | 63.21 | 44 | 227,556 | 0.637206 | 120.76 | 307.56 | 1.77 | 2169.0 |
| S5-50 | 50 | 32.6 | 36.0 | 117,360 | 1.235515 | 77.93 | 306.61 | 4.6 | 2176.0 |
| S5-100 | 100 | 65.2 | 48.0 | 234,720 | 0.617757 | 131.35 | 307.75 | 0.77 | 2176.0 |
| S5-150 | 150 | 96.9 | 76.0 | 348,840 | 0.415663 | 173.78 | 308.75 | 4.51 | 2176.0 |
| S5-200 | 200 | 126.7 | 170.0 | 456,120 | 0.317899 | 208.38 | 312.47 | 1.46 | 2176.61 |
| S5-250 | 250 | 141.8 | 485.0 | 510,480 | 0.284046 | 228.15 | 315.12 | 2.62 | 2176.0 |
| S5-300 | 300 | 147.1 | 740.0 | 529,560 | 0.273812 | 235.5 | 316.54 | 2.15 | 2176.58 |

El punto de operación es **S5-150**: 150 usuarios, 96.9 req/s, P95 76.0 ms y 0 errores. En una hora esa tasa produce 348,840 inferencias exitosas. El cociente exacto guardado en el CSV es:

```text
0.145 / (96.9 * 3600) = 0.0000004156633413599357871803692237 USD por inferencia
```

Eso equivale a 0.415663 USD por millón de inferencias exitosas. En S1, con 3.23 req/s, el mismo equipo cuesta 12.469900 USD por millón: la tarifa horaria se reparte entre menos predicciones.

El menor costo unitario de la tabla está en S5-300: 147.1 req/s y 0.273812 USD por millón. Ahí el P95 es 740.0 ms. El punto de operación se queda en S5-150, donde el P95 es 76.0 ms. A partir de S5-200 el P95 pasa a 170.0 ms y el costo unitario ya bajó la mayor parte del camino entre S1 y el techo.

En reposo la fila Base sigue en 0.145 USD por hora, con 0 inferencias. La API ocupa 303.9 MiB y MLflow 2166.51 MiB.

## 4. Qué consume la instancia

En S5-150 la API usa 173.78 % de CPU (1.74 núcleos de un máximo de 4) y 308.75 MiB. MLflow usa 4.51 % de CPU y 2176.0 MiB. La suma de memoria es 308.75 + 2176.0 = 2484.75 MiB, y MLflow es 2176.0 / 2484.75 = 87.6 % de esa suma. La memoria de la API casi no cambia desde el reposo (303.9 MiB) hasta 300 usuarios (316.54 MiB). MLflow se mantiene junto a 2.17 GiB en todos los escenarios, porque la API carga el Champion al arrancar y después predice en memoria.

## 5. CI/CD, fuera del costo por predicción

La corrida manual #9 de `MiBici-pipeline`, modo `integration` sobre `main`, duró 2 minutos y 22 segundos en `ubuntu-24.04` y aprobó 11 de 11 pruebas con el Champion sintético (predicción de referencia 2.785). El Champion real de las pruebas de carga responde 3.694163560787005. El repositorio es público. Esos minutos de GitHub Actions comprueban el contrato del pipeline y quedan fuera del denominador de inferencias del Champion real. El detalle está en [`cicd.md`](cicd.md).

## 6. Propuestas

Cada propuesta usa las mediciones de arriba. Las que cambian el proceso o la arquitectura quedan pendientes de repetir S5.

1. Operar cerca de 96.9 req/s (150 usuarios, P95 76.0 ms). Entre S5-150 y S5-300 el costo por millón pasa de 0.415663 a 0.273812 USD, y el P95 pasa de 76.0 ms a 740.0 ms. La operación recomendada conserva la latencia de S5-150.
2. Tratar a MLflow como costo fijo. En S5-150 concentra el 87.6 % de la memoria de los dos contenedores y 4.51 % de CPU. Separarlo del servicio de inferencia es la palanca de tamaño de instancia. En el mismo feed, el `c7g.large` (2 vCPU, 4 GiB, Linux, US East N. Virginia) cuesta 0.0725 USD por hora y el `c7g.medium` (1 vCPU, 2 GiB) cuesta 0.0363 USD por hora. La API en S5-150 ya usa 1.74 núcleos, así que un `c7g.medium` no cubre esa CPU. Un `c7g.large` deja poco margen sobre 1.74 núcleos. Si una repetición de S5 sostuviera 96.9 req/s en el `c7g.large`, la tarifa horaria pasaría de 0.145 a 0.0725 USD y el costo por inferencia se reduciría a la mitad. Esa repetición no existe: 0.0725 USD por hora es el techo publicado de la diferencia, no un ahorro medido.
3. Probar `OMP_NUM_THREADS=1` en el servicio `api`, como hipótesis de CPU por solicitud en [`load_testing_analysis.md`](load_testing_analysis.md). Hace falta otra corrida de S5 antes de asignarle un costo.
4. Probar Uvicorn con `--workers 2` para usar núcleos libres. Cada proceso carga otra copia del modelo, del orden de 300 MiB adicionales, y en S5-150 la VM todavía tenía memoria libre. También queda sin un costo en dólares hasta medir el nuevo req/s y el P95.
5. Alertar o escalar con el P95. En esta campaña el error se mantuvo en 0 cuando el P95 ya había llegado a 740.0 ms.

## 7. Límites

- La carga corrió una vez, durante pocos minutos, en una máquina virtual local sobre macOS. Los valores absolutos de CPU y memoria describen esa VM.
- `docker stats` aporta instantáneas. El costo usa el req/s exitoso y la tarifa horaria, no el tiempo de CPU acumulado.
- La tarifa 0.145 USD por hora es la lista pública del `c7g.xlarge` el 22 de noviembre de 2026. No incluye almacenamiento EBS, red, IPv4, soporte ni impuestos. Otra región u otra fecha cambian el cociente; la fórmula se vuelve a evaluar con otro `--hourly-usd`.
- El Champion sintético del CI no entra en la tabla de costo.

## 8. Reproducción

Desde la raíz del repositorio, con el entorno de Python 3.12 del proyecto:

```bash
python -m scripts.week10.estimate_inference_cost --hourly-usd 0.145
python -m pytest tests/test_week10_finops.py -v
```

El CSV queda en `reports/week10/finops/cost_by_scenario.csv`. La prueba cubre la fórmula, el req/s en cero y la elección de 150 usuarios con P95 menor a 80 ms. No descarga la tarifa ni vuelve a ejecutar Locust.
