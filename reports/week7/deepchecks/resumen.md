# Validación de datos con DeepChecks

## P1 (regression)

- Filas por split: {'test': 529968, 'train': 2119872, 'val': 529968}
- Llave estación-hora duplicada: 0
- Filas de train repetidas en características y etiqueta: 817,067
- Filas de train con nulos en rezagos: 60,816
- Muestra por split para DeepChecks: 200,000 (semilla 42)

| Suite | Check | Estado | Recomendación |
|---|---|---|---|
| integridad_train | IsSingleValue | aprobado |  |
| integridad_train | SpecialCharacters | aprobado |  |
| integridad_train | MixedNulls | aprobado |  |
| integridad_train | MixedDataTypes | aprobado |  |
| integridad_train | StringMismatch | aprobado |  |
| integridad_train | DataDuplicates | aprobado |  |
| integridad_train | StringLengthOutOfBounds | aprobado |  |
| integridad_train | FeatureLabelCorrelation | aprobado |  |
| integridad_train | IdentifierLabelCorrelation | aprobado |  |
| integridad_train | FeatureFeatureCorrelation | aprobado |  |
| integridad_train | PercentOfNulls | aprobado |  |
| integridad_test | IsSingleValue | aprobado |  |
| integridad_test | SpecialCharacters | aprobado |  |
| integridad_test | MixedNulls | aprobado |  |
| integridad_test | MixedDataTypes | aprobado |  |
| integridad_test | StringMismatch | aprobado |  |
| integridad_test | DataDuplicates | aprobado |  |
| integridad_test | StringLengthOutOfBounds | aprobado |  |
| integridad_test | FeatureLabelCorrelation | aprobado |  |
| integridad_test | IdentifierLabelCorrelation | aprobado |  |
| integridad_test | FeatureFeatureCorrelation | aprobado |  |
| integridad_test | PercentOfNulls | aprobado |  |
| train_test | DatasetsSizeComparison | aprobado |  |
| train_test | NewCategoryTrainTest | aprobado |  |
| train_test | StringMismatchComparison | aprobado |  |
| train_test | DateTrainTestLeakageDuplicates | aprobado |  |
| train_test | DateTrainTestLeakageOverlap | aprobado |  |
| train_test | FeatureLabelCorrelationChange | aprobado |  |
| train_test | FeatureDrift | aprobado |  |
| train_test | LabelDrift | aprobado |  |

## P2 (binary)

- Filas por split: {'test': 529968, 'train': 2119872, 'val': 529968}
- Llave estación-hora duplicada: 0
- Filas de train repetidas en características y etiqueta: 657,852
- Filas de train con nulos en rezagos: 60,816
- Muestra por split para DeepChecks: 200,000 (semilla 42)

| Suite | Check | Estado | Recomendación |
|---|---|---|---|
| integridad_train | ConflictingLabels | problema | Revisar filas con las mismas características y etiquetas distintas. |
| integridad_train | ClassImbalance | advertencia | Mantener métricas robustas al desbalance (F1, AP) y el umbral elegido en validación. |
| integridad_train | IsSingleValue | aprobado |  |
| integridad_train | SpecialCharacters | aprobado |  |
| integridad_train | MixedNulls | aprobado |  |
| integridad_train | MixedDataTypes | aprobado |  |
| integridad_train | StringMismatch | aprobado |  |
| integridad_train | DataDuplicates | aprobado |  |
| integridad_train | StringLengthOutOfBounds | aprobado |  |
| integridad_train | FeatureLabelCorrelation | aprobado |  |
| integridad_train | IdentifierLabelCorrelation | aprobado |  |
| integridad_train | FeatureFeatureCorrelation | aprobado |  |
| integridad_train | PercentOfNulls | aprobado |  |
| integridad_test | ConflictingLabels | problema | Revisar filas con las mismas características y etiquetas distintas. |
| integridad_test | ClassImbalance | advertencia | Mantener métricas robustas al desbalance (F1, AP) y el umbral elegido en validación. |
| integridad_test | IsSingleValue | aprobado |  |
| integridad_test | SpecialCharacters | aprobado |  |
| integridad_test | MixedNulls | aprobado |  |
| integridad_test | MixedDataTypes | aprobado |  |
| integridad_test | StringMismatch | aprobado |  |
| integridad_test | DataDuplicates | aprobado |  |
| integridad_test | StringLengthOutOfBounds | aprobado |  |
| integridad_test | FeatureLabelCorrelation | aprobado |  |
| integridad_test | IdentifierLabelCorrelation | aprobado |  |
| integridad_test | FeatureFeatureCorrelation | aprobado |  |
| integridad_test | PercentOfNulls | aprobado |  |
| train_test | DatasetsSizeComparison | aprobado |  |
| train_test | NewLabelTrainTest | aprobado |  |
| train_test | NewCategoryTrainTest | aprobado |  |
| train_test | StringMismatchComparison | aprobado |  |
| train_test | DateTrainTestLeakageDuplicates | aprobado |  |
| train_test | DateTrainTestLeakageOverlap | aprobado |  |
| train_test | FeatureLabelCorrelationChange | aprobado |  |
| train_test | FeatureDrift | aprobado |  |
| train_test | LabelDrift | aprobado |  |

## Checks excluidos

- `OutlierSampleDetection`: llama quantile(interpolation=...), argumento que NumPy 2 eliminó.
- `TrainTestSamplesMix`: usa DataFrame.applymap, que pandas 3 eliminó.
- `MultivariateDrift`: su modelo interno no implementa __sklearn_tags__ (scikit-learn 1.6+).
- `IndexTrainTestLeakage`: la misma estación aparece en train y test por diseño; la fuga se mide por fecha.

## Condiciones reemplazadas

- `FeatureFeatureCorrelation`: la condición original usa stack(), que en pandas 3 ya no descarta NaN y marca todos los pares como correlacionados; la evalúo con stack().dropna().
- `ClassImbalance`: la condición original avisa cuando las clases están balanceadas; la invierto para avisar cuando la clase minoritaria es menor al 10 % de la mayoritaria.

## Notas de interpretación

- `DataDuplicates` compara el registro completo, incluida la fecha; el conteo exacto compara solo características y etiqueta. Filas distintas con los mismos valores, como horas sin viajes, no son registros repetidos: la llave estación-hora no se duplica.
- Los nulos están solo en los rezagos y al inicio de train, donde todavía no hay historia; el pipeline de candidatos los descarta con `dropna`.
