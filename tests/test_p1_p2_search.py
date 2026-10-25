"""Validaciones de búsqueda sin entrenar ni conectar al servidor real."""
from unittest.mock import MagicMock, patch
import pytest
from mibici.modeling import p1_p2_search as search

@pytest.mark.parametrize('question,count', [('p1',3), ('p2',3)])
def test_search_space(question, count):
    assert len(search.SEARCH_SPACE[question]) == count

@pytest.mark.parametrize('question,expected', [('p1','val_mae'),('p2','val_average_precision')])
def test_selection_metric(question, expected):
    assert search.SELECTION_METRIC[question] == expected

def test_invalid_question():
    with pytest.raises(ValueError):
        search.run_search('p4', 'http://127.0.0.1:5000')

@pytest.mark.parametrize('question,scores,expected_best', [
    ('p1', [0.9, 0.8, 0.85], 'r2'),
    ('p2', [0.3, 0.36, 0.34], 'r2'),
])
def test_parent_child_and_validation_selection(question, scores, expected_best):
    parent = MagicMock()
    parent.info.run_id = 'parent-id'
    context = MagicMock()
    context.__enter__.return_value = parent
    runs = {}
    criterion = search.SELECTION_METRIC[question]
    for i, score in enumerate(scores, 1):
        run = MagicMock()
        run.data.tags = {'mlflow.parentRunId': 'parent-id'}
        run.data.metrics = {criterion: score}
        runs[f'r{i}'] = run
    client = MagicMock()
    client.get_run.side_effect = lambda run_id: runs[run_id]
    with (patch.object(search.mlflow, 'start_run', return_value=context),
          patch.object(search.mlflow, 'set_tracking_uri'),
          patch.object(search.mlflow, 'set_experiment'),
          patch.object(search.mlflow, 'set_tags'),
          patch.object(search.mlflow, 'log_params'),
          patch.object(search.mlflow, 'log_metrics'),
          patch.object(search.mlflow, 'log_dict'),
          patch.object(search, 'MlflowClient', return_value=client),
          patch.object(search, 'run_candidate', side_effect=['r1','r2','r3']) as candidate):
        result = search.run_search(question, 'http://localhost:5000')
    assert result['parent_run_id'] == 'parent-id'
    assert result['children'] == 3
    assert result['best']['run_id'] == expected_best
    assert candidate.call_count == 3
    assert all(call.kwargs['nested'] is True for call in candidate.call_args_list)
