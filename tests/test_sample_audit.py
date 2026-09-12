import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'sample_audit', Path(__file__).parents[1] / 'scripts/audit_handwriting_samples.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_excludes_both_conflicting_labels_and_preserves_source():
    records = [
        dict(id='a', expected='a', category='single_character', strokes=[[dict(x=1, y=2)]]),
        dict(id='b', expected='b', category='single_character', strokes=[[dict(x=1, y=2)]]),
        dict(id='c', expected='c', category='single_character', strokes=[[dict(x=3, y=4)]]),
    ]
    valid, report = module.audit(records)
    assert [r['id'] for r in valid] == ['c']
    assert len(records) == 3
    assert len(report['excluded']) == 2


def test_audit_keeps_only_one_copy_of_identical_labeled_ink():
    record = dict(id='a', expected='a', category='single_character', strokes=[[dict(x=1, y=2)]])
    valid, report = module.audit([record, dict(record, id='a2')])
    assert valid == [record]
    assert report['excluded'][0]['reason'] == 'duplicate_labeled_ink'
