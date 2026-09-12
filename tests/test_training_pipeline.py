import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from assistive_writing_pad.recognition.adapters import install_adapter, load_adapter


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_style_splits_and_evaluation_text_exclusion():
    prepare = script('prepare_penpal_training')
    rows = [dict(row_idx=i, row=dict(author=author, text=text, strokes=[[
        dict(points=[dict(x=0, y=0), dict(x=10, y=20)]),
    ]])) for i, (author, text) in enumerate([(0, 'Train'), (9, 'Validate'), (11, 'Test'),
                                           (1, 'existing example'), (12, 'Train!')])]
    result = prepare.build_records(rows, {prepare.text_key('existing example')})
    assert {key: len(records) for key, records in result.items()} == dict(train=1, validation=1, test=1)
    script('train_trocr_adapter').validate_splits(result)
    result['test'][0]['style_id'] = 0
    with pytest.raises(ValueError, match='leakage'):
        script('train_trocr_adapter').validate_splits(result)


def test_adapter_trains_only_small_weights_and_round_trips(tmp_path):
    torch = pytest.importorskip('torch')
    safetensors = pytest.importorskip('safetensors.torch')
    def make_model():
        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.attn = torch.nn.Module()
        model.decoder.attn.q_proj = torch.nn.Linear(4, 4)
        model.decoder.attn.v_proj = torch.nn.Linear(4, 4)
        return model
    torch.manual_seed(3)
    model = make_model()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    values = torch.randn(2, 4)
    expected = model.decoder.attn.q_proj(values).detach()
    params = install_adapter(model, 2, 4)
    torch.testing.assert_close(model.decoder.attn.q_proj(values), expected)
    assert all(('lora_' in name) == p.requires_grad for name, p in model.named_parameters())
    optimizer = torch.optim.SGD(params.values(), lr=.1)
    model.decoder.attn.q_proj(values).square().sum().backward()
    optimizer.step()
    trained = model.decoder.attn.q_proj(values).detach()
    assert not torch.equal(expected, trained)
    safetensors.save_file({name: p.detach() for name, p in params.items()}, str(tmp_path/'adapter.safetensors'))
    metadata = dict(base_model='test', rank=2, alpha=4,
                    sha256=hashlib.sha256((tmp_path/'adapter.safetensors').read_bytes()).hexdigest())
    (tmp_path/'adapter.json').write_text(json.dumps(metadata))
    restored = make_model()
    restored.load_state_dict(original)
    load_adapter(restored, tmp_path, 'test')
    torch.testing.assert_close(restored.decoder.attn.q_proj(values), trained)
    with pytest.raises(ValueError, match='base checkpoint'):
        load_adapter(make_model(), tmp_path, 'other-model')
    metadata['sha256'] = 'invalid'
    (tmp_path/'adapter.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='checksum'):
        load_adapter(make_model(), tmp_path, 'test')
