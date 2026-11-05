"""Pinned encoder identities; the Qwen3.5 amendment preserves earlier evidence."""
import json
from pathlib import Path

from data import ROOT
from improvement_common import EXP


def encoder_specs():
    specs = {
        'bert': {'repository': 'google-bert/bert-base-uncased',
                 'revision': '86b5e0934494bd15c9632b12f734a8a67f723594',
                 'local_dir': 'models/pretrained/bert', 'layers': list(range(13))},
        'qwen': {'repository': 'Qwen/Qwen2.5-1.5B',
                 'revision': '8faed761d45a263340a0528343f099c05c9a4323',
                 'local_dir': 'models/pretrained/qwen2.5-1.5b', 'layers': list(range(4,29,2))},
        'qwen7': {'repository': 'Qwen/Qwen2.5-7B',
                  'revision': 'd149729398750b98c0af14eb82c78cfe92750796',
                  'local_dir': 'models/pretrained/qwen2.5-7b', 'layers': list(range(4,29,2))},
    }
    for name in ['qwen35_protocol_amendment.json','qwen35_2b_protocol_amendment.json']:
        amendment = EXP/name
        if amendment.exists():
            extra=json.loads(amendment.read_text())['models']
            if set(extra)&set(specs):
                raise ValueError('An additive model amendment cannot replace an existing identity')
            specs.update(extra)
    return specs


def active_encoders():
    return ['bert', 'qwen', 'qwen35_08', 'qwen35_2', 'qwen35_4']


def encoder_protocol_path(encoder):
    return EXP/('qwen35_2b_protocol_amendment.json' if encoder=='qwen35_2' else 'qwen35_protocol_amendment.json')


def probe_encoders():
    return ['bert', 'bert_svd', *active_encoders()[1:]]


def load_frozen_text_model(encoder, folder=None, device='cpu', dtype=None):
    """Qwen3.5 loads only the language backbone, with strict key verification."""
    import torch
    from transformers import AutoModel
    folder = Path(folder) if folder is not None else ROOT/encoder_specs()[encoder]['local_dir']
    if not folder.is_dir():
        raise FileNotFoundError(f'Local encoder is not available at {folder}. Core COLING inference requires only bundled BERT; Qwen branches are optional.')
    dtype = dtype or torch.float32
    if encoder.startswith('qwen35_'):
        # Imported only in the isolated Qwen3.5 environment.
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
        from transformers.initialization import no_init_weights
        from safetensors import safe_open
        config = Qwen3_5TextConfig(**json.loads((folder/'config.json').read_text())['text_config'])
        # Keep computed nonpersistent RoPE buffers on CPU; meta construction
        # would leave those buffers without data after assigning state_dict.
        with no_init_weights():
            model = Qwen3_5TextModel(config)
        prefix = 'model.language_model.'
        state = {}
        for path in sorted(folder.glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as weights:
                for key in weights.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = weights.get_tensor(key).to(dtype)
        # Raises on any missing or extra language weights: no random initialization.
        model.load_state_dict(state, strict=True, assign=True)
        model.config.use_cache = False
    else:
        model = AutoModel.from_pretrained(folder, local_files_only=True,
                                         use_safetensors=True, torch_dtype=dtype)
    return model.to(device).eval().requires_grad_(False)
