"""Portable, mask-free task bundles with separate profiling and attack readers."""
from pathlib import Path
import json
import numpy as np
from .aes import SBOX, INV_SBOX


def byte_array(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in 'iu' or (value < 0).any() or (value > 255).any():
        raise ValueError(name + ' must contain integer bytes')
    return value.astype(np.uint8, copy=False)


def hypotheses(metadata, rule):
    keys = np.arange(256, dtype=np.uint8)[None, :]
    if rule == 'first_sbox':
        plaintext = byte_array(metadata['plaintext'], 'plaintext')
        if plaintext.ndim != 1:
            raise ValueError('plaintext must contain the selected byte, one per trace')
        return SBOX[plaintext[:, None] ^ keys]
    if rule == 'last_transition':
        ciphertext = byte_array(metadata['ciphertext'], 'ciphertext')
        if ciphertext.ndim != 2 or ciphertext.shape[1] != 16:
            raise ValueError('ciphertext must have shape [N,16]')
        return INV_SBOX[ciphertext[:, 11, None] ^ keys] ^ ciphertext[:, 7, None]
    raise ValueError('Unknown label rule: ' + rule)


def normalize_traces(traces, length):
    x = np.asarray(traces, dtype=np.float32)
    if x.ndim != 2 or not len(x) or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError('traces must be finite [samples, time] arrays')
    if x.shape[1] != length:
        if (x.shape[1], length) != (500, 700):
            raise ValueError('Select the profiling-only window before packing the task')
        old, new = np.linspace(0, 1, 500), np.linspace(0, 1, 700)
        x = np.stack([np.interp(new, old, row) for row in x]).astype(np.float32)
    return ((x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-6)).astype(np.float32)


def _split(path, length, attack=False):
    with np.load(path, allow_pickle=False) as bundle:
        forbidden = {'mask', 'masks', 'offset', 'offsets'}.intersection(bundle.files)
        if forbidden:
            raise ValueError('The paper protocol uses no mask/offset fields')
        x = normalize_traces(bundle['traces'], length)
        y = byte_array(bundle['labels'], 'labels')
        if y.shape != (len(x),):
            raise ValueError('labels must have one entry per trace')
        result = dict(traces=x, labels=y.astype(np.int64))
        if attack:
            for name in ('plaintext', 'ciphertext', 'key_byte', 'group_id', 'trace_ids'):
                if name in bundle:
                    result[name] = bundle[name].copy()
    return result


class TaskStore:
    """Read only train/validation during fitting; open attack files during evaluation."""
    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.length = config.get('trace_length', 700)
        self.tasks = {row['name']: row for row in config['tasks']}
        if len(self.tasks) != len(config['tasks']) or any('/' in n or '\\' in n or n in ('.', '..') for n in self.tasks):
            raise ValueError('Task names must be unique directory names')

    def profile(self, name):
        folder = self.root / name
        train = _split(folder / 'train.npz', self.length)
        val = _split(folder / 'validation.npz', self.length)
        expected = self.tasks[name].get('samples', {})
        for role, part in [('train', train), ('validation', val)]:
            if role in expected and len(part['labels']) != expected[role]:
                raise ValueError(f'{name} {role} count differs from the experiment configuration')
        return dict(train_traces=train['traces'], train_labels=train['labels'],
                    val_traces=val['traces'], val_labels=val['labels'])

    def attack(self, name):
        row = self.tasks[name]
        data = _split(self.root / name / 'attack.npz', self.length, attack=True)
        expected = row.get('samples', {}).get('attack')
        if expected is not None and len(data['labels']) != expected:
            raise ValueError(name + ' attack count differs from the experiment configuration')
        key = byte_array(data['key_byte'], 'key_byte')
        if key.ndim == 0:
            key = np.full(len(data['traces']), key, dtype=np.uint8)
        if key.shape != (len(data['traces']),):
            raise ValueError('key_byte must be scalar or one byte per attack trace')
        data['key_byte'] = key
        h = hypotheses(data, row['label_rule'])
        if h.shape != (len(key), 256) or not np.array_equal(h[np.arange(len(key)), key], data['labels']):
            raise ValueError('Attack labels do not match the actual key hypotheses')
        data['hypothesis_labels'] = h
        return data


def pack_task(input_path, output, rule):
    """Split a user's aligned bundle into files that separate attack access."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty task output directory')
    with np.load(input_path, allow_pickle=False) as source:
        if any('mask' in key.lower() or 'offset' in key.lower() for key in source.files):
            raise ValueError('Remove mask/offset metadata; this interface does not use it')
        parts = {}
        for prefix, filename in [('train', 'train'), ('val', 'validation'), ('attack', 'attack')]:
            x = source[prefix + '_traces']
            y = byte_array(source[prefix + '_labels'], prefix + '_labels')
            ids = np.asarray(source[prefix + '_ids'])
            if x.ndim != 2 or y.shape != (len(x),) or ids.shape != (len(x),):
                raise ValueError('Trace, label, and sample-ID lengths must match')
            if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
                raise ValueError('Sample IDs must be unique integers from one acquisition')
            parts[filename] = dict(traces=x, labels=y, trace_ids=ids)
        sets = [set(parts[key]['trace_ids'].tolist()) for key in parts]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Training, validation, and attack samples overlap')
        for name in ('plaintext', 'ciphertext', 'key_byte', 'group_id'):
            key = 'attack_' + name
            if key in source:
                parts['attack'][name] = source[key].copy()
        h = hypotheses(parts['attack'], rule)
        keys = byte_array(parts['attack']['key_byte'], 'key_byte')
        if keys.ndim == 0:
            keys = np.full(len(h), keys, dtype=np.uint8)
        if keys.shape != (len(h),) or not np.array_equal(h[np.arange(len(h)), keys], parts['attack']['labels']):
            raise ValueError('Attack labels disagree with the supplied ciphertext/plaintext and key')
    output.mkdir(parents=True, exist_ok=True)
    for name, data in parts.items():
        np.savez(output / (name + '.npz'), **data)
    return {name: len(data['labels']) for name, data in parts.items()}
