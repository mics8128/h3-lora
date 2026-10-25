#!/usr/bin/env python3
"""Lossless-layout conversion of author BF16 Eros beta5; never reads base payloads.

Official headers are a shape/dtype contract only. Curve-form AdaLN is retained,
not expanded or replaced. ComfyUI comfy/ldm/minimax/model.py consumes table
coordinates directly (no SiLU), with linear interpolation over clamped t*[N-1].
Its QKV bands are contiguous; native h3.c expects head/band interleaving.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

SOURCE_SHA256 = '098f138f3e899d03821dd8296c8db3db6fa16371e50fcdd38661d347fd9a1dfa'
SOURCE_BYTES = 40222982192
CHUNK = 8 * 1024 * 1024
EXTRAS = {'adaln_basis': [8, 2688], 'adaln_mean': [2688], 'adaln_t_table': [1025, 8]}
TIME_KEYS = {'time_embedder.proj_in.bias', 'time_embedder.proj_in.weight',
             'time_embedder.proj_out.bias', 'time_embedder.proj_out.weight'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_header(stream):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError('truncated safetensors prefix')
    size, = struct.unpack('<Q', prefix)
    if not 2 <= size <= 16 * 1024 * 1024:
        raise ValueError('invalid or excessive safetensors header')
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('truncated safetensors header')
    return json.loads(raw, object_pairs_hook=unique_object), 8 + size


def validate_source(header, payload_size):
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    ranges = []
    for name, tensor in tensors.items():
        shape = tensor['shape']
        if tensor['dtype'] != 'BF16':
            raise ValueError(f'{name}: requires author BF16 source, not INT8/ConvRot')
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'{name}: invalid shape')
        start, end = tensor['data_offsets']
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError(f'{name}: invalid data offsets')
        if end - start != math.prod(shape) * 2:
            raise ValueError(f'{name}: payload size does not match shape')
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError('source payload has gaps or overlapping tensors')
        cursor = end
    if cursor != payload_size:
        raise ValueError('source payload truncated or has trailing bytes')
    return tensors


def make_plan(tensors, schema):
    schema = {k: v for k, v in schema.items() if k != '__metadata__'}
    if set(schema) - set(tensors) != TIME_KEYS or set(tensors) - set(schema) != set(EXTRAS):
        raise ValueError('unexpected source/schema tensor names')
    output, qkv, widened = {}, [], []
    offset = 0
    for name, tensor in tensors.items():
        shape = tensor['shape']
        curve = name.endswith('.adaln_proj.linear.weight') or name.endswith('.adaln_proj.linear.bias')
        if name in EXTRAS:
            expected_shape = EXTRAS[name]
            dtype = 'F32' if name == 'adaln_t_table' else 'BF16'
        else:
            expected_shape = list(schema[name]['shape'])
            dtype = schema[name]['dtype']
            if curve:
                if name.endswith('.weight'):
                    if len(expected_shape) != 2 or expected_shape[1] != 2688:
                        raise ValueError(f'{name}: unexpected full AdaLN schema')
                    expected_shape[1] = 8
                dtype = 'F32'
        if shape != expected_shape or dtype not in ('BF16', 'F32'):
            raise ValueError(f'{name}: unexpected shape/dtype')
        if dtype == 'F32':
            widened.append(name)
        if name.endswith('.attn.qkv_proj.weight'):
            if shape != [21504, 5376]:
                raise ValueError(f'{name}: unexpected QKV shape')
            qkv.append(name)
        size = math.prod(shape) * (4 if dtype == 'F32' else 2)
        output[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
    if len(qkv) != 52:
        raise ValueError('expected 52 QKV matrices')
    curves = [k for k in tensors if k.endswith('.adaln_proj.linear.weight')]
    if len(curves) != 51:
        raise ValueError('expected 50 block and one final rank8 projection')
    return output, set(qkv), set(widened), offset


def widen_bf16(data):
    if len(data) % 2:
        raise ValueError('unaligned BF16 chunk')
    words = array.array('H', data)
    if sys.byteorder != 'little':
        words.byteswap()
    wide = array.array('I', (int(word) << 16 for word in words))
    if sys.byteorder != 'little':
        wide.byteswap()
    return wide.tobytes()


def copy_exact(src, dst, count, widen=False, hasher=None):
    while count:
        data = src.read(min(count, CHUNK))
        if not data:
            raise ValueError('source changed or truncated while converting')
        count -= len(data)
        data = widen_bf16(data) if widen else data
        dst.write(data)
        if hasher is not None:
            hasher.update(data)


def copy_qkv(src, dst, start, heads=56, head_dim=128, columns=5376, hasher=None):
    block = head_dim * columns * 2
    for head in range(heads):
        for band in range(3):
            src.seek(start + (band * heads + head) * block)
            copy_exact(src, dst, block, hasher=hasher)


def digest(stream):
    stream.seek(0)
    result = hashlib.sha256()
    while data := stream.read(CHUNK):
        result.update(data)
    return result.hexdigest()


def convert(source, schema_path, output, expected_sha256=SOURCE_SHA256):
    source, schema_path, output = map(Path, (source, schema_path, output))
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes, object_pairs_hook=unique_object)
    if output.exists():
        raise ValueError('output already exists; refusing to overwrite')
    with source.open('rb') as src:
        before = source.stat()
        source_hash = digest(src)
        if source_hash != expected_sha256:
            raise ValueError(f'source SHA256 mismatch: {source_hash}')
        if expected_sha256 == SOURCE_SHA256 and before.st_size != SOURCE_BYTES:
            raise ValueError('official beta5 source size mismatch')
        src.seek(0)
        header, base = read_header(src)
        tensors = validate_source(header, before.st_size - base)
        plan, qkv, widened, payload_size = make_plan(tensors, schema)
        plan['__metadata__'] = {
            'source_filename': source.name, 'source_sha256': source_hash,
            'schema_sha256': hashlib.sha256(schema_bytes).hexdigest(),
            'conversion': 'h3.c beta5 rank8-v1; QKV band/head permutation; exact BF16 widening; no base payloads',
            'adaln_mode': 'curve_rank8_no_silu',
            'original_metadata': json.dumps(header.get('__metadata__', {}), sort_keys=True),
            'reference_loader': 'https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/comfy/ldm/minimax/model.py',
        }
        encoded = json.dumps(plan, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.partial')
        created = False
        try:
            with temporary.open('xb') as dst:
                created = True
                output_hash = hashlib.sha256()
                prefix = struct.pack('<Q', len(encoded))
                dst.write(prefix)
                dst.write(encoded)
                output_hash.update(prefix)
                output_hash.update(encoded)
                for name, tensor in tensors.items():
                    start, end = tensor['data_offsets']
                    if name in qkv:
                        copy_qkv(src, dst, base + start, hasher=output_hash)
                    else:
                        src.seek(base + start)
                        copy_exact(src, dst, end - start, name in widened, output_hash)
                if dst.tell() != 8 + len(encoded) + payload_size:
                    raise ValueError('output length mismatch')
            after = source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('source changed during conversion')
            # Atomic publication without overwriting an existing user file.
            import os
            os.link(temporary, output)
            temporary.unlink()
        except BaseException:
            if created:
                temporary.unlink(missing_ok=True)
            raise
    return {'output': str(output), 'source_sha256': source_hash,
            'output_sha256': output_hash.hexdigest(),
            'qkv_permuted': len(qkv), 'widened': sorted(widened), 'payload_bytes': payload_size}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--schema', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-sha256', default=SOURCE_SHA256)
    args = parser.parse_args()
    try:
        print(json.dumps(convert(args.source, args.schema, args.output, args.expected_sha256), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'conversion failed: {exc}\n')


if __name__ == '__main__':
    main()
