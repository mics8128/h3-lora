import importlib.util
import io
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('converter', Path(__file__).parents[1] / 'tools/convert_eros_beta5.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def test_widen_preserves_all_bf16_bit_patterns(self):
        source = struct.pack('<65536H', *range(65536))
        result = struct.unpack('<65536I', converter.widen_bf16(source))
        self.assertEqual(result, tuple(n << 16 for n in range(65536)))

    def test_qkv_permutation_on_distinct_rows(self):
        source = io.BytesIO(struct.pack('<12H', *range(12)))
        output = io.BytesIO()
        converter.copy_qkv(source, output, 0, heads=2, head_dim=1, columns=2)
        self.assertEqual(struct.unpack('<12H', output.getvalue()),
                         (0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11))

    def test_truncation_fails_not_partial_success(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            converter.copy_exact(io.BytesIO(b'ab'), io.BytesIO(), 4)

    def test_overlaps_and_trailing_bytes_rejected(self):
        a = {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}
        b = {'dtype': 'BF16', 'shape': [2], 'data_offsets': [2, 6]}
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            converter.validate_source({'a': a, 'b': b}, 6)
        with self.assertRaisesRegex(ValueError, 'trailing'):
            converter.validate_source({'a': a}, 5)

    def test_int8_convrot_cannot_be_mislabeled_bf16(self):
        with self.assertRaisesRegex(ValueError, 'INT8/ConvRot'):
            converter.validate_source({'w': {'dtype': 'I8', 'shape': [2], 'data_offsets': [0, 2]}}, 2)

    def test_duplicate_header_keys_rejected(self):
        raw = b'{"a":1,"a":2}'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            converter.read_header(io.BytesIO(struct.pack('<Q', len(raw)) + raw))

    def test_rank8_plan_preserves_curve_and_f32_projection(self):
        schema = {k: {'dtype': 'F32', 'shape': [1]} for k in converter.TIME_KEYS}
        tensors = {k: {'dtype': 'BF16', 'shape': shape} for k, shape in converter.EXTRAS.items()}
        for index in range(52):
            name = f'blocks.{index}.attn.qkv_proj.weight'
            schema[name] = {'dtype': 'BF16', 'shape': [21504, 5376]}
            tensors[name] = dict(schema[name])
        for index in range(51):
            name = (f'blocks.{index}' if index < 50 else 'final_layer') + '.adaln_proj.linear.weight'
            rows = 96768 if index < 50 else 10752
            schema[name] = {'dtype': 'BF16', 'shape': [rows, 2688]}
            tensors[name] = {'dtype': 'BF16', 'shape': [rows, 8]}
        plan, qkv, wide, _ = converter.make_plan(tensors, schema)
        self.assertEqual(plan['adaln_t_table']['dtype'], 'F32')
        self.assertEqual(plan['final_layer.adaln_proj.linear.weight']['shape'], [10752, 8])
        self.assertIn('final_layer.adaln_proj.linear.weight', wide)
        self.assertEqual(len(qkv), 52)
        tensors['adaln_t_table']['shape'] = [1024, 8]
        with self.assertRaisesRegex(ValueError, 'shape'):
            converter.make_plan(tensors, schema)


if __name__ == '__main__':
    unittest.main()
