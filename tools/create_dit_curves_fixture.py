#!/usr/bin/env python3
"""Generate actual PyTorch BF16 RMSNorm/F32 modulation oracles at 16 and 5376."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    tensors = {}
    for prefix, width in [('small', 16), ('production', 5376)]:
        # Four packed rows: the kernel skips the first without a slice copy.
        coordinates = torch.arange(4 * width, dtype=torch.float32)
        x = (torch.sin(coordinates * 0.173) * 1.37 + torch.cos(coordinates * 0.037) * 0.19).reshape(4, width).to(torch.bfloat16)
        weight = (0.71 + (torch.arange(width, dtype=torch.float32) % 31) / 37.0).to(torch.bfloat16)
        # Fractional values exercise the precision lost by an early BF16 cast.
        modulation = (torch.sin(torch.arange(8 * width, dtype=torch.float32) * 0.17) * 0.13 + 1.0 / 4096).reshape(4, 2, width)
        row_map = torch.tensor([3, 0, 2], dtype=torch.int64)
        norm = torch.nn.functional.rms_norm(x[1:], (width,), weight, 1e-5)
        assert norm.dtype == torch.bfloat16
        if width == 5376:
            raw_norm = torch.nn.functional.rms_norm(x[1:].float(), (width,), weight.float(), 1e-5)
            discarded = raw_norm.contiguous().view(torch.int32).bitwise_and(65535)
            midpoint_distance = (discarded - 32768).abs()
            # Include values within 2% of a BF16 rounding midpoint, not exact ties.
            assert torch.any((midpoint_distance > 0) & (midpoint_distance < 1311))
        expected = norm.float() * (1.0 + modulation[row_map, 1]) + modulation[row_map, 0]
        assert torch.any(expected != expected.to(torch.bfloat16).float())
        for name, tensor in {'input': x, 'weight': weight, 'modulation': modulation,
                             'row_map': row_map.to(torch.int32), 'normalized': norm,
                             'expected': expected}.items():
            tensors[f'{prefix}.{name}'] = tensor.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('refusing to overwrite an existing fixture')
    save_file(tensors, str(args.output),
              metadata={'torch_version': torch.__version__, 'epsilon': '1e-5',
                        'widths': '16,5376', 'input_offset': 'one packed row',
                        'reference': 'torch.nn.functional.rms_norm BF16 output, then separate F32 multiply/add'})


if __name__ == '__main__':
    main()
