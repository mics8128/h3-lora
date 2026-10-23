#!/usr/bin/env python3
"""Run the public Turbo inventory on ted; standard library only, no downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

PROMPT = ('An adult woman wearing a blue jacket, a white shirt, and long trousers walks '
          'along a sunny park path, turns toward the camera, smiles and waves. '
          'Continuous natural walking motion, stable camera, green trees in the background. '
          'Audio: synchronized footsteps, gentle wind in the leaves and distant birds. No music.')
REF_PROMPT = ('<Picture 1> supplies the adult woman identity and clothing. Preserve her identity '
              'and clothing from Picture 1. ' + PROMPT)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture(cmd):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'command': cmd, 'returncode': p.returncode, 'output': p.stdout}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Required nonempty file missing: ' + str(path))


def inventory_base(root, ref):
    """Validate all indexed shards and record index/config identities, without reading weights."""
    components = ['FL2VA/text_encoder', 'FL2VA/transformer', 'FL2VA/video_vae/source',
                  'FL2VA/audio_vae']
    if ref:
        components += ['Ref2VA/transformer', 'Ref2VA/text_encoder',
                       'Ref2VA/video_vae/source', 'Ref2VA/audio_vae']
    result = []
    required = ['FL2VA/transformer/config.json', 'FL2VA/tokenizer/tokenizer.json']
    if ref:
        required += ['Ref2VA/transformer/config.json', 'Ref2VA/tokenizer/tokenizer.json']
    for name in required:
        p = root / name
        require_file(p)
        result.append({'path': str(p), 'sha256': digest(p)})
    for component in components:
        directory = root / component
        indices = list(directory.glob('*.safetensors.index.json'))
        weights = set()
        for index in indices:
            mapping = json.loads(index.read_text())['weight_map']
            weights.update(directory / name for name in mapping.values())
            result.append({'path': str(index), 'sha256': digest(index)})
        if not indices:
            weights.update(directory.glob('*.safetensors'))
        if not weights:
            raise ValueError('No safetensors weights in ' + str(directory))
        for weight in sorted(weights):
            require_file(weight)
            result.append({'path': str(weight), 'bytes': weight.stat().st_size})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for option in ['engine-dir', 'model-dir', 'ref-model-dir', 'lora-dir', 'reference-image', 'output-dir']:
        ap.add_argument('--' + option, type=Path, required=True)
    ap.add_argument('--suite', choices=['representatives', 'all'], default='representatives')
    ap.add_argument('--frames', type=int, default=124)
    ap.add_argument('--width', type=int)
    ap.add_argument('--height', type=int)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--only', help='Comma-separated file IDs or case IDs; file 3271482 includes both samplers')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--execute', action='store_true',
                    help='Explicitly permit heavy inference on the capable execution machine')
    args = ap.parse_args()
    if args.frames <= 0 or any(x is not None and x <= 0 for x in [args.width, args.height]):
        ap.error('Frames and dimensions must be positive')
    if not args.preflight_only and not args.execute:
        ap.error('Pass --execute only on the capable remote machine; local use is preflight-only')
    manifest = json.loads(Path(__file__).with_name('turbo_models.json').read_text())
    cases = [dict(row, case_id=str(row['id'])) for row in manifest['files']
             if args.suite == 'all' or row['representative']]
    discrepancy = dict(next(row for row in manifest['files'] if row['id'] == 3271482))
    discrepancy.update(manifest['discrepancy'])
    discrepancy['case_id'] = '3271482-official-lcm'
    cases.append(discrepancy)
    if args.only:
        only = set(args.only.split(','))
        known = {c['case_id'] for c in cases} | {str(c['id']) for c in cases}
        if only - known:
            ap.error('Unknown/unselected --only values: ' + ', '.join(sorted(only - known)))
        cases = [c for c in cases if c['case_id'] in only or str(c['id']) in only]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'summary.json'
    if summary_path.exists():
        ap.error('Output directory already contains summary.json; choose a fresh directory')
    summary = {'suite': args.suite, 'manifest_sha256': digest(Path(__file__).with_name('turbo_models.json')),
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'host': socket.gethostname(), 'platform': platform.platform(),
               'deviations': {'frames': args.frames if args.frames != 124 else None,
                              'width_override': args.width, 'height_override': args.height},
               'cases': [], 'preflight_ok': False, 'success': False}
    try:
        engine = args.engine_dir.resolve() / 'h3'
        require_file(engine)
        if not os.access(engine, os.X_OK):
            raise ValueError('Engine is not executable: ' + str(engine))
        for program in ['ffmpeg', 'ffprobe', 'git', 'sysctl', 'vm_stat']:
            if not shutil.which(program):
                raise ValueError('Required executable missing: ' + program)
        if not Path('/usr/bin/time').is_file():
            raise ValueError('macOS /usr/bin/time is required')
        help_result = capture([str(engine), '--help'])
        summary['engine_help'] = help_result
        required = ['--model-dir', '--prompt', '--output', '--lora', '--lora-model', '--sampler',
                    '--scheduler', '--video-shift', '--audio-shift', '--width', '--height',
                    '--frames', '--steps', '--seed', '--layers', '--reuse', '--core-reuse',
                    '--ref-image', '--ref-image-size']
        missing = [flag for flag in required if flag not in help_result['output']]
        if help_result['returncode'] or missing:
            raise ValueError('Engine help failed or required options missing: ' + repr(missing))
        summary['engine_sha256'] = digest(engine)
        summary['engine_commit'] = capture(['git', '-C', str(args.engine_dir), 'rev-parse', 'HEAD'])
        summary['engine_worktree'] = capture(['git', '-C', str(args.engine_dir), 'status', '--porcelain'])
        summary['hardware'] = capture(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu'])
        require_file(args.reference_image)
        summary['reference_sha256'] = digest(args.reference_image)
        roots = {c['target']: (args.ref_model_dir if c['target'] == 'ref2va' else args.model_dir).resolve()
                 for c in cases}
        summary['bases'] = {target: inventory_base(root, target == 'ref2va') for target, root in roots.items()}
        summary['adapters'] = []
        for file_id in sorted({c['id'] for c in cases}):
            row = next(c for c in cases if c['id'] == file_id)
            weight = args.lora_dir / row['name']
            require_file(weight)
            actual = digest(weight)
            if weight.stat().st_size != row['bytes'] or actual.lower() != row['sha256'].lower():
                raise ValueError('Adapter size/SHA-256 mismatch: ' + str(weight))
            summary['adapters'].append({'file_id': file_id, 'sha256': actual})
        summary['preflight_ok'] = True
        save(summary_path, summary)
        if args.preflight_only:
            summary['preflight_only'] = True
            save(summary_path, summary)
            return 0
        for case in cases:
            directory = args.output_dir / case['case_id']
            directory.mkdir(exist_ok=False)
            output = directory.resolve() / 'output.mp4'
            prompt = REF_PROMPT if case['target'] == 'ref2va' else PROMPT
            cmd = [str(engine), '--model-dir', str(roots[case['target']]), '--prompt', prompt,
                   '--output', str(output), '--lora', str((args.lora_dir / case['name']).resolve()),
                   str(case['strength']), '--lora-model', case['target'], '--scheduler', 'simple',
                   '--sampler', case['sampler'], '--video-shift', str(case['video_shift']),
                   '--audio-shift', str(case['audio_shift']), '--steps', str(case['steps']),
                   '--width', str(args.width or case['width']), '--height', str(args.height or case['height']),
                   '--frames', str(args.frames), '--seed', str(args.seed), '--layers', '50',
                   '--reuse', '1', '--core-reuse', '1']
            cmd += ['--profile']
            if case['target'] == 'ref2va':
                cmd += ['--ref-image', str(args.reference_image.resolve()), '--ref-image-size', 'match']
            record = {'case': case, 'command': cmd, 'shell_command': shlex.join(cmd), 'prompt': prompt,
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                      'reference_sha256': summary['reference_sha256'] if case['target'] == 'ref2va' else None,
                      'seed': args.seed, 'rng_policy': 'Native engine seeded RNG; not PyTorch-identical noise',
                      'precision_policy': 'Native engine default; full loading/precision diagnostics in engine.log',
                      'cfg': 1, 'requested_frames': args.frames, 'expected_fps': 24, 'success': False}
            try:
                record['swap_before'] = capture(['sysctl', 'vm.swapusage'])
                record['vm_before'] = capture(['vm_stat'])
                started = time.monotonic()
                with (directory / 'engine.log').open('w') as log:
                    completed = subprocess.run(['/usr/bin/time', '-l'] + cmd,
                        cwd=args.engine_dir.resolve(), stdout=log, stderr=subprocess.STDOUT)
                record['wall_seconds'] = time.monotonic() - started
                record['engine_returncode'] = completed.returncode
                record['swap_after'] = capture(['sysctl', 'vm.swapusage'])
                record['vm_after'] = capture(['vm_stat'])
                record['physical_memory_evidence'] = 'engine.log: /usr/bin/time -l maximum resident set size and peak memory footprint; swap snapshots are host-wide'
                if completed.returncode:
                    raise ValueError('Engine failed; see engine.log')
                traces = re.findall(r'h3 sampler: (\{[^\n]+\})',
                    (directory / 'engine.log').read_text())
                if len(traces) != 1:
                    raise ValueError('Expected one observed native sampler trace')
                observed = json.loads(traces[0])
                record['observed_sampler'] = observed
                if (observed['nfe'] != case['steps'] or not observed['success'] or
                    observed['kind'] != case['sampler'] or observed['scheduler'] != 'simple' or
                    len(observed['video_sigmas']) != case['steps'] + 1 or
                    len(observed['audio_sigmas']) != case['steps'] + 1):
                    raise ValueError('Observed sampler execution differs from the requested contract')
                require_file(output)
                probe = capture(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                 '-show_format', '-of', 'json', str(output)])
                record['ffprobe'] = probe
                if probe['returncode']:
                    raise ValueError('ffprobe failed')
                media = json.loads(probe['output'])
                videos = [s for s in media['streams'] if s['codec_type'] == 'video']
                audios = [s for s in media['streams'] if s['codec_type'] == 'audio']
                if not videos or not audios:
                    raise ValueError('Output requires video AND audio streams')
                video = videos[0]
                if (int(video.get('nb_read_frames', 0)) != args.frames or
                    video['width'] != (args.width or case['width']) or
                    video['height'] != (args.height or case['height'])):
                    raise ValueError('Actual media dimensions/frame count differ from request')
                numerator, denominator = map(int, video['avg_frame_rate'].split('/'))
                if not denominator or numerator / denominator != 24:
                    raise ValueError('Actual frame rate is not 24 fps')
                with (directory / 'decode.log').open('w') as log:
                    decode = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output),
                                             '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'],
                                            stdout=log, stderr=subprocess.STDOUT)
                record['decode_returncode'] = decode.returncode
                if decode.returncode:
                    raise ValueError('Full audiovisual decode failed; see decode.log')
                record.update(success=True, output=str(output), output_bytes=output.stat().st_size,
                              output_sha256=digest(output), actual_media=media)
            except (OSError, ValueError, KeyError) as error:
                record['error'] = str(error)
            save(directory / 'result.json', record)
            summary['cases'].append(record)
            save(summary_path, summary)
        summary['success'] = bool(summary['cases']) and all(r['success'] for r in summary['cases'])
    except (OSError, ValueError, KeyError) as error:
        summary['error'] = str(error)
    save(summary_path, summary)
    print(json.dumps({'summary': str(summary_path), 'success': summary['success'],
                      'passed': sum(c['success'] for c in summary['cases']),
                      'failed': sum(not c['success'] for c in summary['cases']),
                      'error': summary.get('error')}, indent=2))
    return 0 if summary['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
