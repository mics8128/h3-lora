#!/usr/bin/env python3
"""Durable, local H3 task planner and serialized native worker (stdlib only).

Configuration: {"engine":"/absolute/h3", "device":"metal:0", "profiles":
{"name-v1":{"model_dir":"/models", "target":"ref2va", "dependencies":
[{"role":"transformer","path":"/models/model.safetensors","sha256":"..."}],
"params":{"width":512,"height":512,"steps":20},"max_frames":124,
"quality_evidence":"unverified","exact":true}}}.
Paths are administrator configuration, never model-authored task arguments.
Dependencies MUST enumerate every selected model, encoder, decoder, adapter,
tokenizer/config and shader dependency; the native executable is pinned separately.
Digests are cached against device/inode/size/mtime_ns/ctime_ns, not filenames alone.
The configured device ID must be shared by every local state root using that GPU.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import sqlite3
import stat
import struct
import subprocess
import sys
import selectors
import signal
import time
import tempfile
import uuid

TERMINAL = {'paused', 'cancelled', 'succeeded', 'failed', 'interrupted'}
NORMALIZATION_POLICIES = {
    'require_exact': {'id': 'require_exact', 'version': 1, 'transformations': []},
    'standard_media_v1': {'id': 'standard_media_v1', 'version': 1,
        'transformations': ['cover_crop', 'reference_resize', 'time_resample', 'audio_resample', 'audio_right_pad'],
        'excludes': ['overrun_trim', 'clip_floor', 'frame_floor', 'short_clip_to_image',
                     'source_selection', 'condition_drop', 'speech_rewrite', 'internal_downrender',
                     'reference_trim', 'reference_frame_floor', 'reference_audio_trim', 'reference_first_frame']}}


class TaskError(Exception):
    def __init__(self, code, field, message, details=None, repairs=None):
        self.value = dict(code=code, field=field, message=message,
                          details=details or {}, repairs=repairs or [], retryable=False)
        super().__init__(message)


def fail(code, field, message, repairs=None, **details):
    raise TaskError(code, field, message, details, repairs)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with open(temporary, 'w') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        fail('invalid_request', field, 'Expected an integer.', minimum=minimum)
    return value


def lower_intent(intent, references):
    """Deterministic formatting only: no ContextIR, asset captioning or inference."""
    allowed = {'setting', 'subjects', 'actions', 'camera', 'dialogue', 'reference_use', 'ambience', 'music'}
    if not isinstance(intent, dict) or set(intent) - allowed:
        fail('invalid_request', 'task.intent', 'Unknown structured intent field.')
    def text(value, field):
        if not isinstance(value, str) or len(value) > 16000:
            fail('invalid_request', field, 'Expected bounded text (at most 16000 characters).')
        if re.search(r'</?d>|<(?:Subject|Picture|Video|Audio)\s+\d+>|(?:^|\n)(?:subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):', value):
            fail('invalid_request', field, 'Reserved native prompt delimiters must not appear in plain structured text.')
        return value
    labels, counts = {}, {'Picture': 0, 'Video': 0, 'Audio': 0}
    for ref in references:
        kinds = {'image': ['Picture'], 'video': ['Video'], 'audio': ['Audio'], 'video_audio': ['Audio', 'Video']}.get(ref.get('role'))
        if ref.get('role') == 'video' and ref.get('include_embedded_audio'):
            kinds = ['Audio', 'Video']
        if kinds is None:
            fail('invalid_request', 'task.references', 'Unknown reference role.')
        if ref.get('asset_id') in labels:
            fail('intent_conflict', 'task.references', 'Structured reference IDs must be unambiguous.')
        present = []
        for kind in kinds:
            counts[kind] += 1
            present.append(f'<{kind} {counts[kind]}>')
        labels[ref['asset_id']] = ' '.join(present)
    subjects, speakers, subject_ids = [], {}, {}
    for line in intent.get('dialogue', []):
        if not isinstance(line, dict) or not isinstance(line.get('speaker_id'), str):
            fail('invalid_request', 'task.intent.dialogue', 'Dialogue requires a string speaker_id.')
        identity = line['speaker_id']
        if identity not in speakers:
            speakers[identity] = f'(S{len(speakers)+1})'
    for index, subject in enumerate(intent.get('subjects', []), 1):
        field = f'task.intent.subjects[{index-1}]'
        if not isinstance(subject, dict) or set(subject) - {'id', 'description', 'reference_ids'}:
            fail('invalid_request', field, 'Unknown subject field.')
        identity = subject.get('id')
        if not isinstance(identity, str) or not identity or identity in subject_ids:
            fail('intent_conflict', field + '.id', 'Subject IDs must be unique nonempty strings.')
        subject_ids[identity] = f'<Subject {index}>' + (' ' + speakers[identity] if identity in speakers else '')
        refs = []
        for asset_id in subject.get('reference_ids', []):
            if asset_id not in labels:
                fail('unknown_asset', field + '.reference_ids', 'Subject reference must appear in ordered references.')
            refs.append(labels[asset_id])
        subjects.append(subject_ids[identity] + ': ' + text(subject.get('description', ''), field) +
                        (' Reference: ' + ', '.join(refs) + '.' if refs else ''))
    retention = []
    for index, use in enumerate(intent.get('reference_use', [])):
        field = f'task.intent.reference_use[{index}]'
        if not isinstance(use, dict) or set(use) - {'asset_id', 'role', 'relationship', 'use', 'subject_id'}:
            fail('invalid_request', field, 'Unknown reference-use field.')
        identity = use.get('asset_id')
        if identity not in labels:
            fail('unknown_asset', field + '.asset_id', 'Reference use must name an ordered reference.')
        reference = next(r for r in references if r['asset_id'] == identity)
        role = use.get('role', reference['role'])
        audio_bearing = reference['role'] == 'video_audio' or (reference['role'] == 'video' and reference.get('include_embedded_audio'))
        if role != reference['role'] and not (audio_bearing and role in ('audio', 'video')):
            fail('intent_conflict', field + '.role', 'Reference role conflicts with the ordered reference.')
        if use.get('use') == 'voice':
            if not (role in ('audio', 'video_audio') or (role == 'video' and reference.get('include_embedded_audio'))):
                fail('intent_conflict', field + '.use', 'Voice binding requires an audio-bearing reference.')
            subject_id = use.get('subject_id')
            if subject_id not in subject_ids:
                fail('intent_conflict', field + '.subject_id', 'Voice binding requires a defined subject.')
            audio_label = re.search(r'<Audio \d+>', labels[identity]).group()
            retention.append(audio_label + ' is the voice-timbre reference for ' + subject_ids[subject_id] + '.')
            if 'relationship' not in use:
                continue
        elif 'use' in use:
            fail('invalid_request', field + '.use', 'Only explicit voice binding is defined; other reference uses require relationship.')
        markers = ({'reference', 'weak_reference', 'fully_copy', 'partially_copy'} if role == 'audio' else
                   {'fully_preserved', 'partially_preserved', 'attribute_transfer', 'weak_reference'})
        if use.get('relationship') not in markers:
            fail('invalid_request', field + '.relationship', 'Unknown explicit retention relationship.', allowed=sorted(markers))
        selected_label = labels[identity]
        if role in ('audio', 'video'):
            selected_label = re.search(r'<Audio \d+>' if role == 'audio' else r'<Video \d+>', selected_label).group()
        retention.append(selected_label + ': ' + use['relationship'])
    actions = intent.get('actions', [])
    if isinstance(actions, str):
        actions = [actions]
    if not isinstance(actions, list):
        fail('invalid_request', 'task.intent.actions', 'Actions must be text or a list of text.')
    action_text = ' '.join(text(a, 'task.intent.actions') for a in actions)
    dialogue = []
    for index, line in enumerate(intent.get('dialogue', [])):
        field = f'task.intent.dialogue[{index}]'
        if not isinstance(line, dict) or set(line) - {'speaker_id', 'language', 'text'}:
            fail('invalid_request', field, 'Unknown dialogue field.')
        if line.get('speaker_id') not in subject_ids:
            fail('intent_conflict', field + '.speaker_id', 'Dialogue speaker must be a defined subject.')
        language = text(line.get('language', ''), field + '.language')
        if not language:
            fail('invalid_request', field + '.language', 'Dialogue language must be explicit.')
        dialogue.append(speakers[line['speaker_id']] + f' <d>[{language}] ' + text(line.get('text', ''), field + '.text') + '</d>')
    setting = text(intent.get('setting', ''), 'task.intent.setting')
    camera = text(intent.get('camera', ''), 'task.intent.camera')
    prefix = '[reference generation + audio reference] ' if labels and counts['Audio'] else '[reference generation] ' if labels else ''
    sections = {'subject_definitions': '\n'.join(subjects), 'summary': prefix + ' '.join(x for x in (setting, action_text) if x),
                'retention_analysis': '\n'.join(retention),
                'detailed_description': '[Shot1]\n' + '\n'.join(x for x in (setting, action_text, camera, *dialogue) if x),
                'overall_soundscape': text(intent.get('ambience', ''), 'task.intent.ambience'),
                'non_diegetic_music': text(intent.get('music', ''), 'task.intent.music')}
    return '\n\n'.join(f'{name}:\n{value or "N/A"}' for name, value in sections.items())


class ProcessAborted(Exception):
    """A requested throughput process abort, not a native sampler boundary."""


def reference_canvas(asset, params, kind):
    """Mirror the released h3_host reference-sizing equations, not guide crop."""
    width, height = asset['width'], asset['height']
    if kind == 'image':
        size_policy = params.get('reference_image_size', 0)
        if size_policy not in (0, 1):
            fail('profile_incompatible', 'task.profile', 'Unknown native reference image sizing policy.')
        scale = (min(1, 2048 / min(width, height)) if size_policy == 1 else
                 min(1, math.sqrt(params['width'] * params['height'] / (width * height))))
        return [max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)]
    ratio = width / height
    nominal_w, nominal_h = (768 * ratio, 768) if ratio >= 1 else (768, 768 / ratio)
    if nominal_w * nominal_h > 768 * 1344:
        scale = math.sqrt(768 * 1344 / (nominal_w * nominal_h))
        nominal_w, nominal_h = nominal_w * scale, nominal_h * scale
    result = [max(32, round(nominal_w / 32) * 32), max(32, round(nominal_h / 32) * 32)]
    if width * height < result[0] * result[1]:
        result = [max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)]
    return result


class Tasks:
    def __init__(self, root, config=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ('assets', 'plans', 'jobs'):
            (self.root / name).mkdir(exist_ok=True)
        self.config = config or {}
        self.db = sqlite3.connect(self.root / 'tasks.sqlite', timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, digest TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, plan TEXT NOT NULL, state TEXT NOT NULL,
 body TEXT NOT NULL, worker TEXT);
CREATE TABLE IF NOT EXISTS idempotency(key TEXT PRIMARY KEY, plan TEXT NOT NULL, job TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS leases(device TEXT PRIMARY KEY, owner TEXT NOT NULL, pid INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS fingerprints(path TEXT PRIMARY KEY, signature TEXT NOT NULL, sha TEXT NOT NULL);
''')

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def get(self, table, identity, field):
        row = self.db.execute('SELECT body FROM ' + table + ' WHERE id=?', (identity,)).fetchone()
        if not row:
            fail('unknown_asset' if table == 'assets' else 'invalid_request', field,
                 'Unknown managed identity.', identity=identity)
        return json.loads(row['body'])
    def sha(self, path):
        path = str(Path(path).resolve())
        info = os.stat(path)
        signature = canonical([info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns])
        cached = self.db.execute('SELECT signature,sha FROM fingerprints WHERE path=?', (path,)).fetchone()
        if cached and cached['signature'] == signature:
            return cached['sha']
        sha = file_digest(path)
        after = os.stat(path)
        if signature != canonical([after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]):
            fail('dependency_changed', 'path', 'Dependency changed during fingerprinting.')
        self.db.execute('INSERT OR REPLACE INTO fingerprints VALUES(?,?,?)', (path, signature, sha))
        return sha


    def media_tool(self, name):
        return self.config.get(name) or os.environ.get('H3_' + name.upper()) or name

    def native_environment(self):
        environment = os.environ.copy()
        for name in ('ffmpeg', 'ffprobe'):
            environment['H3_' + name.upper()] = self.media_tool(name)
        return environment

    def native(self):
        engine = self.config.get('engine')
        if not engine or not Path(engine).is_file():
            return {'schema_version': 1, 'build_id': None, 'capabilities': {}}
        try:
            environment = self.native_environment()
            p = subprocess.run([engine, '--capabilities-json'], capture_output=True, text=True, timeout=20, env=environment)
            value = json.loads(p.stdout)
            if p.returncode or value.get('schema_version') != 1:
                raise ValueError('Unsupported capability protocol')
            value['executable_sha256'] = self.sha(engine)
            value['execution_environment'] = {k: v for k, v in environment.items() if k.startswith('H3_') and not k.startswith('H3_TASK_')}
            value['media_tools'] = {name: {'path': shutil.which(self.media_tool(name)),
                'sha256': self.sha(shutil.which(self.media_tool(name))) if shutil.which(self.media_tool(name)) else None}
                for name in ('ffmpeg', 'ffprobe')}
            return value
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            fail('engine_failed', 'config.engine', 'Native capability probe failed.', reason=str(e))

    def profiles(self):
        result = []
        for name, source in sorted(self.config.get('profiles', {}).items()):
            profile = json.loads(canonical(source))
            policy_id = profile.setdefault('normalization_policy', 'standard_media_v1')
            if policy_id not in NORMALIZATION_POLICIES:
                fail('profile_incompatible', 'config.profiles.' + name + '.normalization_policy', 'Unknown versioned media normalization policy.')
            profile['normalization_policy_definition'] = NORMALIZATION_POLICIES[policy_id]
            profile.setdefault('execution_mode', self.config.get('execution_mode', 'throughput'))
            if profile['execution_mode'] not in ('throughput', 'resumable'):
                fail('profile_incompatible', 'config.profiles.' + name + '.execution_mode', 'Unknown execution mode.')
            profile['reference_sizing'] = {'version': 1, 'image': 'native match/max area then nearest32 stretch',
                                           'video': 'native768_short_edge_max1032192_pixels_no_upscale_nearest32_stretch'}
            missing = []
            for dep in profile.get('dependencies', []):
                if not Path(dep['path']).is_file():
                    missing.append(dep)
                elif not dep.get('sha256') or self.sha(dep['path']) != dep['sha256']:
                    missing.append(dict(dep, reason='digest_mismatch'))
            if not profile.get('dependencies'):
                missing.append({'role': 'explicit_model_inventory', 'reason': 'not_configured'})
            result.append(dict(profile, id=name, availability='missing_model' if missing else 'available',
                               missing=missing, profile_digest=digest(profile)))
        return result

    def capabilities(self, request):
        return dict(schema_version=1, native=self.native(), profiles=self.profiles(),
                    operations=['capabilities', 'profiles', 'asset.register', 'prepare', 'submit',
                                'inspect', 'continue', 'revise', 'resume', 'pause', 'cancel', 'worker'],
                    execution_modes={'throughput': {'pause': False, 'resume': False, 'cancel': 'process_abort_without_checkpoint'},
                                     'resumable': {'pause': True, 'resume': True, 'cancel': 'transition_boundary'}},
                    normalization_policies=NORMALIZATION_POLICIES,
                    constraints={'fps': 24, 'frame_lattice': '17k+5', 'mask': '0 preserve, 1 edit',
                                 'guide_source_frame_clock': 24,
                                 'media_formats': ['PNG', 'JPEG', 'RIFF-WAV', 'FLAC', 'Ogg', 'MP3', 'self-contained MP4/MOV'],
                                 'one_worker_per_device': True})

    def probe_native(self, path):
        with open(path, 'rb') as f:
            header = f.read(40)
            if len(header) != 40 or header[:8] != b'H3NATIVE':
                return None
            version, kind, endian, reserved, size, checksum = struct.unpack('=IIIIQQ', header[8:])
            if (version != 1 or kind not in (4, 5) or endian != 0x01020304 or reserved or
                    size != Path(path).stat().st_size - 40 or size > 64 * 1024 ** 3):
                fail('latent_incompatible', 'path', 'Invalid native envelope header.')
            value = 14695981039346656037
            for block in iter(lambda: f.read(1024 * 1024), b''):
                for byte in block:
                    value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
            if value != checksum:
                fail('latent_incompatible', 'path', 'Native envelope checksum mismatch.')
            f.seek(40)
            if kind == 5:
                if size < 64:
                    fail('latent_incompatible', 'path', 'Truncated latent dimensions.')
                frames, height, width, audio_frames, vc, ac, source_frames, mapping, pad = struct.unpack('=7QII', f.read(64))
                if pad or mapping not in (0, 1) or (mapping == 1 and not source_frames):
                    fail('latent_incompatible', 'path', 'Invalid temporal mapping provenance.')
                if source_frames and (source_frames < 5 or (source_frames - 5) % 17 or
                        frames != ((source_frames - 5) // 17) * 5 + 2):
                    fail('latent_incompatible', 'path', 'Latent time count disagrees with generated cadence.')
                if (vc != 24 * frames * height * width or ac != 64 * audio_frames or
                    (not vc and (frames or height or width)) or (not ac and audio_frames) or
                    not (vc or ac) or size != 64 + 4 * (vc + ac)):
                    fail('latent_incompatible', 'path', 'Invalid native latent shape.')
                metadata = dict(schema_version=1, kind='clean', normalization='h3-native',
                                video_frames=frames, latent_height=height, latent_width=width, audio_frames=audio_frames,
                                source_frames=source_frames, temporal_mapping=mapping)
                streams = (['video'] if vc else []) + (['audio'] if ac else [])
            else:
                if size < 136:
                    fail('checkpoint_incompatible', 'path', 'Truncated solver state.')
                fields = struct.unpack('=12Q', f.read(96))
                solver, steps, start, vc, ac, seed, step, history, rng, r0, r1, r2 = fields
                identity = f.read(32).hex()
                arrays = 7 if solver == 3 else 6
                revision = 1 if solver == 3 else 0
                if (solver > 3 or steps > 100000 or not (vc or ac) or start > step or step > steps or r0 > 1 or r1 != revision or r2 or
                        size != 136 + 8 * (steps + 1) + 4 * arrays * (vc + ac)):
                    fail('checkpoint_incompatible', 'path', 'Invalid or unsupported solver state dimensions/revision.')
                metadata = dict(schema_version=1, kind='checkpoint', request_digest=identity,
                                steps=steps, step=step, sampler=solver)
                streams = (['video'] if vc else []) + (['audio'] if ac else [])
                f.seek(40 + 128)
            low, high = math.inf, -math.inf
            for block in iter(lambda: f.read(1024 * 1024), b''):
                if len(block) % 4:
                    fail('latent_incompatible', 'path', 'Truncated native tensor.')
                for (value,) in struct.iter_unpack('=f', block):
                    if not math.isfinite(value):
                        fail('latent_incompatible', 'path', 'Nonfinite native tensor.')
                    low, high = min(low, value), max(high, value)
            if kind == 5:
                metadata['value_range'] = [low, high]
            return dict(kind='latent', streams=streams, metadata=metadata)


    def media_demuxer(self, path):
        """Classify before any multimedia tool can dereference an input."""
        with open(path, 'rb') as source:
            magic = source.read(64)
        if magic.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png_pipe'
        if magic.startswith(b'\xff\xd8\xff'):
            return 'jpeg_pipe'
        if magic[:4] == b'RIFF' and magic[8:12] == b'WAVE':
            return 'wav'
        if magic.startswith(b'fLaC'):
            return 'flac'
        if magic.startswith(b'OggS'):
            return 'ogg'
        if magic.startswith(b'ID3') or (len(magic) >= 2 and magic[0] == 255 and magic[1] & 224 == 224):
            return 'mp3'
        if magic[4:8] in (b'ftyp', b'moov', b'wide', b'free', b'mdat'):
            self.self_contained_mov(path)
            return 'mov'
        fail('invalid_request', 'path', 'Only self-contained PNG, JPEG, WAV, FLAC, Ogg, MP3 and MP4/MOV media are accepted; playlists and external-resource formats are prohibited.')

    def self_contained_mov(self, path):
        # ISO BMFF/QuickTime data references are not pinned by copying a movie.
        # Inspect data-reference tables without reading potentially huge mdat.
        containers = {b'moov', b'trak', b'mdia', b'minf', b'dinf'}
        remaining = 100000
        with open(path, 'rb') as source:
            def boxes(start, end, depth=0):
                nonlocal remaining
                if depth > 16:
                    fail('invalid_request', 'path', 'Excessive movie metadata nesting.')
                position = start
                while position < end:
                    remaining -= 1
                    source.seek(position)
                    header = source.read(8)
                    if len(header) != 8 or remaining < 0:
                        fail('invalid_request', 'path', 'Invalid or excessive movie box table.')
                    size, kind = struct.unpack('>I4s', header)
                    header_size = 8
                    if size == 1:
                        extended = source.read(8)
                        if len(extended) != 8:
                            fail('invalid_request', 'path', 'Truncated extended movie box.')
                        size, header_size = struct.unpack('>Q', extended)[0], 16
                    elif size == 0:
                        size = end - position
                    if size < header_size or position + size > end:
                        fail('invalid_request', 'path', 'Invalid movie box bounds.')
                    body, stop = position + header_size, position + size
                    if kind in (b'rmra', b'rmda', b'rdrf'):
                        fail('invalid_request', 'path', 'External QuickTime movie references are prohibited.')
                    if kind in containers:
                        boxes(body, stop, depth + 1)
                    elif kind == b'dref':
                        source.seek(body)
                        table = source.read(8)
                        if len(table) != 8 or body + 8 > stop:
                            fail('invalid_request', 'path', 'Truncated movie data-reference table.')
                        count = struct.unpack('>I', table[4:])[0]
                        cursor = body + 8
                        if count > 10000:
                            fail('invalid_request', 'path', 'Excessive movie data-reference table.')
                        for _ in range(count):
                            source.seek(cursor)
                            entry = source.read(12)
                            if len(entry) != 12 or cursor + 12 > stop:
                                fail('invalid_request', 'path', 'Truncated movie data reference.')
                            length, entry_kind, flags = struct.unpack('>I4sI', entry)
                            if length < 12 or cursor + length > stop or entry_kind != b'url ' or flags != 1:
                                fail('invalid_request', 'path', 'External or unsupported movie data references are prohibited.')
                            cursor += length
                        if cursor != stop:
                            fail('invalid_request', 'path', 'Invalid movie data-reference table length.')
                    position = stop
            boxes(0, Path(path).stat().st_size)

    def media_input_args(self, path, demuxer):
        args = ['-protocol_whitelist', 'file', '-format_whitelist', demuxer, '-f', demuxer]
        if demuxer == 'mov':
            args += ['-enable_drefs', '0', '-use_absolute_path', '0']
        return args

    def count_video_timeline(self, path, demuxer):
        args = [self.media_tool('ffmpeg'), '-v', 'error', '-nostdin',
                *self.media_input_args(path, demuxer), '-i', str(path), '-map', '0:v:0', '-an',
                '-vf', 'fps=24', '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=self.config.get('probe_timeout', 600))
        except (OSError, subprocess.TimeoutExpired) as error:
            fail('invalid_request', 'path', 'Cannot probe the canonical 24 fps video timeline.', reason=str(error))
        counts = re.findall(r'^frame=(\d+)$', result.stdout, re.MULTILINE)
        if result.returncode or not counts:
            fail('invalid_request', 'path', 'Cannot decode the canonical 24 fps video timeline.', diagnostics=result.stderr[-4000:])
        return int(counts[-1])

    def count_audio_samples(self, path, demuxer):
        args = [self.media_tool('ffmpeg'), '-v', 'error', '-nostdin',
                *self.media_input_args(path, demuxer), '-i', str(path), '-map', '0:a:0', '-vn',
                '-af', 'aresample=32000,asetpts=PTS-STARTPTS', '-ac', '2', '-ar', '32000',
                '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=self.config.get('probe_timeout', 600))
        except (OSError, subprocess.TimeoutExpired) as error:
            fail('invalid_request', 'path', 'Cannot probe the native audio sample clock.', reason=str(error))
        times = re.findall(r'^out_time_us=(\d+)$', result.stdout, re.MULTILINE)
        if result.returncode or not times:
            fail('invalid_request', 'path', 'Cannot decode the native audio sample clock.', diagnostics=result.stderr[-4000:])
        # Progress timestamps are rounded to microseconds. At 32 kHz that
        # error is <0.016 samples, so nearest-integer recovers the sample count.
        return round(int(times[-1]) * 32000 / 1000000)

    def probe(self, path):
        native = self.probe_native(path)
        if native is not None:
            return native
        # Detect foreign safetensors and require an explicit native conversion.
        with open(path, 'rb') as f:
            prefix = f.read(8)
            size = struct.unpack('<Q', prefix)[0] if len(prefix) == 8 else 0
            if 2 <= size <= 16 * 1024 * 1024:
                header_bytes = f.read(size)
                try:
                    header = json.loads(header_bytes)
                except (ValueError, UnicodeDecodeError):
                    header = None
                if isinstance(header, dict) and '__metadata__' in header:
                    fail('latent_incompatible', 'path', 'Foreign safetensors require explicit native layout conversion.')
        demuxer = self.media_demuxer(path)
        try:
            p = subprocess.run([self.media_tool('ffprobe'), '-v', 'error',
                                *self.media_input_args(path, demuxer), '-count_frames', '-show_streams',
                                '-show_format', '-of', 'json', str(path)], capture_output=True, text=True,
                               timeout=self.config.get('probe_timeout', 600))
            data = json.loads(p.stdout)
            streams = data['streams']
            if p.returncode or not streams:
                raise ValueError(p.stderr)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as e:
            fail('invalid_request', 'path', 'Asset could not be probed.', reason=str(e))
        video = next((s for s in streams if s['codec_type'] == 'video'), {})
        audio = next((s for s in streams if s['codec_type'] == 'audio'), {})
        rate = video.get('avg_frame_rate', '0/1')
        a, b = rate.split('/')
        fps = float(a) / float(b) if float(b) else 0
        duration = float(data.get('format', {}).get('duration', video.get('duration', audio.get('duration', 0))))
        frames = int(video.get('nb_read_frames', video.get('nb_frames', 0)) or 0)
        frames_24 = self.count_video_timeline(path, demuxer) if video and demuxer not in ('png_pipe', 'jpeg_pipe') else frames
        audio_samples = self.count_audio_samples(path, demuxer) if audio else 0
        audio_duration = audio_samples / 32000
        return dict(kind='media', streams=[k for k, s in [('video', video), ('audio', audio)] if s],
                    width=video.get('width'), height=video.get('height'), frames=frames,
                    fps=fps, duration=duration, audio_rate=audio.get('sample_rate'),
                    audio_channels=audio.get('channels'), audio_duration=audio_duration,
                    frames_24=frames_24, timeline_fps=24, demuxer=demuxer, audio_samples_32k=audio_samples,
                    media_policy='self_contained_v1', protocols=['file'])

    def register(self, request):
        source = Path(request.get('path', ''))
        try:
            mode = source.lstat().st_mode
        except OSError:
            fail('invalid_request', 'path', 'Asset must be a readable regular file.')
        if not stat.S_ISREG(mode):
            fail('invalid_request', 'path', 'Asset must be a regular file.')
        temporary = self.root / 'assets' / (uuid.uuid4().hex + '.tmp')
        try:
            with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as src, open(temporary, 'xb') as out:
                if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                    fail('invalid_request', 'path', 'Asset changed to a nonregular file during registration.')
                shutil.copyfileobj(src, out)
                out.flush()
                os.fsync(out.fileno())
            sha = file_digest(temporary)
            identity = 'a_' + sha
            existing = self.db.execute('SELECT body FROM assets WHERE id=?', (identity,)).fetchone()
            if existing:
                registered = json.loads(existing['body'])
                if registered.get('kind') != 'media' or registered.get('media_policy') == 'self_contained_v1':
                    return registered
            metadata = self.probe(temporary)
            pinned = self.root / 'assets' / sha
            os.replace(temporary, pinned)
            pinned.chmod(0o444)
            directory_fd = os.open(pinned.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            asset = dict(asset_id=identity, sha256=sha, path=str(pinned), **metadata)
            self.db.execute('INSERT OR REPLACE INTO assets VALUES(?,?)', (identity, canonical(asset)))
            return asset
        finally:
            temporary.unlink(missing_ok=True)

    def asset(self, identity, field):
        asset = self.get('assets', identity, field)
        if asset.get('kind') == 'media' and asset.get('media_policy') != 'self_contained_v1':
            fail('invalid_request', field, 'Legacy media metadata lacks immutable self-contained validation; register the pinned file again before preparing a new plan.')
        return asset

    def required(self, native, name, field):
        cap = native.get('capabilities', {}).get(name)
        if not isinstance(cap, dict) or cap.get('availability') != 'available':
            fail('profile_incompatible', field, 'Native capability is unavailable.', capability=name, reported=cap)

    def prepare(self, request):
        task = json.loads(canonical(request.get('task', {})))
        allowed = {'profile', 'prompt', 'frames', 'references', 'guides', 'outputs', 'normalization',
                   'initial_latent', 'mask', 'start_transition', 'operation', 'parent', 'assembly', 'intent', 'execution_mode'}
        for key in task:
            if key not in allowed:
                fail('invalid_request', 'task.' + key, 'Unknown task field.')
        if task.get('operation', 'generate') != 'generate':
            fail('invalid_request', 'task.operation', 'Use the dedicated resume operation for checkpoints.')
        profiles = {p['id']: p for p in self.profiles()}
        profile = profiles.get(task.get('profile'))
        if not profile:
            fail('profile_incompatible', 'task.profile', 'Unknown explicit profile.')
        if not isinstance(task.get('prompt', ''), str):
            fail('invalid_request', 'task.prompt', 'Prompt must be text.')
        if 'intent' in task and task.get('prompt'):
            fail('intent_conflict', 'task.prompt', 'Provide structured intent or freeform prompt, not both.')
        resolved_prompt = lower_intent(task['intent'], task.get('references', [])) if 'intent' in task else task.get('prompt', '')
        outputs = task.get('outputs', {'media': True, 'latent': True})
        if (not isinstance(outputs, dict) or set(outputs) - {'media', 'latent'} or
                any(type(v) is not bool for v in outputs.values()) or not any(outputs.values())):
            fail('invalid_request', 'task.outputs', 'Select media and/or latent boolean outputs.')
        task['outputs'] = outputs
        for key in ('references', 'guides'):
            if not isinstance(task.get(key, []), list) or len(task.get(key, [])) > 64:
                fail('invalid_request', 'task.' + key, 'Expected an ordered list of at most 64 entries.')
        frames = integer(task.get('frames', 124), 'task.frames', 5)
        transforms = []
        approval = task.get('normalization', profile['normalization_policy'])
        if isinstance(approval, str):
            if approval not in NORMALIZATION_POLICIES:
                fail('invalid_request', 'task.normalization', 'Unknown normalization policy.')
            policy = NORMALIZATION_POLICIES[approval]
            approved = policy['transformations']
        elif isinstance(approval, list) and all(isinstance(x, str) for x in approval):
            approved = approval
            policy = {'id': 'explicit_transform_list', 'version': 1, 'transformations': approval}
        else:
            fail('invalid_request', 'task.normalization', 'Use a named policy or an explicit list of transformations.')
        def transform(name, field, before, after):
            item = dict(kind=name, field=field, requested=before, resolved=after)
            if name not in approved:
                fail('normalization_required', field, 'Explicit transformation approval required.', transformation=item,
                     repairs=[{'operation': 'prepare', 'patch': {'normalization': list(approved) + [name]}}])
            transforms.append(item)
            return after
        if (frames - 5) % 17:
            frames = transform('frame_floor', 'task.frames', frames, (frames - 5) // 17 * 17 + 5)
        if frames > profile.get('max_frames', 124) and not profile.get('experimental', False):
            fail('profile_incompatible', 'task.frames', 'Duration exceeds the validated profile range.')
        defaults = {'denoise_reuse': 1, 'core_reuse': 1, 'dit_layers': 50, 'token_reduction': 0, 'ssd_streaming': 0}
        params = dict(defaults, **profile.get('params', {}))
        params['frames'] = frames
        runtime_paths = {'initial_latent_path', 'mask_path', 'resume_path', 'checkpoint_path', 'output_path', 'latent_output_path', 'input_path'}
        if set(params) & runtime_paths:
            fail('profile_incompatible', 'task.profile', 'Runtime media/control paths belong to managed task assets, not profile params.')
        controlled_edit = any(key in task for key in ('initial_latent', 'mask', 'start_transition'))
        mode = task.get('execution_mode', profile['execution_mode'])
        if mode not in ('throughput', 'resumable'):
            fail('invalid_request', 'task.execution_mode', 'Use throughput or resumable.')
        if controlled_edit:
            if task.get('execution_mode') == 'throughput':
                fail('profile_incompatible', 'task.execution_mode', 'Initial-latent, masked and partial edits require controlled execution.',
                     repairs=[{'operation': 'prepare', 'patch': {'execution_mode': 'resumable'}}])
            mode = 'resumable'
        if mode == 'resumable':
            for name, expected in defaults.items():
                if params[name] != expected:
                    fail('profile_incompatible', 'task.profile', 'Resumable/control-dependent execution requires full-layer uncached sampling.', parameter=name, expected=expected, actual=params[name])
        if 'mask' in task and 'initial_latent' not in task:
            fail('invalid_request', 'task.mask', 'A mask requires clean initial audiovisual latents.')
        native = self.native()
        self.required(native, task.get('operation', 'generate'), 'task.operation')
        if mode == 'resumable':
            self.required(native, 'resume', 'task.execution_mode')
        reference_preprocessing, guide_spans = [], []
        total_reference_audio_samples = 0
        references, guides, assets, occupied = [], [], {}, []
        for i, ref in enumerate(task.get('references', [])):
            field = f'task.references[{i}]'
            if set(ref) - {'asset_id', 'role', 'include_embedded_audio', 'audio_asset_id'}:
                fail('invalid_request', field, 'Unknown reference field.')
            a = self.asset(ref.get('asset_id'), field + '.asset_id')
            if a['kind'] != 'media':
                fail('invalid_request', field + '.asset_id', 'References require registered media, not latent/checkpoint tensors.')
            role = ref.get('role')
            if role not in ('image', 'video', 'audio', 'video_audio'):
                fail('invalid_request', field + '.role', 'Unknown reference role.')
            self.required(native, 'references', field)
            needed = {'audio'} if role == 'audio' else {'video', 'audio'} if role == 'video_audio' else {'video'}
            if type(ref.get('include_embedded_audio', False)) is not bool:
                fail('invalid_request', field + '.include_embedded_audio', 'Expected a boolean.')
            if ref.get('include_embedded_audio'):
                if role not in ('video', 'video_audio'):
                    fail('invalid_request', field + '.include_embedded_audio', 'Embedded audio requires a video reference.')
                needed.add('audio')
            replacement = None
            if ref.get('audio_asset_id'):
                if role != 'video_audio' or ref.get('include_embedded_audio'):
                    fail('invalid_request', field + '.audio_asset_id', 'Replacement audio requires video_audio without embedded-audio selection.')
                replacement = self.asset(ref['audio_asset_id'], field + '.audio_asset_id')
                if 'audio' not in replacement['streams']:
                    fail('invalid_request', field + '.audio_asset_id', 'Replacement asset has no audio stream.')
                needed.discard('audio')
                assets[replacement['asset_id']] = replacement
            if role == 'video_audio' and not replacement and ref.get('include_embedded_audio') is False:
                fail('invalid_request', field + '.include_embedded_audio', 'Paired reference needs embedded audio or an explicit replacement asset.')
            if not needed.issubset(a['streams']):
                fail('invalid_request', field, 'Asset lacks required reference streams.')
            references.append(dict(kind='video' if role == 'video_audio' and not replacement else role,
                                   path=a['path'], include_embedded_audio=not replacement if role == 'video_audio' else ref.get('include_embedded_audio', False)))
            if replacement:
                references[-1]['audio_path'] = replacement['path']
            assets[a['asset_id']] = a
            prepared_reference = {'asset_id': a['asset_id'], 'source_frame_clock': 24}
            if role != 'audio':
                canvas = reference_canvas(a, params, 'image' if role == 'image' else 'video')
                if canvas != [a['width'], a['height']]:
                    transform('reference_resize', field, [a['width'], a['height']],
                              {'canvas': canvas, 'fit': 'stretch', 'policy': 'native reference sizing v1'})
                prepared_reference['canvas'] = canvas
                if role == 'image':
                    if a.get('frames', 0) > 1:
                        transform('reference_first_frame', field, a['frames'], 1)
                    prepared_reference['video_frames'] = 1
                else:
                    if a.get('fps') != 24:
                        transform('time_resample', field, a.get('fps'), 24)
                    source_frames = a.get('frames_24')
                    if type(source_frames) is not int or source_frames < 5:
                        fail('invalid_request', field, 'Reference video requires at least five frames on its probed 24 fps timeline.')
                    selected = min(source_frames, frames)
                    if selected < source_frames:
                        transform('reference_trim', field, {'source_frames_24': source_frames}, {'start_frame': 0, 'frames': selected})
                    cadence = (selected - 5) // 17 * 17 + 5
                    if cadence != selected:
                        transform('reference_frame_floor', field, selected, cadence)
                    prepared_reference['video_frames'] = cadence
                    prepared_reference['video_selection'] = {'start_frame': 0, 'frames': cadence, 'fps': 24}
            audio_bearing = role in ('audio', 'video_audio') or ref.get('include_embedded_audio', False)
            if audio_bearing:
                audio_asset = replacement or a
                audio_field = field + '.audio_asset_id' if replacement else field
                samples = audio_asset.get('audio_samples_32k')
                if type(samples) is not int or samples < 64000:
                    fail('invalid_request', audio_field, 'Reference audio requires at least two decoded seconds; guides have a separate, shorter-audio policy.')
                paired = role != 'audio'
                if paired:
                    limit = math.floor(prepared_reference['video_frames'] * 32000 / 24 + 0.5)
                    if samples > limit:
                        samples = transform('reference_audio_trim', audio_field, samples, limit)
                elif samples > 480000:
                    fail('invalid_request', audio_field, 'Standalone reference audio exceeds the native 15-second limit; register an explicitly selected source.')
                if samples < 64000:
                    fail('invalid_request', audio_field, 'The selected video soundtrack is shorter than the native two-second reference minimum; increase target/reference span or use an audio guide.')
                if str(audio_asset.get('audio_rate')) != '32000' or audio_asset.get('audio_channels') != 2:
                    transform('audio_resample', audio_field, [audio_asset.get('audio_rate'), audio_asset.get('audio_channels')], [32000, 2])
                padded_samples = (samples + 799) // 800 * 800
                if padded_samples != samples:
                    transform('audio_right_pad', audio_field, samples, padded_samples)
                total_reference_audio_samples += samples
                if total_reference_audio_samples > 480000:
                    fail('invalid_request', 'task.references', 'Ordered reference audio exceeds 15 decoded seconds in total.')
                prepared_reference['audio'] = {'asset_id': audio_asset['asset_id'], 'samples_32k': samples,
                                                'padded_samples_32k': padded_samples, 'channels': 2}
            reference_preprocessing.append(prepared_reference)
        if profile.get('target') not in ('fl2va', 'ref2va') or (references and profile.get('target') != 'ref2va'):
            fail('profile_incompatible', 'task.profile', 'References require an explicit Ref2VA target; target selection is never silently changed.')
        for i, guide in enumerate(task.get('guides', [])):
            field = f'task.guides[{i}]'
            if set(guide) - {'asset_id', 'kind', 'frame', 'source_start_frame', 'source_frames', 'audio_asset_id'}:
                fail('invalid_request', field, 'Unknown guide field.')
            a = self.asset(guide.get('asset_id'), field + '.asset_id')
            if a['kind'] != 'media':
                fail('invalid_request', field + '.asset_id', 'Guides require registered media, not latent/checkpoint tensors.')
            kind = guide.get('kind')
            if kind not in ('image', 'clip', 'audio', 'clip_audio'):
                fail('invalid_request', field + '.kind', 'Unknown guide kind.')
            self.required(native, kind + '_guide', field)
            start = guide.get('frame', 0)
            if type(start) is not int:
                fail('invalid_request', field + '.frame', 'Placement must be an integer pixel frame.')
            if start < 0:
                start += frames
            if start < 0 or start >= frames:
                fail('guide_out_of_range', field + '.frame', 'Guide begins outside the resolved target.', target_frames=frames)
            source_start = integer(guide.get('source_start_frame', 0), field + '.source_start_frame')
            if kind == 'image':
                if source_start != 0:
                    fail('invalid_request', field + '.source_start_frame', 'Image guides require source_start_frame zero.')
                image_frames = integer(guide.get('source_frames', 1), field + '.source_frames', 1)
                if image_frames != 1:
                    fail('invalid_request', field + '.source_frames', 'Image guides require exactly one source frame.')
            length = 1
            streams = ['audio'] if kind == 'audio' else ['video', 'audio'] if kind == 'clip_audio' else ['video']
            audio_asset = a
            if guide.get('audio_asset_id'):
                if kind != 'clip_audio':
                    fail('invalid_request', field + '.audio_asset_id', 'Replacement audio requires clip_audio.')
                audio_asset = self.asset(guide['audio_asset_id'], field + '.audio_asset_id')
                assets[audio_asset['asset_id']] = audio_asset
                if 'audio' not in audio_asset['streams'] or 'video' not in a['streams']:
                    fail('invalid_request', field, 'Paired audiovisual guide lacks a required stream.')
            if not set(streams).issubset(set(a['streams']) | ({'audio'} if guide.get('audio_asset_id') else set())):
                fail('invalid_request', field, 'Asset lacks required guide streams.')
            if kind in ('clip', 'clip_audio'):
                source_frames = a.get('frames_24')
                if type(source_frames) is not int or source_frames < 1:
                    fail('invalid_request', field + '.asset_id', 'Clip source lacks a probed 24 fps timeline.')
                length = integer(guide.get('source_frames', source_frames - source_start), field + '.source_frames', 1)
                if source_start + length > source_frames:
                    fail('invalid_request', field + '.source_frames', 'Clip selection exceeds the post-resampling 24 fps source timeline.', source_frames_24=source_frames)
                if length < 5:
                    fail('normalization_required', field + '.kind', 'Short clips require explicit image interpretation.')
                if (length - 5) % 17:
                    length = transform('clip_floor', field + '.source_frames', length, (length - 5) // 17 * 17 + 5)
            if kind == 'audio':
                duration = a.get('audio_duration', 0) - source_start / 24
                if duration <= 0:
                    fail('invalid_request', field, 'Audio guide selection is empty.')
                length = integer(guide.get('source_frames', max(1, math.ceil(duration * 24))), field + '.source_frames', 1)
            if kind != 'image' and start + length > frames:
                length = transform('overrun_trim', field, length, frames - start)
                if kind in ('clip', 'clip_audio') and (length < 5 or (length - 5) % 17):
                    fail('normalization_required', field, 'Trimmed clip must retain a 17k+5 length.')
            if kind != 'audio':
                if a.get('fps') not in (0, 24, None) and kind != 'image':
                    transform('time_resample', field, a.get('fps'), 24)
                if (a.get('width'), a.get('height')) != (params.get('width'), params.get('height')):
                    transform('cover_crop', field, [a.get('width'), a.get('height')], [params.get('width'), params.get('height')])
            if 'audio' in streams and (str(audio_asset.get('audio_rate')) != '32000' or audio_asset.get('audio_channels') != 2):
                transform('audio_resample', field, [audio_asset.get('audio_rate'), audio_asset.get('audio_channels')], [32000, 2])
            spans = {stream: length for stream in streams}
            if 'audio' in streams:
                skip_samples = (source_start * 4000 + 1) // 3
                requested_samples = (length * 4000 + 1) // 3
                available_samples = audio_asset.get('audio_samples_32k', 0) - skip_samples
                selected_samples = min(available_samples, requested_samples)
                remaining_ticks = (3 * ((frames * 5 + 1) // 3) - 5 * start) // 3
                native_samples = min(selected_samples, remaining_ticks * 800)
                if native_samples <= 0:
                    fail('invalid_request', field, 'Audio guide selection is empty on the native sample clock.')
                if (source_start * 4000) % 3 or (length * 4000) % 3 or native_samples != selected_samples:
                    transform('time_resample', field + '.audio_clock',
                              {'start_frame': source_start, 'source_frames': length, 'selected_samples_32k': selected_samples},
                              {'skip_samples_32k': skip_samples, 'selected_samples_32k': native_samples,
                               'remaining_40hz_ticks': remaining_ticks, 'policy': 'native frame/sample/40Hz boundary rounding'})
                padded_samples = (native_samples + 799) // 800 * 800
                if padded_samples != native_samples:
                    transform('audio_right_pad', field, native_samples, padded_samples)
                spans['audio'] = padded_samples / 32000 * 24
            for prior_stream, lo, hi, prior in occupied:
                if prior_stream in streams and start < hi and lo < start + spans[prior_stream]:
                    fail('guide_overlap', field, 'Overlapping guides on the same stream.', previous=prior, stream=prior_stream)
            occupied.extend((stream, start, start + spans[stream], i) for stream in streams)
            guides.append(dict(kind=kind, path=a['path'], frame_index=start,
                               source_start_frame=source_start, source_frames=length,
                               include_embedded_audio=kind == 'clip_audio'))
            if guide.get('audio_asset_id'):
                guides[-1]['audio_path'] = audio_asset['path']
                guides[-1]['include_embedded_audio'] = False
            assets[a['asset_id']] = a
            guide_spans.append({'asset_id': a['asset_id'], 'source_start_frame': source_start,
                                'source_frames': length, 'source_frame_clock': 24, 'target_frame': start,
                                'target_stream_spans': spans})
        encode_source = None
        for key, native_key, capability in [('initial_latent', 'initial_latent_path', 'initial_latent'),
                                             ('mask', 'mask_path', 'masks'), ('checkpoint', 'resume_path', 'resume')]:
            if key in task:
                self.required(native, capability, 'task.' + key)
                a = self.asset(task[key], 'task.' + key)
                if a['kind'] != 'latent':
                    if key != 'initial_latent':
                        fail('latent_incompatible', 'task.' + key, 'Expected native latent asset.')
                    self.required(native, 'encode', 'task.initial_latent')
                    if set(a['streams']) != {'video', 'audio'}:
                        fail('latent_incompatible', 'task.initial_latent', 'Media revision requires both real video and audio streams; explicit silence muxing is a separate requested operation.')
                    if a.get('frames') != frames or a.get('fps') != 24:
                        fail('latent_incompatible', 'task.initial_latent', 'Media revision requires source frame count and 24 fps to match the resolved target; select/transcode the source explicitly first.')
                    transform('encode_source', 'task.initial_latent', a['asset_id'],
                              {'format': 'H3NATIVE-v1-type5', 'temporal_mapping': 'encoder-linear-endpoints',
                               'source_frames': frames, 'video_latent_frames': ((frames - 5) // 17) * 5 + 2,
                               'audio_latent_frames': (frames * 5 + 1) // 3,
                               'audio_policy': '32k stereo; right-pad then discard at most one encoder-only excess tick'})
                    encode_source = a
                elif key == 'initial_latent' and a.get('metadata', {}).get('kind') != 'clean':
                    fail('latent_incompatible', 'task.initial_latent', 'Revision requires clean latents, not solver state.')
                if key == 'initial_latent' and a['kind'] == 'latent':
                    if set(a['streams']) != {'video', 'audio'}:
                        fail('latent_incompatible', 'task.initial_latent', 'Generation initialization requires both native video and audio streams.')
                    metadata = a['metadata']
                    expected_audio = (frames * 5 + 1) // 3
                    if metadata.get('audio_frames') != expected_audio:
                        fail('latent_incompatible', 'task.initial_latent', 'Clean audio latent length differs from the resolved generation target.', expected=expected_audio, actual=metadata.get('audio_frames'))
                    actual = [metadata.get('video_frames'), metadata.get('latent_height'), metadata.get('latent_width')]
                    expected = [((frames - 5) // 17) * 5 + 2, params.get('height', 0) // 16, params.get('width', 0) // 16]
                    if 'video' in a['streams'] and actual != expected:
                        fail('latent_incompatible', 'task.initial_latent', 'Clean video latent shape differs from the resolved generation target.', expected=expected, actual=actual)
                if key == 'mask':
                    bounds = a.get('metadata', {}).get('value_range')
                    if not bounds or bounds[0] < 0 or bounds[1] > 1:
                        fail('latent_incompatible', 'task.mask', 'Masks require finite values in [0,1]; 0 preserves, 1 edits.')
                params[native_key] = a['path']
                assets[a['asset_id']] = a
        if 'start_transition' in task:
            start = integer(task['start_transition'], 'task.start_transition')
            if start > params.get('steps', 0) or 'initial_latent' not in task:
                fail('invalid_request', 'task.start_transition', 'Partial denoise requires clean latents and a valid transition.')
            params['start_transition'] = start
        manifest = dict(schema_version=1, operation=task.get('operation', 'generate'),
                        model_dir=profile.get('model_dir'), target=profile.get('target'),
                        prompt=resolved_prompt, params=params, references=references, guides=guides)
        if profile.get('loras'):
            manifest['loras'] = profile['loras']
            pinned_paths = {d['path'] for d in profile.get('dependencies', [])}
            if any(lora['path'] not in pinned_paths for lora in profile['loras']):
                fail('profile_incompatible', 'task.profile', 'Every adapter requires a pinned dependency digest.')
        resolved = dict(manifest=manifest, profile=profile, assets=list(assets.values()),
                        native=native, assembly=task.get('assembly'),
                        execution_digest=digest({'manifest': manifest, 'mode': mode, 'profile': profile,
                                                 'assets': list(assets.values()), 'native': native,
                                                 'policy': policy, 'transformations': transforms}),
                        encode_source=encode_source, normalization_policy=policy,
                        reference_preprocessing=reference_preprocessing, guide_spans=guide_spans,
                        execution_mode=mode, execution_reason='controlled_edit' if controlled_edit else 'requested_or_profile_default',
                        controls={'pause': mode == 'resumable', 'resume': mode == 'resumable',
                                  'cancel': 'transition_boundary' if mode == 'resumable' else 'process_abort_without_checkpoint'})
        plan = dict(schema_version=1, intent=task, resolved=resolved, transformations=transforms,
                    warnings=['Guide adherence and decoded pixel identity are not guaranteed.'] +
                    ['Reference/voice bindings request conditioning; they do not prove identity, adherence, lip sync or bit-exact copying.'] +
                    ([] if 'intent' in task else ['Freeform prompt is an explicit unvalidated mode; no ContextIR was run.']) +
                    (['Duration is outside the official 4–15 second prompt range; this is experimental/smoke evidence only.'] if not 96 <= frames <= 360 else []),
                    quality_evidence=profile.get('quality_evidence', 'unverified'),
                    submittable=profile['availability'] == 'available',
                    summary=dict(frames=frames, fps=24, profile=profile['id'],
                                 operation=manifest['operation'], missing=profile['missing']))
        sha = digest(plan)
        plan.update(plan_id='p_' + sha, plan_digest=sha)
        atomic(self.root / 'plans' / (sha + '.json'), canonical(plan))
        self.db.execute('INSERT OR IGNORE INTO plans VALUES(?,?,?)', (plan['plan_id'], sha, canonical(plan)))
        return plan

    def dependencies(self, plan):
        for a in plan['resolved']['assets']:
            if a.get('kind') == 'media' and a.get('media_policy') != 'self_contained_v1':
                fail('dependency_changed', 'plan.assets', 'Legacy media plan lacks self-contained media validation; register assets and prepare a new plan.')
            if not Path(a['path']).is_file() or self.sha(a['path']) != a['sha256']:
                fail('dependency_changed', 'plan.assets', 'Pinned asset changed.', asset_id=a['asset_id'])
        for dep in plan['resolved']['profile'].get('dependencies', []):
            if not Path(dep['path']).is_file() or self.sha(dep['path']) != dep['sha256']:
                fail('dependency_changed', 'plan.profile', 'Pinned model dependency changed.', role=dep.get('role'))
        if self.native() != plan['resolved']['native']:
            fail('dependency_changed', 'plan.native', 'Native execution identity changed.')

    def submit(self, request):
        if set(request) - {'operation', 'plan_id', 'plan_digest', 'idempotency_key'}:
            fail('invalid_request', '', 'Submit accepts only plan identity and idempotency key.')
        key = request.get('idempotency_key')
        if not isinstance(key, str) or not key or len(key) > 256:
            fail('invalid_request', 'idempotency_key', 'A bounded nonempty idempotency key is required.')
        plan = self.get('plans', request.get('plan_id'), 'plan_id')
        if request.get('plan_digest') != plan['plan_digest']:
            fail('invalid_request', 'plan_digest', 'Plan digest mismatch.')
        with self.transaction():
            old = self.db.execute('SELECT * FROM idempotency WHERE key=?', (key,)).fetchone()
            if old:
                if old['plan'] != plan['plan_id']:
                    fail('idempotency_conflict', 'idempotency_key', 'Key already belongs to another plan.')
                return self.get('jobs', old['job'], 'job_id')
            if not plan['submittable']:
                fail('missing_model', 'plan_id', 'Plan has unavailable model prerequisites.', missing=plan['summary']['missing'])
            self.dependencies(plan)
            job = dict(job_id='j_' + uuid.uuid4().hex, plan_id=plan['plan_id'], state='queued', outputs={}, checkpoint_id=None,
                       execution_mode=plan['resolved']['execution_mode'], controls=plan['resolved']['controls'])
            self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,NULL)', (job['job_id'], plan['plan_id'], 'queued', canonical(job)))
            self.db.execute('INSERT INTO idempotency VALUES(?,?,?)', (key, plan['plan_id'], job['job_id']))
            self.event(job['job_id'], dict(type='queued'))
            return job

    def event(self, job, body):
        self.db.execute('INSERT INTO events(job,body) VALUES(?,?)', (job, canonical(body)))

    def update(self, job, state, **changes):
        job.update(changes, state=state)
        self.db.execute('UPDATE jobs SET state=?,body=? WHERE id=?', (state, canonical(job), job['job_id']))
        self.event(job['job_id'], dict(type=state, **changes))

    def inspect(self, request):
        if request.get('plan_id'):
            return self.get('plans', request['plan_id'], 'plan_id')
        job = self.get('jobs', request.get('job_id'), 'job_id')
        cursor = integer(request.get('cursor', 0), 'cursor')
        limit = integer(request.get('limit', 50), 'limit', 1)
        if limit > 200:
            fail('invalid_request', 'limit', 'Event page limit is 200.')
        rows = self.db.execute('SELECT seq,body FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT ?',
                               (job['job_id'], cursor, limit)).fetchall()
        requeue = None
        if job['state'] == 'paused' and job.get('never_started'):
            plan = self.get('plans', job['plan_id'], 'job_id')
            requeue = dict(operation='submit', plan_id=plan['plan_id'], plan_digest=plan['plan_digest'],
                           requires=['new idempotency_key'], reason='Never started; no solver checkpoint exists.')
        return dict(job, events=[dict(seq=r['seq'], **json.loads(r['body'])) for r in rows],
                    cursor=rows[-1]['seq'] if rows else cursor,
                    requeue=requeue,
                    next_operations=['inspect'] + (['resume'] if job.get('controls', {}).get('resume') and job['checkpoint_id'] and job['state'] in ('paused', 'interrupted', 'cancelled') else []) +
                    (['revise'] if job.get('outputs') and job['state'] == 'succeeded' else []) +
                    (['continue'] if job.get('outputs', {}).get('media') and job['state'] == 'succeeded' else []) +
                    (['submit'] if requeue else []) +
                    (['cancel'] if job['state'] in ('queued', 'running', 'pausing') else []) +
                    (['pause'] if job.get('controls', {}).get('pause') and job['state'] in ('queued', 'running', 'pausing') else []))

    def control(self, request, operation):
        with self.transaction():
            job = self.get('jobs', request.get('job_id'), 'job_id')
            state = job['state']
            if operation == 'pause' and not job.get('controls', {}).get('pause'):
                fail('profile_incompatible', 'job_id', 'Throughput jobs cannot pause or produce exact-resume checkpoints.',
                     repairs=[{'operation': 'prepare', 'patch': {'execution_mode': 'resumable'}}])
            if operation == 'cancel' and request.get('keep_checkpoint') and job.get('execution_mode') == 'throughput':
                fail('profile_incompatible', 'keep_checkpoint', 'Throughput cancellation aborts the process without a checkpoint.',
                     repairs=[{'operation': 'prepare', 'patch': {'execution_mode': 'resumable'}}])
            if state == 'paused' and operation == 'cancel':
                keep = request.get('keep_checkpoint', False)
                if type(keep) is not bool:
                    fail('invalid_request', 'keep_checkpoint', 'Expected a boolean.')
                self.update(job, 'cancelled', checkpoint_id=job.get('checkpoint_id') if keep else None, keep_checkpoint=keep)
                return job
            if state in TERMINAL or state == 'cancelling' or (state == 'pausing' and operation == 'pause'):
                return job
            if operation == 'cancel':
                job['cancel_mechanism'] = 'queue_removal' if state == 'queued' else ('process_abort' if job.get('execution_mode') == 'throughput' else 'transition_boundary')
                if type(request.get('keep_checkpoint', False)) is not bool:
                    fail('invalid_request', 'keep_checkpoint', 'Expected an explicit boolean.')
                job['keep_checkpoint'] = request.get('keep_checkpoint', False)
            if state == 'queued':
                self.update(job, 'cancelled' if operation == 'cancel' else 'paused', never_started=True)
            else:
                directory = self.root / 'jobs' / job['job_id']
                directory.mkdir(exist_ok=True)
                atomic(directory / 'control', canonical({'action': operation}))
                self.update(job, 'pausing' if operation == 'pause' else 'cancelling')
            return job

    def source(self, request):
        if request.get('asset_id'):
            return self.asset(request['asset_id'], 'asset_id'), None
        job = self.get('jobs', request.get('job_id'), 'job_id')
        identity = job.get('outputs', {}).get('media')
        if job['state'] != 'succeeded' or not identity:
            fail('invalid_request', 'job_id', 'Source job has no published media.')
        return self.asset(identity, 'job_id'), job

    def continue_task(self, request):
        source, parent = self.source(request)
        tail = integer(request.get('tail_frames', 22), 'tail_frames', 5)
        additional = integer(request.get('additional_frames'), 'additional_frames', 1)
        if additional % 17:
            fail('normalization_required', 'additional_frames', 'Additional frames must be a positive multiple of 17 so overlap+suffix stays on the native lattice.')
        if tail > source.get('frames', 0) or (tail - 5) % 17:
            fail('invalid_request', 'tail_frames', 'Tail must fit the source and use 17k+5 frames.')
        mode = request.get('audio_mode')
        join = request.get('join_mode')
        if mode not in ('condition', 'silent', 'discard') or join not in ('new_segment', 'concatenated'):
            fail('invalid_request', 'audio_mode', 'Explicit audio and join policies are required.')
        if source.get('fps') != 24:
            fail('normalization_required', 'asset_id', 'Continuation source must be explicitly registered at 24 fps.')
        task = dict(request.get('task', {}))
        profile = self.config.get('profiles', {}).get(task.get('profile'), {})
        if join == 'concatenated' and (source.get('width'), source.get('height')) != (profile.get('params', {}).get('width'), profile.get('params', {}).get('height')):
            fail('profile_incompatible', 'task.profile', 'Concatenation requires the profile canvas to match the retained source prefix.')
        if 'guides' in task or 'frames' in task:
            fail('invalid_request', 'task', 'Continuation owns tail guides and target length.')
        task.update(frames=tail + additional, parent=parent['job_id'] if parent else source['asset_id'],
                    guides=[dict(asset_id=source['asset_id'], kind='clip_audio' if mode == 'condition' else 'clip',
                                 frame=0, source_start_frame=source['frames'] - tail, source_frames=tail)],
                    assembly=dict(source_asset=source['asset_id'], source_frames=source['frames'],
                                  overlap_frames=tail, added_frames=additional, join_mode=join, audio_mode=mode,
                                  video_codec='libx264', audio_codec='aac' if mode != 'discard' else None,
                                  audio_rate=32000, audio_channels=2,
                                  audio_padding='right-pad to declared segment duration; silence where no source audio exists',
                                  final_frames=source['frames'] + additional if join == 'concatenated' else tail + additional,
                                  join_timestamp=source['frames'] / 24,
                                  policy='explicit_reencode_no_crossfade', discarded_generated_overlap=tail if join == 'concatenated' else 0))
        if request.get('assembly_policy') != 'explicit_reencode_no_crossfade':
            fail('normalization_required', 'assembly_policy', 'Explicit re-encoding assembly approval is required.')
        return self.prepare({'task': task})

    def revise(self, request):
        if request.get('asset_id'):
            if request.get('job_id'):
                fail('invalid_request', 'asset_id', 'Choose one revision source.')
            source = self.asset(request['asset_id'], 'asset_id')
            latent, parent = source['asset_id'], source['asset_id']
            baseline = request.get('task')
            if not isinstance(baseline, dict) or 'profile' not in baseline or 'frames' not in baseline:
                fail('invalid_request', 'task', 'Registered-asset revision requires explicit profile and target frames.')
        else:
            job = self.get('jobs', request.get('job_id'), 'job_id')
            old = self.get('plans', job['plan_id'], 'job_id')
            latent = job.get('outputs', {}).get('latent') or job.get('outputs', {}).get('media')
            if not latent:
                fail('latent_incompatible', 'job_id', 'Revision requires published clean latents or media to encode.')
            baseline, parent = old['intent'], job['job_id']
            source = self.asset(latent, 'job_id')
        patch = request.get('patch', {})
        if set(patch) - {'prompt', 'intent', 'references', 'guides', 'mask', 'start_transition', 'normalization', 'execution_mode'}:
            fail('invalid_request', 'patch', 'Revision patch changes an immutable profile or shape.')
        task = dict(baseline)
        if source['kind'] == 'media':
            task['frames'] = source['frames']
            source_canvas = [source['width'], source['height']]
            profile_params = self.config.get('profiles', {}).get(task.get('profile'), {}).get('params', {})
            if source_canvas != [profile_params.get('width'), profile_params.get('height')]:
                fail('profile_incompatible', 'task.profile', 'Revision profile canvas must match the selected published media.', source_canvas=source_canvas)
        task['execution_mode'] = patch.get('execution_mode', 'resumable')
        if 'prompt' in patch:
            task.pop('intent', None)
        if 'intent' in patch:
            task.pop('prompt', None)
        for key in ('checkpoint', 'assembly'):
            task.pop(key, None)
        task.update(patch, operation='generate', initial_latent=latent, parent=parent)
        task.setdefault('start_transition', 0)
        return self.prepare({'task': task})

    def resume(self, request):
        if set(request) - {'operation', 'job_id'}:
            fail('checkpoint_incompatible', '', 'Exact resume forbids intent or profile changes.')
        job = self.get('jobs', request.get('job_id'), 'job_id')
        if not job.get('controls', {}).get('resume'):
            fail('profile_incompatible', 'job_id', 'Throughput execution has no exact-resume state.',
                 repairs=[{'operation': 'prepare', 'patch': {'execution_mode': 'resumable'}}])
        if job['state'] not in ('paused', 'interrupted', 'cancelled') or not job.get('checkpoint_id'):
            fail('checkpoint_incompatible', 'job_id', 'No durable resumable checkpoint.')
        old = self.get('plans', job['plan_id'], 'job_id')
        self.dependencies(old)
        self.required(old['resolved']['native'], 'resume', 'job_id')
        plan = json.loads(canonical(old))
        checkpoint = self.asset(job['checkpoint_id'], 'job_id')
        plan['resolved']['assets'].append(checkpoint)
        plan['resolved']['manifest']['operation'] = 'resume'
        plan['resolved']['manifest']['params']['resume_path'] = checkpoint['path']
        for key in ('initial_latent_path', 'mask_path'):
            plan['resolved']['manifest']['params'].pop(key, None)
        plan['resolved']['encode_source'] = None
        plan['intent'] = dict(old['intent'], operation='resume', parent=job['job_id'])
        plan['summary']['operation'] = 'resume'
        for key in ('plan_id', 'plan_digest'):
            plan.pop(key, None)
        sha = digest(plan)
        plan.update(plan_id='p_' + sha, plan_digest=sha)
        atomic(self.root / 'plans' / (sha + '.json'), canonical(plan))
        self.db.execute('INSERT OR IGNORE INTO plans VALUES(?,?,?)', (plan['plan_id'], sha, canonical(plan)))
        return plan

    def assemble(self, job, plan, generated, directory):
        assembly = plan['resolved'].get('assembly')
        if not assembly:
            return generated
        source = self.asset(assembly['source_asset'], 'assembly.source_asset')
        ffmpeg = self.media_tool('ffmpeg')
        tail = assembly['overlap_frames']
        audio = assembly['audio_mode']
        output = directory / 'assembled.mp4'
        args = [ffmpeg, '-v', 'error', '-y']
        if assembly['join_mode'] == 'concatenated':
            args += ['-i', source['path'], '-i', str(generated)]
            video = f'[0:v]trim=end_frame={assembly["source_frames"]},setpts=PTS-STARTPTS[v0];[1:v]trim=start_frame={tail},setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[v]'
            if audio != 'discard':
                prefix = (f'[0:a]aresample=32000,aformat=channel_layouts=stereo,apad,atrim=end={assembly["source_frames"]/24},asetpts=PTS-STARTPTS[a0]' if 'audio' in source['streams'] else
                          f'anullsrc=r=32000:cl=stereo,atrim=duration={assembly["source_frames"]/24}[a0]')
                suffix = (f'[1:a]aresample=32000,aformat=channel_layouts=stereo,apad,atrim=start={tail/24}:duration={assembly["added_frames"]/24},asetpts=PTS-STARTPTS[a1]' if audio == 'condition' else
                          f'anullsrc=r=32000:cl=stereo,atrim=duration={assembly["added_frames"]/24}[a1]')
                video += f';{prefix};{suffix};[a0][a1]concat=n=2:v=0:a=1[a]'
            args += ['-filter_complex', video, '-map', '[v]']
            if audio != 'discard':
                args += ['-map', '[a]']
        else:
            args += ['-i', str(generated)]
            if audio == 'silent':
                args += ['-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=stereo']
            args += ['-map', '0:v:0']
            if audio == 'condition':
                args += ['-map', '0:a:0']
            elif audio == 'silent':
                args += ['-map', '1:a:0', '-t', str(plan['summary']['frames'] / 24)]
        args += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        if audio != 'discard':
            args += ['-c:a', 'aac']
        args += [str(output)]
        p = subprocess.run(args, capture_output=True, text=True)
        if p.returncode:
            fail('output_failed', 'assembly', 'Explicit media assembly failed.', diagnostics=p.stderr[-4000:])
        return output

    def execute_native(self, job, path, directory, resumable):
        """Consume bounded JSONL while checking abort requests even in quiet phases."""
        final, pending, aborted_at = None, b'', None
        def observe(line):
            nonlocal final
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError('Native event must be an object')
                event['type'] = event.get('event', event.get('type'))
                if event.get('schema_version') != 1 or event.get('type') not in ('progress', 'result', 'error', 'paused', 'cancelled'):
                    raise ValueError('Invalid native event schema')
            except (ValueError, UnicodeDecodeError):
                fail('engine_failed', 'engine.events', 'Native stdout is not a versioned JSONL event stream.')
            self.event(job['job_id'], event)
            if event['type'] == 'progress':
                with self.transaction():
                    current = self.get('jobs', job['job_id'], 'job_id')
                    current['last_phase'] = event.get('phase')
                    current['last_transition'] = event.get('completed', event.get('transition'))
                    self.db.execute('UPDATE jobs SET body=? WHERE id=?', (canonical(current), job['job_id']))
            else:
                final = event
        with open(directory / 'diagnostics.log', 'w') as diagnostics:
            process = subprocess.Popen([self.config['engine'], '--manifest', str(path)], stdout=subprocess.PIPE,
                stderr=diagnostics, pass_fds=(self.worker_lock,), start_new_session=True, env=self.native_environment())
            selector = selectors.DefaultSelector()
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            try:
                while selector.get_map():
                    if not resumable:
                        control = directory / 'control'
                        if aborted_at is None and control.is_file():
                            action = json.loads(control.read_text()).get('action')
                            if action == 'cancel':
                                aborted_at = time.monotonic()
                                if process.poll() is None:
                                    try:
                                        os.killpg(process.pid, signal.SIGTERM)
                                    except ProcessLookupError:
                                        pass
                            elif action != 'run':
                                fail('engine_failed', 'control', 'Throughput controls accept only run or explicit cancel/process_abort.')
                        if aborted_at is not None and time.monotonic() - aborted_at > 5 and process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    for key, _ in selector.select(0.1):
                        try:
                            chunk = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(key.fileobj)
                            break
                        if aborted_at is not None:
                            continue  # Partial lines after process termination are not native boundary events.
                        pending += chunk
                        while b'\n' in pending:
                            line, pending = pending.split(b'\n', 1)
                            if len(line) > 1024 * 1024:
                                fail('engine_failed', 'engine.events', 'Native event exceeds the one-MiB event bound.')
                            observe(line)
                        if len(pending) > 1024 * 1024:
                            fail('engine_failed', 'engine.events', 'Native event exceeds the one-MiB event bound.')
                code = process.wait()
                if aborted_at is not None:
                    raise ProcessAborted(code)
                if pending:
                    observe(pending)
                return code, final
            finally:
                selector.close()
                process.stdout.close()
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()

    def worker(self, request):
        if type(request.get('once', False)) is not bool:
            fail('invalid_request', 'once', 'Expected a boolean.')
        device = self.config.get('device')
        if not isinstance(device, str) or not device:
            fail('invalid_request', 'config.device', 'Explicit device identity is required.')
        lock_path = Path(tempfile.gettempdir()) / ('h3-task-device-' + str(os.getuid()) + '-' + digest(device) + '.lock')
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fail('resource_busy', 'device', 'Another worker or surviving native process owns this device.')
            self.worker_lock = descriptor
            return self.run_worker(request)
        finally:
            # Closing, rather than explicit unlock, keeps the inherited native
            # descriptor locked if this worker dies while sampling is alive.
            os.close(descriptor)

    def run_worker(self, request):
        device = self.config.get('device')
        if not device:
            fail('invalid_request', 'config.device', 'Explicit device identity is required.')
        owner = uuid.uuid4().hex
        with self.transaction():
            lease = self.db.execute('SELECT * FROM leases WHERE device=?', (device,)).fetchone()
            if lease:
                try:
                    os.kill(lease['pid'], 0)
                except ProcessLookupError:
                    self.db.execute('DELETE FROM leases WHERE device=?', (device,))
                else:
                    fail('resource_busy', 'device', 'A device worker already owns the lease.')
            # The inherited OS device lock proves no previous native process
            # remains alive, including workers that removed their SQLite lease
            # while unwinding an interruption.
            for row in self.db.execute("SELECT body FROM jobs WHERE state IN ('running','pausing','cancelling')").fetchall():
                orphan = json.loads(row['body'])
                checkpoint = self.root / 'jobs' / orphan['job_id'] / 'checkpoint.h3latent'
                if orphan['state'] == 'cancelling' and not orphan.get('keep_checkpoint'):
                    checkpoint.unlink(missing_ok=True)
                if checkpoint.is_file():
                    try:
                        orphan['checkpoint_id'] = self.register({'path': str(checkpoint)})['asset_id']
                    except TaskError as e:
                        orphan['checkpoint_failure'] = e.value
                self.update(orphan, 'interrupted', failure={'code': 'engine_failed', 'message': 'Worker exited; no automatic replay.'})
            self.db.execute('INSERT INTO leases VALUES(?,?,?)', (device, owner, os.getpid()))
        processed = []
        try:
            while True:
                with self.transaction():
                    row = self.db.execute("SELECT body FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
                    if not row:
                        break
                    job = json.loads(row['body'])
                    self.db.execute('UPDATE jobs SET worker=? WHERE id=?', (owner, job['job_id']))
                    self.update(job, 'running')
                directory = self.root / 'jobs' / job['job_id']
                directory.mkdir(exist_ok=True)
                try:
                    plan = self.get('plans', job['plan_id'], 'plan_id')
                    self.dependencies(plan)
                    manifest = json.loads(canonical(plan['resolved']['manifest']))
                    encode_source = plan['resolved'].get('encode_source')
                    if encode_source:
                        encoded_path = directory / 'encoded.h3latent'
                        encode_manifest = dict(schema_version=1, operation='encode', model_dir=manifest['model_dir'],
                            target=manifest['target'], request_digest=plan['resolved']['execution_digest'],
                            params=dict(input_path=encode_source['path'], stream_policy=3,
                                        width=manifest['params']['width'], height=manifest['params']['height'],
                                        frames=manifest['params']['frames'], latent_output_path=str(encoded_path)))
                        if 'vae_precision' in manifest['params']:
                            encode_manifest['params']['vae_precision'] = manifest['params']['vae_precision']
                        encode_path = directory / 'encode-manifest.json'
                        atomic(encode_path, canonical(encode_manifest))
                        with open(directory / 'encode-diagnostics.log', 'w') as diagnostics:
                            encoded = subprocess.run([self.config['engine'], '--manifest', str(encode_path)],
                                stdout=subprocess.PIPE, stderr=diagnostics, text=True, pass_fds=(self.worker_lock,), env=self.native_environment())
                        terminal = None
                        for line in encoded.stdout.splitlines():
                            event = json.loads(line)
                            if event.get('schema_version') != 1:
                                fail('engine_failed', 'encode', 'Invalid encode event protocol.')
                            self.event(job['job_id'], event)
                            terminal = event
                        if encoded.returncode or not terminal or terminal.get('event', terminal.get('type')) != 'result':
                            fail('engine_failed', 'encode', 'Native source encoding failed.', native=terminal)
                        self.probe(encoded_path)
                        manifest['params']['initial_latent_path'] = str(encoded_path)
                    resumable = plan['resolved']['execution_mode'] == 'resumable'
                    manifest['request_digest'] = plan['resolved']['execution_digest']
                    if resumable:
                        manifest['control_path'] = str(directory / 'control')
                    selected = plan['intent'].get('outputs', {'media': True, 'latent': True})
                    if selected.get('media'):
                        manifest['params']['output_path'] = str(directory / 'generated.mp4')
                    if selected.get('latent'):
                        manifest['params']['latent_output_path'] = str(directory / 'output.h3latent')
                    if resumable:
                        manifest['params']['checkpoint_path'] = str(directory / 'checkpoint.h3latent')
                    path = directory / 'manifest.json'
                    atomic(path, canonical(manifest))
                    code, final = self.execute_native(job, path, directory, resumable)
                    job = self.get('jobs', job['job_id'], 'job_id')
                    if final is None or final['type'] == 'error' or (code and final['type'] not in ('paused', 'cancelled')):
                        fail('engine_failed', 'engine', 'Native execution failed.', returncode=code, native=final)
                    if final.get('diagnostic_preview'):
                        fail('output_failed', 'outputs', 'A noisy diagnostic preview cannot be published as a final generation.')
                    outputs = final.get('outputs', {})
                    checkpoint = outputs.get('checkpoint')
                    if checkpoint and directory not in Path(checkpoint).resolve().parents:
                        fail('output_failed', 'checkpoint', 'Checkpoint escaped the job directory.')
                    if final['type'] == 'cancelled' and checkpoint and not job.get('keep_checkpoint'):
                        Path(checkpoint).unlink(missing_ok=True)
                        checkpoint = None
                    checkpoint_id = self.register({'path': checkpoint})['asset_id'] if checkpoint else None
                    if final['type'] in ('paused', 'cancelled'):
                        if final['type'] == 'paused' and not checkpoint_id:
                            fail('output_failed', 'checkpoint', 'Native pause did not publish a durable checkpoint.')
                        self.update(job, final['type'], checkpoint_id=checkpoint_id)
                    else:
                        published = {}
                        if any(selected.get(role) and not outputs.get(role) for role in ('media', 'latent')):
                            fail('output_failed', 'outputs', 'Native result omitted a requested output.')
                        for role in ('media', 'latent'):
                            if outputs.get(role):
                                output_path = Path(outputs[role]).resolve()
                                if directory not in output_path.parents:
                                    fail('output_failed', 'outputs', 'Engine output escaped the job directory.')
                                if role == 'media':
                                    output_path = self.assemble(job, plan, output_path, directory)
                                asset = self.register({'path': str(output_path)})
                                if role == 'media':
                                    assembly = plan['resolved'].get('assembly')
                                    expected_frames = assembly['final_frames'] if assembly else plan['summary']['frames']
                                    params = plan['resolved']['manifest']['params']
                                    if (asset.get('frames') != expected_frames or asset.get('width') != params['width'] or
                                            asset.get('height') != params['height'] or asset.get('fps') != 24):
                                        fail('output_failed', 'outputs.media', 'Published video metadata differs from the immutable plan.', expected_frames=expected_frames, observed=asset)
                                elif asset.get('metadata', {}).get('kind') != 'clean' or set(asset['streams']) != {'video', 'audio'}:
                                    fail('output_failed', 'outputs.latent', 'Generation must publish a clean audiovisual latent bundle.')
                                published[role] = asset['asset_id']
                        if not published:
                            fail('output_failed', 'outputs', 'Engine reported success without outputs.')
                        with self.transaction():
                            job = self.get('jobs', job['job_id'], 'job_id')
                            if not resumable and job['state'] == 'cancelling':
                                self.update(job, 'cancelled', outputs={}, checkpoint_id=None,
                                            cancel_mechanism='process_abort', native_returncode=code)
                            else:
                                self.update(job, 'succeeded', outputs=published, checkpoint_id=checkpoint_id)
                except ProcessAborted as error:
                    job = self.get('jobs', job['job_id'], 'job_id')
                    self.update(job, 'cancelled', outputs={}, checkpoint_id=None,
                                cancel_mechanism='process_abort', native_returncode=error.args[0])
                except TaskError as e:
                    job = self.get('jobs', job['job_id'], 'job_id')
                    self.update(job, 'failed', failure=e.value)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                    job = self.get('jobs', job['job_id'], 'job_id')
                    self.update(job, 'failed', failure=dict(code='engine_failed', message=str(e)))
                processed.append(job['job_id'])
                if request.get('once', False):
                    break
        finally:
            self.db.execute('DELETE FROM leases WHERE device=? AND owner=?', (device, owner))
        return dict(processed=processed)

    def dispatch(self, operation, request):
        methods = {'asset.register': self.register, 'continue': self.continue_task,
                   'profiles': lambda r: {'profiles': self.profiles()},
                   'pause': lambda r: self.control(r, 'pause'), 'cancel': lambda r: self.control(r, 'cancel')}
        method = methods.get(operation) or (getattr(self, operation, None) if operation in
                  ('capabilities', 'prepare', 'submit', 'inspect', 'revise', 'resume', 'worker') else None)
        if method is None:
            fail('invalid_request', 'operation', 'Unknown operation.')
        if not isinstance(request, dict):
            fail('invalid_request', '', 'Expected a JSON object.')
        try:
            return method(request)
        except (TypeError, AttributeError, KeyError, ValueError) as e:
            fail('invalid_request', '', 'Malformed request field.', reason=str(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['capabilities', 'profiles', 'asset.register', 'prepare', 'submit',
                                             'inspect', 'continue', 'revise', 'resume', 'pause', 'cancel', 'worker'])
    parser.add_argument('--json', action='store_true', required=True)
    parser.add_argument('--root', required=True, help='Explicit durable state directory')
    parser.add_argument('--config', help='Administrator JSON profile and binary configuration')
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            fail('invalid_request', '', 'Expected a JSON object.')
        result = Tasks(args.root, config).dispatch(args.operation, request)
        print(canonical(dict(ok=True, **result)))
        return 0
    except TaskError as e:
        print(canonical(dict(ok=False, error=e.value)))
        return 1
    except (OSError, ValueError, sqlite3.Error) as e:
        print(canonical(dict(ok=False, error=dict(code='invalid_request', field='', message=str(e), retryable=False))))
        return 1


if __name__ == '__main__':
    sys.exit(main())
