"""Behavioral tests for the public task boundary; no inference or fake native binaries.

Set H3_TASK_ENGINE to an isolated built native CLI to exercise planner and queue
identity tests. These use an explicit inventory fixture, not model inference.
"""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import time
import unittest
import wave
import zlib

SPEC = importlib.util.spec_from_file_location('h3_tasks', Path(__file__).resolve().parents[1] / 'tools' / 'h3_tasks.py')
h3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h3)


def native_latent(path, values=None):
    values = [0.25] * 24 if values is None else values
    payload = struct.pack('=7QII', 1, 1, 1, 0, 24, 0, 0, 0, 0) + struct.pack('=24f', *values)
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte) * 1099511628211) & 0xffffffffffffffff
    path.write_bytes(b'H3NATIVE' + struct.pack('=IIIIQQ', 1, 5, 0x01020304, 0, len(payload), checksum) + payload)


def native_av_latent(path):
    # Legitimate synthetic imported clean tensors, not a fake engine output.
    vc, ac = 24 * 7 * 16 * 16, 64 * 37
    payload = struct.pack('=7QII', 7, 16, 16, 37, vc, ac, 22, 0, 0) + b'\0' * (4 * (vc + ac))
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte) * 1099511628211) & 0xffffffffffffffff
    path.write_bytes(b'H3NATIVE' + struct.pack('=IIIIQQ', 1, 5, 0x01020304, 0, len(payload), checksum) + payload)


def native_checkpoint(path, solver, revision, arrays):
    # Synthetic solver-state fixture, not an engine or inference substitute.
    payload = struct.pack('=12Q', solver, 2, 0, 1, 1, 42, 1, 1, 123, 0, revision, 0)
    payload += b'\x01' * 32 + struct.pack('=2f', 12., 3.)
    payload += struct.pack('=6f', .9, .4, 0., 9 / 13, 1 / 7, 0.)
    payload += b'\0' * (4 * arrays * 2)
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte) * 1099511628211) & 0xffffffffffffffff
    path.write_bytes(b'H3NATIVE' + struct.pack('=IIIIQQ', 1, 4, 0x01020304, 0, len(payload), checksum) + payload)


class AssetBehavior(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = h3.Tasks(self.root / 'state')
        self.addCleanup(self.tasks.db.close)

    def error(self, code, operation, request):
        with self.assertRaises(h3.TaskError) as caught:
            self.tasks.dispatch(operation, request)
        self.assertEqual(code, caught.exception.value['code'])
        return caught.exception.value

    def test_capability_absence_never_advertises_native_mechanisms(self):
        result = self.tasks.dispatch('capabilities', {})
        self.assertEqual({}, result['native']['capabilities'])
        self.assertEqual([], result['profiles'])

    def test_registered_bytes_are_pinned_and_content_addressed(self):
        source = self.root / 'source.native'
        native_latent(source)
        first = self.tasks.dispatch('asset.register', {'path': str(source)})
        copied = self.root / 'same.native'
        copied.write_bytes(source.read_bytes())
        same = self.tasks.dispatch('asset.register', {'path': str(copied)})
        self.assertEqual(first['asset_id'], same['asset_id'])
        source.write_bytes(b'changed after registration')
        self.assertNotEqual(source.read_bytes(), Path(first['path']).read_bytes())
        self.assertEqual('clean', first['metadata']['kind'])
        self.assertEqual(['video'], first['streams'])

    def test_corrupt_native_envelope_is_not_registered(self):
        source = self.root / 'broken.native'
        native_latent(source)
        source.write_bytes(source.read_bytes()[:-1])
        self.error('latent_incompatible', 'asset.register', {'path': str(source)})

    def test_nonfinite_latents_rejected_even_with_valid_checksum(self):
        source = self.root / 'nan.native'
        native_latent(source, [float('nan')] + [0.] * 23)
        self.error('latent_incompatible', 'asset.register', {'path': str(source)})

    def test_checkpoint_history_formats_require_their_matching_revision(self):
        for name, solver, revision, arrays in [('res', 2, 0, 6), ('er', 3, 1, 7)]:
            with self.subTest(name=name):
                source = self.root / (name + '.checkpoint')
                native_checkpoint(source, solver, revision, arrays)
                asset = self.tasks.dispatch('asset.register', {'path': str(source)})
                self.assertEqual('checkpoint', asset['metadata']['kind'])
                self.assertEqual(['video', 'audio'], asset['streams'])
        for name, revision, arrays in [('future', 2, 7), ('missing-history', 1, 6)]:
            with self.subTest(name=name):
                source = self.root / (name + '.checkpoint')
                native_checkpoint(source, 3, revision, arrays)
                self.error('checkpoint_incompatible', 'asset.register', {'path': str(source)})

    def test_nonregular_assets_rejected_before_probe(self):
        self.error('invalid_request', 'asset.register', {'path': str(self.root)})

    def test_resume_rejects_edits_before_job_lookup(self):
        self.error('checkpoint_incompatible', 'resume', {'job_id': 'unknown', 'prompt': 'changed'})

    def test_unknown_profile_has_field_error(self):
        error = self.error('profile_incompatible', 'prepare', {'task': {'profile': 'unknown'}})
        self.assertEqual('task.profile', error['field'])

    def test_external_playlists_are_rejected_before_any_media_tool(self):
        self.tasks.config.update(ffprobe=str(self.root / 'no-ffprobe'), ffmpeg=str(self.root / 'no-ffmpeg'))
        for index, contents in enumerate([
            '#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttps://example.invalid/segment.ts\n',
            'ffconcat version 1.0\nfile /outside/the/store/video.mp4\n',
            '#EXTM3U\n/outside/the/store/audio.mp3\n']):
            with self.subTest(index=index):
                source = self.root / f'playlist-{index}'
                source.write_text(contents)
                self.error('invalid_request', 'asset.register', {'path': str(source)})

    def test_external_movie_data_references_are_rejected_before_probe(self):
        self.tasks.config['ffprobe'] = str(self.root / 'no-ffprobe')
        def box(kind, body):
            return struct.pack('>I4s', len(body) + 8, kind) + body
        reference = box(b'url ', struct.pack('>I', 0) + b'file:///outside/the/store/video.bin\0')
        body = box(b'dref', struct.pack('>II', 0, 1) + reference)
        for kind in (b'dinf', b'minf', b'mdia', b'trak', b'moov'):
            body = box(kind, body)
        source = self.root / 'external.mov'
        source.write_bytes(box(b'ftyp', b'isom\0\0\0\0isom') + body)
        self.error('invalid_request', 'asset.register', {'path': str(source)})


@unittest.skipUnless(os.environ.get('H3_TASK_ENGINE'), 'requires a real isolated native CLI')
class NativePlannerBehavior(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        dependency = self.root / 'inventory.bin'
        dependency.write_bytes(b'explicit inventory fixture; never used for inference')
        self.config = {'engine': os.environ['H3_TASK_ENGINE'], 'device': 'test-planning-device',
                       'profiles': {'test-exact-v1': {'model_dir': str(self.root), 'target': 'fl2va',
                        'dependencies': [{'role': 'inventory_fixture', 'path': str(dependency),
                                          'sha256': h3.file_digest(dependency)}],
                        'params': {'width': 256, 'height': 256, 'steps': 2}, 'max_frames': 124,
                        'exact': True, 'quality_evidence': 'unverified'}}}
        self.tasks = h3.Tasks(self.root / 'state', self.config)
        self.addCleanup(self.tasks.db.close)

    def prepare(self, **patch):
        task = dict(profile='test-exact-v1', prompt='A fox in snow', frames=22)
        task.update(patch)
        return self.tasks.dispatch('prepare', {'task': task})

    def test_canonical_plan_identity_and_snapshot_survive_config_edits(self):
        first = self.prepare()
        same = self.prepare()
        self.assertEqual(first['plan_id'], same['plan_id'])
        self.config['profiles']['test-exact-v1']['params']['steps'] = 3
        changed = self.prepare()
        self.assertNotEqual(first['plan_id'], changed['plan_id'])
        inspected = self.tasks.dispatch('inspect', {'plan_id': first['plan_id']})
        self.assertEqual(2, inspected['resolved']['manifest']['params']['steps'])

    def test_normalization_requires_named_approval(self):
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(frames=23, normalization='require_exact')
        self.assertEqual('normalization_required', caught.exception.value['code'])
        prepared = self.prepare(frames=23, normalization=['frame_floor'])
        self.assertEqual(22, prepared['summary']['frames'])
        self.assertEqual('frame_floor', prepared['transformations'][0]['kind'])

    def test_standard_media_policy_declares_crop_but_not_semantic_trim(self):
        image = self.image()
        self.config['profiles']['test-exact-v1']['params'].update(width=512, height=512)
        guide = dict(asset_id=image, kind='image', frame=0)
        plan = self.prepare(guides=[guide])
        self.assertEqual('cover_crop', plan['transformations'][0]['kind'])
        self.assertEqual('standard_media_v1', plan['resolved']['normalization_policy']['id'])
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(guides=[guide], normalization='require_exact')
        self.assertEqual('normalization_required', caught.exception.value['code'])
        changed = self.prepare(guides=[guide], normalization=['cover_crop'])
        self.assertNotEqual(plan['plan_digest'], changed['plan_digest'])
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(frames=23)
        self.assertEqual('frame_floor', caught.exception.value['details']['transformation']['kind'])
        before_policy_change = self.prepare()
        self.config['profiles']['test-exact-v1']['normalization_policy'] = 'require_exact'
        after_policy_change = self.prepare()
        self.assertNotEqual(before_policy_change['plan_id'], after_policy_change['plan_id'])

    def test_concurrent_idempotent_submissions_and_restart(self):
        plan = self.prepare()
        request = dict(plan_id=plan['plan_id'], plan_digest=plan['plan_digest'], idempotency_key='same-key')
        def submit(_):
            tasks = h3.Tasks(self.root / 'state', self.config)
            try:
                return tasks.dispatch('submit', request)['job_id']
            finally:
                tasks.db.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            jobs = list(pool.map(submit, range(12)))
        self.assertEqual(1, len(set(jobs)))
        restarted = h3.Tasks(self.root / 'state', self.config)
        self.addCleanup(restarted.db.close)
        self.assertEqual(jobs[0], restarted.dispatch('submit', request)['job_id'])
        other = self.prepare(prompt='A different fox')
        request.update(plan_id=other['plan_id'], plan_digest=other['plan_digest'])
        with self.assertRaises(h3.TaskError) as caught:
            restarted.dispatch('submit', request)
        self.assertEqual('idempotency_conflict', caught.exception.value['code'])

    def test_dependency_change_prevents_admission(self):
        plan = self.prepare()
        Path(self.config['profiles']['test-exact-v1']['dependencies'][0]['path']).write_bytes(b'changed')
        with self.assertRaises(h3.TaskError) as caught:
            self.tasks.dispatch('submit', dict(plan_id=plan['plan_id'], plan_digest=plan['plan_digest'], idempotency_key='key'))
        self.assertEqual('dependency_changed', caught.exception.value['code'])

    def test_queued_cancellation_is_idempotent_and_events_are_bounded(self):
        plan = self.prepare()
        job = self.tasks.dispatch('submit', dict(plan_id=plan['plan_id'], plan_digest=plan['plan_digest'], idempotency_key='cancel'))
        first = self.tasks.dispatch('cancel', {'job_id': job['job_id']})
        second = self.tasks.dispatch('cancel', {'job_id': job['job_id']})
        self.assertEqual('cancelled', first['state'])
        self.assertEqual(first, second)
        page = self.tasks.dispatch('inspect', {'job_id': job['job_id'], 'limit': 1})
        self.assertEqual(1, len(page['events']))
        next_page = self.tasks.dispatch('inspect', {'job_id': job['job_id'], 'cursor': page['cursor'], 'limit': 1})
        self.assertGreater(next_page['cursor'], page['cursor'])

    def image(self, width=256, height=256):
        path = self.root / f'guide-{width}-{height}.png'
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        path.write_bytes(b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress((b'\0' + b'\x7f\x80\x81' * width) * height)) + chunk(b'IEND', b''))
        return self.tasks.dispatch('asset.register', {'path': str(path)})['asset_id']

    def test_end_relative_placement_and_overlap_field_precedence(self):
        image = self.image()
        plan = self.prepare(guides=[dict(asset_id=image, kind='image', frame=-1)])
        self.assertEqual(21, plan['resolved']['manifest']['guides'][0]['frame_index'])
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(guides=[dict(asset_id=image, kind='image', frame=0),
                                 dict(asset_id=image, kind='image', frame=0)])
        self.assertEqual('guide_overlap', caught.exception.value['code'])
        self.assertEqual('task.guides[1]', caught.exception.value['field'])

    def test_disjoint_streams_at_same_time_do_not_conflict(self):
        image = self.image()
        audio = self.root / 'guide.wav'
        with wave.open(str(audio), 'wb') as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(32000)
            output.writeframes(b'\0' * 32000)
        identity = self.tasks.dispatch('asset.register', {'path': str(audio)})['asset_id']
        plan = self.prepare(guides=[dict(asset_id=image, kind='image', frame=0),
                                   dict(asset_id=identity, kind='audio', frame=0)])
        self.assertEqual(2, len(plan['resolved']['manifest']['guides']))

    def test_resume_cannot_be_smuggled_through_prepare(self):
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(operation='resume')
        self.assertEqual('task.operation', caught.exception.value['field'])

    def test_queued_pause_offers_explicit_fresh_admission(self):
        plan = self.prepare(execution_mode='resumable')
        job = self.tasks.dispatch('submit', dict(plan_id=plan['plan_id'], plan_digest=plan['plan_digest'], idempotency_key='pause'))
        self.tasks.dispatch('pause', {'job_id': job['job_id']})
        inspected = self.tasks.dispatch('inspect', {'job_id': job['job_id']})
        self.assertIn('submit', inspected['next_operations'])
        self.assertNotIn('resume', inspected['next_operations'])
        self.assertEqual(plan['plan_id'], inspected['requeue']['plan_id'])

    def test_structured_intent_uses_official_sections_without_invented_context(self):
        plan = self.prepare(prompt='', intent=dict(setting='A bookstore cafe.',
            subjects=[dict(id='person', description='A person in a coat.')],
            actions=['The person opens a book.'], camera='Static medium shot.',
            dialogue=[dict(speaker_id='person', language='Chinese', text='你好。')]))
        prompt = plan['resolved']['manifest']['prompt']
        self.assertTrue(prompt.startswith('subject_definitions:\n<Subject 1> (S1):'))
        self.assertIn('(S1) <d>[Chinese] 你好。</d>', prompt)
        self.assertIn('non_diegetic_music:\nN/A', prompt)
        self.assertIn('A bookstore cafe.', prompt)
        self.assertNotIn('ContextIR', prompt)
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(intent={'setting': 'Conflicts with freeform prompt.'})
        self.assertEqual('intent_conflict', caught.exception.value['code'])

    def test_speaker_ids_follow_dialogue_not_subject_order_and_words_are_literal(self):
        plan = self.prepare(prompt='', intent=dict(
            subjects=[dict(id='alice', description='Alice waits.'), dict(id='bob', description='Bob speaks.'),
                      dict(id='observer', description='An unspoken observer.')],
            dialogue=[dict(speaker_id='bob', language='English', text='A & B.'),
                      dict(speaker_id='alice', language='English', text='I agree.')]))
        prompt = plan['resolved']['manifest']['prompt']
        self.assertIn('<Subject 1> (S2): Alice waits.', prompt)
        self.assertIn('<Subject 2> (S1): Bob speaks.', prompt)
        self.assertIn('<Subject 3>: An unspoken observer.', prompt)
        self.assertIn('(S1) <d>[English] A & B.</d>', prompt)
        self.assertNotIn('&amp;', prompt)
        self.assertIn('detailed_description:\n[Shot1]', prompt)

    def test_voice_binding_is_explicit_and_targets_a_defined_subject(self):
        audio = self.root / 'voice.wav'
        with wave.open(str(audio), 'wb') as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(32000)
            output.writeframes(b'\0' * 32000 * 4 * 2)
        asset = self.tasks.dispatch('asset.register', {'path': str(audio)})['asset_id']
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        plan = self.prepare(prompt='', references=[dict(asset_id=asset, role='audio')], intent=dict(
            subjects=[dict(id='actor', description='A person.')],
            dialogue=[dict(speaker_id='actor', language='English', text='Hello.')],
            reference_use=[dict(asset_id=asset, use='voice', subject_id='actor')]))
        self.assertIn('<Audio 1> is the voice-timbre reference for <Subject 1> (S1).',
                      plan['resolved']['manifest']['prompt'])

    def test_structured_reserved_delimiters_are_rejected_not_rewritten(self):
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(prompt='', intent=dict(setting='A room </d> with a door.'))
        self.assertEqual('invalid_request', caught.exception.value['code'])

    def test_continuation_uses_only_bounded_tail_and_explicit_join_accounting(self):
        video = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        'color=c=blue:s=256x256:r=24', '-frames:v', '56', '-c:v', 'libx264', str(video)], check=True)
        source = self.tasks.dispatch('asset.register', {'path': str(video)})
        plan = self.tasks.dispatch('continue', dict(asset_id=source['asset_id'], tail_frames=22,
            additional_frames=17, audio_mode='discard', join_mode='concatenated',
            assembly_policy='explicit_reencode_no_crossfade',
            task=dict(profile='test-exact-v1', prompt='The blue scene continues.')))
        self.assertEqual(39, plan['summary']['frames'])
        guide = plan['resolved']['manifest']['guides'][0]
        self.assertEqual(34, guide['source_start_frame'])
        self.assertEqual(22, guide['source_frames'])
        self.assertEqual(73, plan['resolved']['assembly']['final_frames'])
        self.assertEqual(22, plan['resolved']['assembly']['discarded_generated_overlap'])

    def test_embedded_video_audio_lowers_to_executable_native_reference_and_independent_ordinals(self):
        video = self.root / 'paired.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=256x256:r=24',
            '-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=stereo', '-frames:v', '56', '-t', '2.333333',
            '-c:v', 'libx264', '-c:a', 'aac', str(video)], check=True)
        paired = self.tasks.dispatch('asset.register', {'path': str(video)})['asset_id']
        audio = self.root / 'standalone.wav'
        with wave.open(str(audio), 'wb') as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(32000)
            output.writeframes(b'\0' * 32000 * 4 * 2)
        sound = self.tasks.dispatch('asset.register', {'path': str(audio)})['asset_id']
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        plan = self.prepare(prompt='', frames=56,
            normalization=['audio_resample', 'audio_right_pad', 'reference_audio_trim'],
            references=[dict(asset_id=sound, role='audio'), dict(asset_id=paired, role='video_audio')], intent=dict(
            subjects=[dict(id='actor', description='An actor.', reference_ids=[paired])],
            reference_use=[dict(asset_id=paired, use='voice', subject_id='actor'),
                           dict(asset_id=paired, role='audio', relationship='fully_copy'),
                           dict(asset_id=paired, role='video', relationship='attribute_transfer')]))
        reference = plan['resolved']['manifest']['references'][1]
        self.assertEqual('video', reference['kind'])
        self.assertTrue(reference['include_embedded_audio'])
        self.assertIn('<Audio 2> <Video 1>', plan['resolved']['manifest']['prompt'])
        self.assertIn('<Audio 2> is the voice-timbre reference for <Subject 1>.', plan['resolved']['manifest']['prompt'])
        prompt = plan['resolved']['manifest']['prompt']
        self.assertIn('<Audio 2>: fully_copy', prompt)
        self.assertIn('<Video 1>: attribute_transfer', prompt)
        self.assertNotIn('<Audio 2> <Video 1>: fully_copy', prompt)
        self.assertNotIn('<Audio 2> <Video 1>: attribute_transfer', prompt)

    def video(self, frames, fps=24):
        source = self.root / f'video-{frames}-{fps}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            f'color=c=blue:s=256x256:r={fps}', '-frames:v', str(frames), '-c:v', 'libx264', str(source)], check=True)
        return self.tasks.dispatch('asset.register', {'path': str(source)})

    def test_reference_target_truncation_requires_explicit_semantic_consent(self):
        source = self.video(56)
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        reference = dict(asset_id=source['asset_id'], role='video')
        for policy in ('require_exact', 'standard_media_v1'):
            with self.subTest(policy=policy), self.assertRaises(h3.TaskError) as caught:
                self.prepare(references=[reference], normalization=policy)
            self.assertEqual('reference_trim', caught.exception.value['details']['transformation']['kind'])
        plan = self.prepare(references=[reference], normalization=['reference_trim'])
        self.assertEqual({'start_frame': 0, 'frames': 22, 'fps': 24},
                         plan['resolved']['reference_preprocessing'][0]['video_selection'])

    def test_reference_cadence_floor_is_not_hidden_by_standard_media(self):
        source = self.video(30)
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        reference = dict(asset_id=source['asset_id'], role='video')
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(frames=56, references=[reference])
        self.assertEqual('reference_frame_floor', caught.exception.value['details']['transformation']['kind'])
        plan = self.prepare(frames=56, references=[reference], normalization=['reference_frame_floor'])
        self.assertEqual(22, plan['resolved']['reference_preprocessing'][0]['video_frames'])

    def test_reference_fps_and_cadence_changes_require_separate_visible_consent(self):
        source = self.video(30, 30)
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        reference = dict(asset_id=source['asset_id'], role='video')
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(frames=56, references=[reference], normalization='require_exact')
        self.assertEqual('time_resample', caught.exception.value['details']['transformation']['kind'])
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(frames=56, references=[reference])
        self.assertEqual('reference_frame_floor', caught.exception.value['details']['transformation']['kind'])
        plan = self.prepare(frames=56, references=[reference], normalization=['time_resample', 'reference_frame_floor'])
        self.assertEqual(['time_resample', 'reference_frame_floor'], [t['kind'] for t in plan['transformations']])
        self.assertEqual(24, plan['transformations'][1]['requested'])

    def test_reference_resize_is_distinct_from_guide_cover_crop(self):
        source = self.image(512, 256)
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        reference = dict(asset_id=source, role='image')
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(references=[reference], normalization='require_exact')
        self.assertEqual('reference_resize', caught.exception.value['details']['transformation']['kind'])
        plan = self.prepare(references=[reference])
        self.assertEqual([352, 192], plan['resolved']['reference_preprocessing'][0]['canvas'])
        self.assertEqual('stretch', plan['transformations'][0]['resolved']['fit'])

    def test_reference_audio_conversion_and_padding_are_disclosed_without_video_cap(self):
        path = self.root / 'mono-reference.wav'
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(44100)
            output.writeframes(b'\0' * (132741 * 2))
        source = self.tasks.dispatch('asset.register', {'path': str(path)})
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        reference = dict(asset_id=source['asset_id'], role='audio')
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(references=[reference], normalization='require_exact')
        self.assertEqual('audio_resample', caught.exception.value['details']['transformation']['kind'])
        plan = self.prepare(references=[reference])
        self.assertEqual(['audio_resample', 'audio_right_pad'], [t['kind'] for t in plan['transformations']])
        self.assertEqual(96320, plan['resolved']['reference_preprocessing'][0]['audio']['samples_32k'])

    def test_paired_reference_soundtrack_trim_is_explicit_not_standard_normalization(self):
        video = self.video(56)
        path = self.root / 'replacement.wav'
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(32000)
            output.writeframes(b'\0' * (96000 * 4))
        audio = self.tasks.dispatch('asset.register', {'path': str(path)})
        self.config['profiles']['test-exact-v1']['target'] = 'ref2va'
        reference = dict(asset_id=video['asset_id'], role='video_audio', audio_asset_id=audio['asset_id'])
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(frames=56, references=[reference])
        self.assertEqual('reference_audio_trim', caught.exception.value['details']['transformation']['kind'])
        plan = self.prepare(frames=56, references=[reference], normalization=['reference_audio_trim', 'audio_right_pad'])
        self.assertEqual(74667, plan['resolved']['reference_preprocessing'][0]['audio']['samples_32k'])
        native = plan['resolved']['manifest']['references'][0]
        self.assertEqual('video_audio', native['kind'])
        self.assertEqual(audio['path'], native['audio_path'])

    def test_clip_selection_uses_post_resampling_clock_in_both_directions(self):
        high_fps = self.video(30, 30)
        self.assertEqual(24, high_fps['frames_24'])
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(guides=[dict(asset_id=high_fps['asset_id'], kind='clip', frame=0,
                                      source_start_frame=8, source_frames=22)])
        self.assertEqual('invalid_request', caught.exception.value['code'])
        self.assertEqual('task.guides[0].source_frames', caught.exception.value['field'])
        low_fps = self.video(12, 12)
        self.assertEqual(24, low_fps['frames_24'])
        plan = self.prepare(guides=[dict(asset_id=low_fps['asset_id'], kind='clip', frame=0,
                                        source_start_frame=2, source_frames=22)])
        self.assertEqual(24, plan['resolved']['guide_spans'][0]['source_frame_clock'])
        self.assertEqual(2, plan['resolved']['manifest']['guides'][0]['source_start_frame'])

    def test_image_guides_reject_ignored_source_spans(self):
        source = self.image()
        for patch, field in [({'source_start_frame': 1}, 'source_start_frame'),
                             ({'source_frames': 22}, 'source_frames'), ({'source_frames': 0}, 'source_frames')]:
            with self.subTest(patch=patch), self.assertRaises(h3.TaskError) as caught:
                self.prepare(guides=[dict(asset_id=source, kind='image', frame=0, **patch)])
            self.assertEqual('invalid_request', caught.exception.value['code'])
            self.assertEqual('task.guides[0].' + field, caught.exception.value['field'])

    def test_stateful_profile_rejection_is_conditional_not_a_throughput_downgrade(self):
        params = self.config['profiles']['test-exact-v1']['params']
        for name, value in [('denoise_reuse', 2), ('core_reuse', 2), ('dit_layers', 45),
                            ('token_reduction', 1), ('ssd_streaming', 1)]:
            with self.subTest(parameter=name):
                params[name] = value
                throughput = self.prepare(execution_mode='throughput')
                self.assertEqual(value, throughput['resolved']['manifest']['params'][name])
                self.assertFalse(throughput['resolved']['controls']['pause'])
                with self.assertRaises(h3.TaskError) as caught:
                    self.prepare(execution_mode='resumable')
                self.assertEqual('profile_incompatible', caught.exception.value['code'])
                self.assertEqual(name, caught.exception.value['details']['parameter'])
                params.pop(name)
        default = self.prepare()
        self.assertEqual('throughput', default['resolved']['execution_mode'])
        self.assertEqual(50, default['resolved']['manifest']['params']['dit_layers'])

    def test_throughput_pause_and_resume_are_repairable_not_falsely_advertised(self):
        plan = self.prepare()
        job = self.tasks.dispatch('submit', dict(plan_id=plan['plan_id'], plan_digest=plan['plan_digest'], idempotency_key='throughput'))
        inspected = self.tasks.dispatch('inspect', {'job_id': job['job_id']})
        self.assertNotIn('pause', inspected['next_operations'])
        self.assertEqual('process_abort_without_checkpoint', inspected['controls']['cancel'])
        for operation in ('pause', 'resume'):
            with self.subTest(operation=operation), self.assertRaises(h3.TaskError) as caught:
                self.tasks.dispatch(operation, {'job_id': job['job_id']})
            self.assertEqual('profile_incompatible', caught.exception.value['code'])
            self.assertEqual('resumable', caught.exception.value['repairs'][0]['patch']['execution_mode'])

    def test_explicit_throughput_edit_is_rejected_before_admission(self):
        with self.assertRaises(h3.TaskError) as caught:
            self.prepare(execution_mode='throughput', initial_latent='unknown')
        self.assertEqual('profile_incompatible', caught.exception.value['code'])
        self.assertEqual('task.execution_mode', caught.exception.value['field'])

    def test_registered_clean_revision_selects_controlled_fresh_history(self):
        path = self.root / 'imported-clean.native'
        native_av_latent(path)
        asset = self.tasks.dispatch('asset.register', {'path': str(path)})
        plan = self.tasks.dispatch('revise', dict(asset_id=asset['asset_id'],
            task=dict(profile='test-exact-v1', frames=22, prompt='Original scene.'),
            patch=dict(prompt='Revised scene.', start_transition=1)))
        resolved = plan['resolved']
        self.assertEqual('resumable', resolved['execution_mode'])
        self.assertEqual('controlled_edit', resolved['execution_reason'])
        self.assertEqual('generate', resolved['manifest']['operation'])
        self.assertEqual(1, resolved['manifest']['params']['start_transition'])
        self.assertNotIn('resume_path', resolved['manifest']['params'])
        self.assertEqual(asset['path'], resolved['manifest']['params']['initial_latent_path'])


@unittest.skipUnless(os.environ.get('H3_TASK_INFERENCE_CONFIG'), 'requires explicit real-model inference approval/config')
class RealMediaRevision(unittest.TestCase):
    def test_media_only_concatenated_output_is_revised_at_published_not_generation_length(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(Path(os.environ['H3_TASK_INFERENCE_CONFIG']).read_text())
            profile = os.environ['H3_TASK_INFERENCE_PROFILE']
            tasks = h3.Tasks(directory, config)
            self.addCleanup(tasks.db.close)
            original = tasks.dispatch('prepare', {'task': dict(profile=profile, frames=22,
                prompt='A blue sphere rests on a plain white table.', outputs={'media': True})})
            job = tasks.dispatch('submit', dict(plan_id=original['plan_id'], plan_digest=original['plan_digest'], idempotency_key='original'))
            tasks.dispatch('worker', {'once': True})
            inspected = tasks.dispatch('inspect', {'job_id': job['job_id']})
            self.assertEqual('succeeded', inspected['state'], inspected.get('failure'))
            continued = tasks.dispatch('continue', dict(job_id=job['job_id'], tail_frames=5,
                additional_frames=17, audio_mode='condition', join_mode='concatenated',
                assembly_policy='explicit_reencode_no_crossfade',
                task=dict(profile=profile, prompt='The sphere remains on the table.', outputs={'media': True})))
            joined = tasks.dispatch('submit', dict(plan_id=continued['plan_id'], plan_digest=continued['plan_digest'], idempotency_key='joined'))
            tasks.dispatch('worker', {'once': True})
            published = tasks.dispatch('inspect', {'job_id': joined['job_id']})
            self.assertEqual('succeeded', published['state'], published.get('failure'))
            self.assertEqual(22, continued['summary']['frames'])
            revised = tasks.dispatch('revise', dict(job_id=joined['job_id'], patch=dict(
                prompt='A blue sphere rests on the same plain white table.', start_transition=1,
                normalization=['encode_source'])))
            self.assertEqual(39, revised['intent']['frames'])
            self.assertEqual(39, revised['resolved']['manifest']['params']['frames'])
            self.assertEqual('resumable', revised['resolved']['execution_mode'])
            child = tasks.dispatch('submit', dict(plan_id=revised['plan_id'], plan_digest=revised['plan_digest'], idempotency_key='revision'))
            tasks.dispatch('worker', {'once': True})
            result = tasks.dispatch('inspect', {'job_id': child['job_id']})
            self.assertEqual('succeeded', result['state'], result.get('failure'))
            self.assertIn('media', result['outputs'])

    def test_running_real_throughput_cancel_aborts_owned_process_without_published_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(os.environ['H3_TASK_INFERENCE_CONFIG']).resolve()
            config = json.loads(config_path.read_text())
            profile = os.environ['H3_TASK_INFERENCE_PROFILE']
            tasks = h3.Tasks(directory, config)
            self.addCleanup(tasks.db.close)
            plan = tasks.dispatch('prepare', {'task': dict(profile=profile, frames=22,
                execution_mode='throughput', prompt='A blue sphere rests on a white table.',
                outputs={'media': True, 'latent': True})})
            self.assertFalse(plan['resolved']['controls']['pause'])
            job = tasks.dispatch('submit', dict(plan_id=plan['plan_id'], plan_digest=plan['plan_digest'], idempotency_key='abort'))
            worker = subprocess.Popen([os.sys.executable, str(Path(h3.__file__).resolve()), 'worker', '--json',
                '--root', directory, '--config', str(config_path)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                worker.stdin.write('{"once":true}')
                worker.stdin.close()
                worker.stdin = None
                deadline = time.monotonic() + 120
                while True:
                    current = tasks.dispatch('inspect', {'job_id': job['job_id']})
                    if current['state'] == 'running':
                        break
                    self.assertEqual('queued', current['state'], current)
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                requested = tasks.dispatch('cancel', {'job_id': job['job_id']})
                self.assertEqual('process_abort', requested['cancel_mechanism'])
                stdout, stderr = worker.communicate(timeout=120)
                self.assertEqual(0, worker.returncode, (stdout, stderr))
                final = tasks.dispatch('inspect', {'job_id': job['job_id']})
                self.assertEqual('cancelled', final['state'], final.get('failure'))
                self.assertEqual({}, final['outputs'])
                self.assertIsNone(final['checkpoint_id'])
                self.assertNotIn('resume', final['next_operations'])
            finally:
                if worker.poll() is None:
                    worker.send_signal(signal.SIGINT)
                    worker.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
