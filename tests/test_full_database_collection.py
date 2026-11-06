import unittest

from beatfcn.audio.mert2 import LOCAL_SOURCE_SHA256, REVISION
from scripts.collect_full_database_cloud_preparation import validate_reports


class CollectionTests(unittest.TestCase):
    def valid(self):
        return {
            'target_preparation': dict(complete=True, run_id='full-database-cloud-preparation-v7',
                manifest_sha256='m', parsed_rows=23211, planned_batches=91,
                completed_batches=list(range(91)), success_count=23211,
                permanent_failure_count=0),
            'raw_content': dict(complete=True, records=10898, verified=10897,
                unique_audio=10898, failures=[], cross_source_audio=16,
                alternate_physical_sources_not_reread=189,
                quarantined_content_count=1, all_content_verified=False,
                unquarantined_content_failures=[],
                quarantined_content=[dict(manifest_quarantined=True, kept=False,
                    audio_sha256='31123bc5b0a08b31486386e55eb2214b887971d8bfdc27313efaf779d92b442f')],
                verification_scope='one selected physical source per unique audio; alternates not reread'),
            'mel_content': dict(complete=True, parsed_audio=10898,
                records=10898, verified=10898, failures=[]),
            'checkpoint_upload': dict(complete=True, revision=REVISION,
                execution='modal_container', files={name: dict(sha256=value, bytes=1)
                    for name, value in LOCAL_SOURCE_SHA256.items()}),
        }

    def test_complete_remote_receipts_are_bound_without_corpus_payloads(self):
        result = validate_reports(self.valid(), 'm')
        self.assertEqual(result['target_charts'], 23211)
        self.assertEqual(result['raw_audio'], 10898)
        self.assertEqual(set(result['checkpoint_files']), set(LOCAL_SOURCE_SHA256))

    def test_pilot_prerequisites_do_not_require_full_mel_verification(self):
        values = self.valid()
        del values['mel_content']
        result = validate_reports(values, 'm', require_mel=False)
        self.assertIsNone(result['mel_audio'])
        self.assertEqual(result['mel_validation'], 'not_required_for_pilot')

    def test_incomplete_or_wrong_clock_receipt_fails(self):
        values = self.valid()
        values['target_preparation']['completed_batches'] = list(range(90))
        with self.assertRaisesRegex(ValueError, '91 batches'):
            validate_reports(values, 'm')
        values = self.valid()
        values['checkpoint_upload']['files']['model.safetensors']['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'pinned file'):
            validate_reports(values, 'm')


if __name__ == '__main__':
    unittest.main()
