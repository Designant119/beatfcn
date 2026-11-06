"""Package-level ambiguity and native-byte preservation for new corpus inputs."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.audit_plain_packages import package_members


class PlainPackageAuditTest(unittest.TestCase):
    def package(self, files, check):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.zip'
            with zipfile.ZipFile(path,'w') as archive:
                for name,raw in files.items():archive.writestr(name,raw)
            check(path)

    def test_native_bytes_and_info_offset_preserved_without_extraction(self):
        def check(path):
            info,content=package_members(path)
            self.assertEqual(info['offset'],.123)
            self.assertEqual(content['chart']['bytes'],b'{"test":1}')
            self.assertEqual(content['music']['bytes'],b'\x00\xffnative-audio')
            self.assertEqual(list(path.parent.iterdir()),[path])
        self.package({'sub/info.yml':b'chart: chart.json\nmusic: Music.ogg\noffset: 0.123\n',
                      'sub/chart.json':b'{"test":1}','sub/Music.ogg':b'\x00\xffnative-audio'},check)

    def test_ambiguous_basename_cannot_silently_select_another_chart(self):
        def check(path):
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                package_members(path)
        self.package({'info.yml':b'chart: chart.json\nmusic: music.ogg\n',
                      'a/chart.json':b'{}','b/chart.json':b'[]','music.ogg':b'a'},check)

    def test_missing_music_is_not_filled_with_preview(self):
        def check(path):
            with self.assertRaisesRegex(ValueError,'music missing'):
                package_members(path)
        self.package({'info.yml':b'chart: chart.json\nmusic: missing.ogg\n',
                      'chart.json':b'{}','preview.ogg':b'a'},check)


if __name__=='__main__':unittest.main()
