import unittest

from beatfcn.events.canonical import Event, LineLoc
from scripts.audit_phigros_resource import rhythm_hash


class RhythmIdentityTest(unittest.TestCase):
    def test_spatial_variants_share_rhythm_identity_but_multiplicity_is_preserved(self):
        a = Event('a', 100_000, 100_000, 'drag', LineLoc('0', 'above', 0.), 'test')
        b = Event('b', 100_000, 100_000, 'drag', LineLoc('1', 'below', 20.), 'test')
        self.assertEqual(rhythm_hash([a]), rhythm_hash([b]))
        self.assertNotEqual(rhythm_hash([a]), rhythm_hash([a, b]))
        self.assertEqual(rhythm_hash([a, b]), rhythm_hash([b, a]))

    def test_type_and_tail_are_part_of_identity(self):
        def event(kind, tail):
            return Event('a', 100_000, tail, kind, LineLoc('0', 'above', 0.), 'test')
        identities = {rhythm_hash([event('tap', 100_000)]),
                      rhythm_hash([event('drag', 100_000)]),
                      rhythm_hash([event('hold', 200_000)]),
                      rhythm_hash([event('hold', 300_000)])}
        self.assertEqual(len(identities), 4)


if __name__ == '__main__':
    unittest.main()
