import unittest

from beatfcn.ingest.phira_policy import (SCREEN_VERSION, metadata_identity,
                                       phira_stratum, screen_metadata)
from beatfcn.train.data import selected_for_training


class PhiraStrata(unittest.TestCase):
    def row(self, **changes):
        return dict(dict(source='phira', difficulty_label='IN Lv.13', difficulty=13,
                         rating=.9, rating_count=50, tags=['regular'], description=''), **changes)

    def test_tags_and_level_boundaries_remain_separate(self):
        self.assertEqual([phira_stratum(self.row(difficulty_label=f'IN Lv.{v}', difficulty=v))['key']
                          for v in [10.9, 11, 12.9, 13, 14.9, 15, 16]],
                         ['IN/8-11', 'IN/11-13', 'IN/11-13', 'IN/13-15',
                          'IN/13-15', 'IN/15-16', 'IN/16+'])
        self.assertEqual([phira_stratum(self.row(difficulty_label=f'{tag} Lv.13'))['key']
                          for tag in ['HD', 'IN', 'AT']], ['HD/13-15', 'IN/13-15', 'AT/13-15'])

    def test_conflicts_and_unknowns_do_not_fill_a_stratum(self):
        for changes in [dict(difficulty=1), dict(difficulty_label='IN', difficulty=0),
                        dict(difficulty_label='SP Lv.13')]:
            self.assertIsNone(phira_stratum(self.row(**changes))['key'])

    def test_explicitly_unreviewed_and_boolean_scores_do_not_pass(self):
        self.assertFalse(selected_for_training(self.row(reviewed=False)))
        self.assertFalse(selected_for_training(self.row(rating=True)))

    def test_explicit_performance_or_simulation_is_excluded(self):
        for tags in [['visual'], ['troll'], ['4K'], ['转谱'], ['表演谱']]:
            row = self.row(tags=tags)
            self.assertEqual(screen_metadata(row)['status'], 'excluded')
            self.assertFalse(selected_for_training(row))

    def test_standard_label_does_not_bypass_ambiguous_source_review(self):
        for description in ['这是4K模拟玩法', '本人转谱作品', '仅供观赏的演出谱']:
            row = self.row(description=description)
            self.assertEqual(screen_metadata(row)['status'], 'review_required')
            self.assertFalse(selected_for_training(row))

    def test_normal_animation_and_artistic_drags_are_not_auto_rejected(self):
        for description in ['包含演出效果；drag铺满这一段是音乐表现',
                            '不是转谱，非表演谱，原创谱面', '没有4K，正常双指谱']:
            self.assertTrue(selected_for_training(self.row(description=description)))

    def test_declared_primary_performance_and_converter_are_reviewed(self):
        for description in ['一张以note表演为主的谱，望大家喜欢',
                            '感谢起名钉子户的MalodyVersusPhigros转换器',
                            '本谱使用Malody制作，配置由我来写',
                            '依据上方按键点击下侧任意区域']:
            self.assertEqual(screen_metadata(self.row(description=description))['status'],
                             'review_required')

    def test_review_is_bound_to_metadata_and_cannot_erase_nonstandard_tags(self):
        row = self.row(description='曲绘来源：ADOFAI同名关卡')
        review = dict(screen_version=SCREEN_VERSION, metadata_identity=metadata_identity(row),
                      decision='eligible_metadata_only', reason='The reference names the illustration source.')
        row['phira_metadata_review'] = review
        self.assertTrue(selected_for_training(row))
        row['description'] = 'ADOFAI转谱'
        self.assertFalse(selected_for_training(row))
        row['tags'] = ['visual']
        review['metadata_identity'] = metadata_identity(row)
        self.assertFalse(selected_for_training(row))

    def test_ingestion_preserves_screening_evidence_even_on_parse_failure(self):
        from tempfile import TemporaryDirectory
        from beatfcn.ingest.pipeline import phira_zip_row
        row = dict(self.row(description='以note表演为主'), id=17,
                   level='IN Lv.13', name='Example')
        with TemporaryDirectory() as folder:
            parsed = phira_zip_row(folder, folder + '/missing.zip', row, folder)
        self.assertEqual(parsed['tags'], row['tags'])
        self.assertEqual(parsed['description'], row['description'])
        self.assertEqual(metadata_identity(parsed), metadata_identity(row))
        self.assertFalse(selected_for_training(parsed))


if __name__ == '__main__':
    unittest.main()
