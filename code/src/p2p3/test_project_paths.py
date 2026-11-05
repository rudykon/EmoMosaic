"""Portability regression tests; these checks require only the standard library."""
import unittest
from project_paths import DATA_ROOT, ROOT, resolve_asset


class PortablePathTests(unittest.TestCase):
    def test_mounted_data_prefix_does_not_capture_model_path(self):
        self.assertEqual(resolve_asset('/mnt/data/projects/old/models/rmnet/weights.pt'),
                         ROOT/'models/rmnet/weights.pt')

    def test_chinese_data_metadata_resolves_to_bundled_media(self):
        old = ('/home/old/project/data/原始数据/附件4-可解释专项视频样本与特征文件/'
               '附件4-可解释专项视频样本与特征文件/对齐版本/videos/01.mp4')
        self.assertEqual(resolve_asset(old), DATA_ROOT/'source_data/attachment4_interpretability/'
                         'attachment4_interpretability/aligned/videos/01.mp4')

    def test_original_source_directory_is_translated_after_mount_prefix(self):
        self.assertEqual(resolve_asset('/mnt/data/projects/old/src/P2P3_model/predict.py'),
                         ROOT/'src/p2p3/predict.py')


if __name__ == '__main__':
    unittest.main()
