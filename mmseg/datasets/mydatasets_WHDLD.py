from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class MYDataset_WHDLD(CustomDataset):
    CLASSES = ("bare soil","building","pavement","road","vegrtation","water")
    PALETTE = [[128, 128, 128],[255, 0, 0], [192, 192, 0],[255,255,0],[0, 255, 0],[0,0,255]]
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         split=split, **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None
