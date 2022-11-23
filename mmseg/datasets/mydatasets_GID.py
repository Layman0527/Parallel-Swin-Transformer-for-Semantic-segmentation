from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class MYDataset_GID(CustomDataset):
    CLASSES = ('Background',"industrial land","urban residential","rural residential","traffic land","paddy field","irrigated land",
               'dry cropland','garden plot','arbor woodland','shrub land','natural grassland','artificial grassland','river','lake','pond')
    PALETTE = [[0,0,0],[200,0,0],[250,0,150], [200,150,150],[250,150,150],[0, 200, 0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],
               [150,150,250],[250,200,0],[200,200,0],[0,0,200],[0,150,200],[0,200,250]]
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         split=split, **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None
