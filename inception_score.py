from cleanfid import fid
import numpy as np
from PIL import Image

# for i in range(10):
#     img = np.random.normal(0.2, 0.1, (32, 32, 3))
#     img = np.clip(img, 0., 1.)
#
#     im = Image.fromarray(img,'RGB')
#     im.save("./data/test/{}.jpeg".format(i))

fid_score = fid.compute_fid('data/test', dataset_name="cifar10", dataset_res=32,  mode="clean", dataset_split="train")
