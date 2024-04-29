# from PIL import Image
# import numpy as np
# root="MT-Dataset/parsing/makeup/"
# paths=["XMY-387.png"]
# for path in paths:
#     path=root+path
#     seg = np.array(Image.open(path))
#     new = np.zeros_like(seg)
#     new[seg == 0] = 0
#     new[seg == 1] = 4
#     new[seg == 2] = 7
#     new[seg == 3] = 2
#     new[seg == 4] = 6
#     new[seg == 5] = 1
#     new[seg == 6] = 8
#     new[seg == 7] = 9
#     new[seg == 8] = 11
#     new[seg == 9] = 13
#     new[seg == 10] = 12
#     new[seg == 11] = 3
#     new[seg == 12] = 5
#     new[seg == 13] = 10
#     img = Image.fromarray(new)
#     img.save(path)


# -----------------------------------------------------------#
from PIL import Image
import numpy as np
paths="Xiqu-Dataset/segs/makeup"
# paths=["XMY-387.png"]
for path in paths:
    # path=root+path
    seg = np.array(Image.open(path))
    new = np.zeros_like(seg)
    new[seg == 0] = 0   # 背景
    new[seg == 1] = 4   # 左耳朵
    new[seg == 2] = 7
    new[seg == 3] = 2   # 右眉毛
    new[seg == 4] = 6   # 脸
    new[seg == 5] = 1   # 右眼
    new[seg == 6] = 8
    new[seg == 7] = 9
    new[seg == 8] = 11      # 左眼
    new[seg == 9] = 13      # 上嘴唇
    new[seg == 10] = 12
    new[seg == 11] = 3
    new[seg == 12] = 5
    new[seg == 13] = 10     # 鼻子
    img = Image.fromarray(new)
    img.save(path)
# -----------------------------------------------------------#