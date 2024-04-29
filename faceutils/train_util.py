import numpy as np
from .api_util import FacePPAPI

'''
facePartsCoordinates using API landmark
    n_local
        3: left_eye, right_eye, mouth
        6: left_eye, right_eye, mouth, nose, left_cheek, right_cheek
        9: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper
        10: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead
        12: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead, mouth_left, mouth_right
'''
'''
    该代码定义了一个名为facePartsCoordinatesAPI的函数，该函数接受一个图像（img）和一个面部标记（landmarks），并返回一组矩形，这些矩形包含了人脸的各个部分，
    例如眼睛、嘴巴、鼻子等。该函数使用FacePPAPI类来获取landmarks中各个部分的坐标，并根据传递进来的参数（n_local、general_ratio和scaling_factor）
    决定返回哪些部位的矩形。最后，该函数返回一个矩形的列表（列表中是矩形的坐标），其中每个矩形表示一个部位。
'''
# def facePartsCoordinatesAPI(img, landmarks,
#                             n_local = 3,
#                             general_ratio = 0.2,
#                             scaling_factor = 1,     # 用于调整矩形的大小 ? ? ? , 1代表什么 ? ?
#                             ):
#     api = FacePPAPI()
#     img_size = img.shape[0]
#
#     left_eye = api.getLandmarkByName(landmarks, "left_eye_lower_left_quarter")[0]
#     # print('left_eye的值', left_eye)          # [197 323]
#
#     right_eye = api.getLandmarkByName(landmarks, "right_eye_lower_right_quarter")[0]
#
#     mouth = api.getLandmarkByName(landmarks, "mouth_lower_lip_top")[0]
#
#     nose = api.getLandmarkByName(landmarks, "nose_bridge3")[0]
#
#     left_cheek = api.getLandmarkByName(landmarks, ["nose_left_contour2", "contour_left8"])
#     left_cheek = np.round(np.array(left_cheek).mean(axis = 0))
#
#     right_cheek = api.getLandmarkByName(landmarks, ["nose_right_contour2", "contour_right8"])
#     right_cheek = np.round(np.array(right_cheek).mean(axis = 0))
#
#     left_eyebrow = api.getLandmarkByName(landmarks, "left_eyebrow_upper_middle")[0]
#
#     right_eyebrow = api.getLandmarkByName(landmarks, "right_eyebrow_upper_middle")[0]
#
#     nose_upper = api.getLandmarkByName(landmarks, "nose_tip")[0]
#
#     forehead = api.getLandmarkByName(landmarks, ['nose_bridge1', 'nose_bridge3'])
#     forehead = forehead[0] + forehead[0] - forehead[1]
#
#     mouth_left = api.getLandmarkByName(landmarks, "mouth_left_corner")[0]
#     mouth_right = api.getLandmarkByName(landmarks, "mouth_right_corner")[0]
#
#     n_local = int(n_local)
#     if n_local == 3:
#         parts = [left_eye, right_eye, mouth]
#     elif n_local == 6:
#         parts = [left_eye, right_eye, mouth,
#             nose, left_cheek, right_cheek]
#     elif n_local == 9:
#         parts = [left_eye, right_eye, mouth,
#             nose, left_cheek, right_cheek,
#             left_eyebrow, right_eyebrow, nose_upper]
#     elif n_local == 10:
#         parts = [left_eye, right_eye, mouth,
#             nose, left_cheek, right_cheek,
#             left_eyebrow, right_eyebrow, nose_upper, forehead]
#     elif n_local == 12:
#         parts = [left_eye, right_eye, mouth,
#             nose, left_cheek, right_cheek,
#             left_eyebrow, right_eyebrow, nose_upper, forehead,
#             mouth_left, mouth_right]
#     else:
#         raise Exception("Unknown number of local parts")
#
#     rects = []
#     for i, part in enumerate(parts):
#         part = (part * scaling_factor).round().astype(int)
#
#         center = part
#         radius = int(img_size * general_ratio / 2)
#
#         rects.append([
#             center[1]-radius, center[1]+radius,
#             center[0]-radius, center[0]+radius,
#         ])
#
#     return rects



def facePartsCoordinatesAPI(img, landmarks,
                            n_local = 3,
                            general_ratio = 0.1,            # xcj edit 原来是0.2
                            scaling_factor = 1,     # 用于调整矩形的大小 ? ? ? , 1代表什么 ? ?
                            ):
    api = FacePPAPI()
    img_size = img.shape[0]     # 512

    left_eye = api.getLandmarkByName(landmarks, "left_eye_lower_left_quarter")[0]
    # print('left_eye的值', left_eye)          # [197 323]

    right_eye = api.getLandmarkByName(landmarks, "right_eye_lower_right_quarter")[0]

    mouth = api.getLandmarkByName(landmarks, "mouth_lower_lip_top")[0]

    nose = api.getLandmarkByName(landmarks, "nose_bridge3")[0]

    left_cheek = api.getLandmarkByName(landmarks, ["nose_left_contour2", "contour_left8"])
    left_cheek = np.round(np.array(left_cheek).mean(axis = 0))

    right_cheek = api.getLandmarkByName(landmarks, ["nose_right_contour2", "contour_right8"])
    right_cheek = np.round(np.array(right_cheek).mean(axis = 0))

    left_eyebrow = api.getLandmarkByName(landmarks, "left_eyebrow_upper_middle")[0]

    right_eyebrow = api.getLandmarkByName(landmarks, "right_eyebrow_upper_middle")[0]

    nose_upper = api.getLandmarkByName(landmarks, "nose_tip")[0]

    forehead = api.getLandmarkByName(landmarks, ['nose_bridge1', 'nose_bridge3'])
    forehead = forehead[0] + forehead[0] - forehead[1]

    mouth_left = api.getLandmarkByName(landmarks, "mouth_left_corner")[0]
    mouth_right = api.getLandmarkByName(landmarks, "mouth_right_corner")[0]

    n_local = int(n_local)
    if n_local == 3:
        parts = [left_eye, right_eye, mouth]
    elif n_local == 6:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek]
    elif n_local == 9:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek,
            left_eyebrow, right_eyebrow, nose_upper]
    elif n_local == 10:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek,
            left_eyebrow, right_eyebrow, nose_upper, forehead]
    elif n_local == 12:
        parts = [left_eye, right_eye, mouth,
            nose, left_cheek, right_cheek,
            left_eyebrow, right_eyebrow, nose_upper, forehead,
            mouth_left, mouth_right]
    else:
        raise Exception("Unknown number of local parts")

    rects = []
    for i, part in enumerate(parts):
        # print('part1的值', part)      # [198. 362.]       [193 286]
        part = (part * scaling_factor).round().astype(int)       # 坐标点          没毛病，根据图片实际大小和训练大小进行缩放了
        # print('part2的值', part)      # [198 362]         [193 286]

        center = part
        radius = int(img_size * general_ratio / 2)

        rects.append([
            center[1]-radius, center[1]+radius,
            center[0]-radius, center[0]+radius,
        ])

    return rects