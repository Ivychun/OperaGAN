import os.path
import torchvision.transforms as transforms

from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable

from pathlib import Path
from numpy import asarray
import pickle
from faceutils.train_util import facePartsCoordinatesAPI

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class SCDataset():

    landmark_file = "landmark106.pk"

    def __init__(self, opt):


        self.random = None
        self.phase=opt.phase
        self.opt = opt
        self.root = opt.dataroot
        self.dir_makeup = opt.dataroot
        self.dir_nonmakeup = opt.dataroot
        self.dir_seg = opt.dirmap  # parsing maps
        self.n_componets = opt.n_componets
        self.makeup_names = []
        self.non_makeup_names = []
        if self.phase == 'train':
            self.makeup_names = [name.strip() for name in
                                 open(os.path.join('Xiqu-Dataset', 'makeup.txt'), "rt").readlines()]
            self.non_makeup_names = [name.strip() for name in
                                     open(os.path.join('Xiqu-Dataset', 'non-makeup.txt'), "rt").readlines()]
            # print('self.makeup_names长度', len(self.makeup_names)) 2046
            # print('长度',len(self.non_makeup_names))  1116

        if self.phase == 'test':
            with open("test.txt", 'r') as f:
                for line in f.readlines():
                    non_makeup_name, make_upname = line.strip().split()
                    self.non_makeup_names.append(non_makeup_name)
                    self.makeup_names.append(make_upname)
        self.transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_mask = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size), interpolation=PIL.Image.NEAREST),
            ToTensor])

        # ------------------------------xcj     edit
        self.data_dir = Path('F:\SCGAN—add-LADN-\SCGAN—add-LADN\Xiqu-Dataset')
        # # Load api landmars if landmark_file is provided
        if self.landmark_file:  # landmark.pk
            global api_landmarks
            api_landmarks = pickle.load(open(self.data_dir.joinpath(self.landmark_file), 'rb'))
            # print('api_landmarks的值', api_landmarks)    # 打印出来是人脸关键点的x和y坐标
            if len(api_landmarks.keys()) >= len(self.non_makeup_names) + len(self.makeup_names):
                self.api_landmarks = api_landmarks
                print("API landmarks loaded successfully!")  # 加载成功，没问题
            else:
                print("Number of landmarks is not enough.")
        self.api_landmarks = api_landmarks
        # ------------------------------xcj     edit

    def __getitem__(self, index):
        if self.phase == 'test':
            makeup_name = self.makeup_names[index]
            nonmakeup_name = self.non_makeup_names[index]
        if self.phase == 'train':
            index = self.pick()
            makeup_name = self.makeup_names[index[0]]
            nonmakeup_name = self.non_makeup_names[index[1]]
        # self.f.write(nonmakeup_name+' '+makeup_name+'\n')
        # self.f.flush()
        nonmakeup_path = os.path.join(self.dir_nonmakeup, nonmakeup_name)       # Xiqu-Dataset/images\non-makeup/181.png
        makeup_path = os.path.join(self.dir_makeup, makeup_name)                # Xiqu-Dataset/images\makeup/620_.png

        makeup_img = Image.open(makeup_path).convert('RGB')
        nonmakeup_img = Image.open(nonmakeup_path).convert('RGB')

        #--------------------------------------------------------------------xcj edit
        size_A = nonmakeup_img.size[0]          # size_A 要没有经过transform的图片
        size_B = makeup_img.size[0]

        num_A = nonmakeup_path.split('/')[-1].split('.')[0]  # 547_
        num_B = makeup_path.split('/')[-1].split('.')[0]  # 672
        path_C = os.path.join("blend", "%s_%s.jpg" % (num_A, num_B))  # blend\82_136.jpg
        img_C = self.read_file_C(path_C)
        size_C = img_C.size[0]                                          # 为什么先获取原图像的尺寸再裁剪  ？？？？？
        img_C = img_C.resize((self.opt.img_size, self.opt.img_size))

        # print('size_A', size_A)         512
        # print('size_B', size_B)         512
        # print('size_C', size_C)         512

        img_A_arr = asarray(nonmakeup_img)      # 要没有经过transform的图片
        img_B_arr = asarray(makeup_img)
        img_C_arr = asarray(img_C)

        use_api_landmark = not self.landmark_file is None
        if use_api_landmark:  # True
            # print('nonmakeup_path', nonmakeup_path.split('\\')[-2:])     # 'images\\non-makeup/155.png'
            # print('makeup_path.split(' / ')[-2:]', makeup_path.split('/')[-2:])
            aa = nonmakeup_path.split('\\')[-2:]
            bb = makeup_path.split('\\')[-2:]
            landmark_A_api = self.api_landmarks[aa[1]]
            landmark_B_api = self.api_landmarks[bb[1]]  # 要的是 before\24.jpg 这种效果
            landmark_C_api = landmark_A_api

        # Get the rects of eyes and mouth
        # 用图像大小为512时裁剪的矩形框（是坐标）
        # print('self.opt.img_size / size_A)的值', self.opt.img_size / size_A)
        rects_A = facePartsCoordinatesAPI(img_A_arr, landmark_A_api, n_local=12, scaling_factor=self.opt.img_size / size_A)
        rects_B = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=12, scaling_factor=self.opt.img_size / size_B)
        rects_C = rects_A

        rects_A = np.array(rects_A).astype(int)  # 和rects_A = facePartsCoordinatesAPI的值没变化
        rects_B = np.array(rects_B).astype(int)
        rects_C = np.array(rects_C).astype(int)
        # print('rects_A_的值', rects_A)
        # print('rects_B_的值', rects_B_)
        # print('rects_C_的值', rects_C_)

        for r in [rects_A, rects_B, rects_C]:
            assert r[:, :2].max() - r[:, :2].min() <= 256
            assert r[:, 2:].max() - r[:, 2:].min() <= 256

        ladn_data = {
            # "img_A": img_A,
            # "img_B": img_B,
            # "img_C": img_C,
            "img_A": self.transform(nonmakeup_img),
            "img_B": self.transform(makeup_img),
            "img_C": self.transform(img_C),
            "rects_A": rects_A,
            "rects_B": rects_B,
            "rects_C": rects_C,
            "index_A": num_A,
            "index_B": num_B,
        }
        # --------------------------------------------------------------------xcj edit

        makeup_seg_img = Image.open(os.path.join(self.dir_seg, makeup_name))
        nonmakeup_seg_img = Image.open(os.path.join(self.dir_seg, nonmakeup_name))
        # makeup_img = makeup_img.transpose(Image.FLIP_LEFT_RIGHT)
        # makeup_seg_img = makeup_seg_img.transpose(Image.FLIP_LEFT_RIGHT)
        # nonmakeup_img=nonmakeup_img.rotate(40)
        # nonmakeup_seg_img=nonmakeup_seg_img.rotate(40)
        # makeup_img=makeup_img.rotate(90)
        # makeup_seg_img=makeup_seg_img.rotate(90)
        makeup_img = self.transform(makeup_img)
        nonmakeup_img = self.transform(nonmakeup_img)


        mask_B = self.transform_mask(makeup_seg_img)  # makeup
        mask_A = self.transform_mask(nonmakeup_seg_img)  # nonmakeup


        makeup_seg = torch.zeros([self.n_componets, 256, 256], dtype=torch.float)
        nonmakeup_seg = torch.zeros([self.n_componets, 256, 256], dtype=torch.float)

        # 左眉毛、右眉毛、左眼、右眼、牙齿
        nonmakeup_unchanged = (mask_A == 7).float() + (mask_A == 2).float() + (mask_A == 6).float() + (
                mask_A == 1).float() + (mask_A == 11).float()
        # 戏曲
        makeup_unchanged = (mask_B == 2).float() + (mask_B == 3).float() + (mask_B == 4).float() + (
                mask_B == 5).float() + (mask_B == 11).float()


        # 上嘴唇、下嘴唇
        mask_A_lip = (mask_A == 9).float() + (mask_A == 13).float()
        # 戏曲
        mask_B_lip = (mask_B == 12).float() + (mask_B == 13).float()
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        makeup_seg[0] = mask_B_lip
        nonmakeup_seg[0] = mask_A_lip


        # 皮肤，鼻子，脖子
        mask_A_skin = (mask_A == 4).float() + (mask_A == 8).float() + (mask_A == 10).float()
        # 戏曲
        mask_B_skin = (mask_B == 1).float() + (mask_B == 10).float() + (mask_B == 14).float()
        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
        makeup_seg[1] = mask_B_skin
        nonmakeup_seg[1] = mask_A_skin


        # 左眼、右眼
        mask_A_eye_left = (mask_A == 6).float()
        mask_A_eye_right = (mask_A == 1).float()
        # 戏曲
        mask_B_eye_left = (mask_B == 4).float()
        mask_B_eye_right = (mask_B == 5).float()
        # 皮肤、鼻子
        mask_A_face = (mask_A == 4).float() + (mask_A == 8).float()
        # 戏曲
        mask_B_face = (mask_B == 1).float() + (mask_B == 10).float()
        # avoid the es of ref are closed
        if not ((mask_B_eye_left > 0).any() and \
                (mask_B_eye_right > 0).any()):
            return {}
        # mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)
        makeup_seg[2] = mask_B_eye_left + mask_B_eye_right
        nonmakeup_seg[2] = mask_A_eye_left + mask_A_eye_right

        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left
        mask_A["index_A_eye_right"] = index_A_eye_right
        mask_A["mask_A_skin"] = mask_A_skin
        mask_A["index_A_skin"] = index_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        mask_A["index_A_lip"] = index_A_lip

        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["index_B_eye_left"] = index_B_eye_left
        mask_B["index_B_eye_right"] = index_B_eye_right
        mask_B["mask_B_skin"] = mask_B_skin
        mask_B["index_B_skin"] = index_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        mask_B["index_B_lip"] = index_B_lip
        return {'nonmakeup_seg': nonmakeup_seg, 'makeup_seg': makeup_seg, 'nonmakeup_img': nonmakeup_img, 'makeup_img': makeup_img,
                'mask_A': mask_A, 'mask_B': mask_B,
                'makeup_unchanged': makeup_unchanged, 'nonmakeup_unchanged': nonmakeup_unchanged,
                'ladn_data' : ladn_data }

    def pick(self):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return [a_index, another_index]

    def __len__(self):
        if self.opt.phase == 'train':
            return len(self.non_makeup_names)
        elif self.opt.phase == 'test':
            return len(self.makeup_names)

    def name(self):
        return 'SCDataset'

    # xcj edit --->
    def read_file_C(self, name):

        base_path = self.data_dir
        # print('patc_C的路径', base_path.joinpath(name).as_posix())       G:/SCGAN—add-LADN/Xiqu-Dataset/blend/488__410_.jpg
        image = Image.open(
            base_path.joinpath(name).as_posix()
        ).convert("RGB")
        return image

    # xcj edit <---

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        mask_A_face = mask_A_face.unsqueeze(0)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6] = \
            mask_A_face[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6]
        mask_B_temp[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6] = \
            mask_A_face[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6]
        # mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        # mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        mask_A_temp = mask_A_temp.squeeze(0)
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        mask_A_face = mask_A_face.squeeze(0)
        mask_B_temp = mask_B_temp.squeeze(0)

        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]

        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        # mask_A = self.to_var(mask_A, requires_grad=False)
        # mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        return mask_A, mask_B, index, index_2

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


class SCDataLoader():
    def __init__(self, opt):
        self.dataset = SCDataset(opt)
        print("Dataset loaded")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches)
            # num_workers=int(opt.nThreads))


    def name(self):
        return 'SCDataLoader'


    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
