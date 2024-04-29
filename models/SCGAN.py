import numpy as np
import torch
import os
import os.path as osp
from torch.autograd import Variable
from torchvision.utils import save_image
from .base_model import BaseModel
from . import net_utils
from .SCDis import SCDis
from .vgg import VGG
from .losses import GANLoss, HistogramLoss
from models.SCGen import SCGen


from torch import nn
from models import networks
# import networks
from models.networks import init_net
# from networks import init_net


class SCGAN(BaseModel):
    def name(self):
        return 'SCGAN'
    def __init__(self,dataset):
        super(SCGAN, self).__init__()
        self.dataloader = dataset

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.layers=['r41']
        self.phase=opt.phase
        self.lips = True
        self.eye = True
        self.skin = True
        self.num_epochs = opt.num_epochs
        self.num_epochs_decay = opt.epochs_decay
        self.g_lr = opt.g_lr
        self.d_lr = opt.d_lr
        self.g_step = opt.g_step
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.img_size = opt.img_size
        self.lambda_idt = opt.lambda_idt
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.lambda_his_lip = opt.lambda_his_lip
        self.lambda_his_skin_1 = opt.lambda_his_skin
        self.lambda_his_skin_2 = opt.lambda_his_skin
        self.lambda_his_eye = opt.lambda_his_eye
        self.lambda_vgg = opt.lambda_vgg
        self.snapshot_step = opt.snapshot_step
        self.save_step = opt.save_step
        self.log_step = opt.log_step
        self.result_path = opt.save_path
        self.snapshot_path = opt.snapshot_path
        self.d_conv_dim = opt.d_conv_dim
        self.d_repeat_num = opt.d_repeat_num
        self.norm1 = opt.norm1
        self.mask_A = {}
        self.mask_B = {}
        self.ispartial=opt.partial
        self.isinterpolation=opt.interpolation
        # xcj edit
        # self.input_dim_a = opt.input
        self.local_style_dis = True
        self.n_local = 12
        # 12: left_eye, right_eye, mouth, nose, left_cheek, right_cheek, left_eyebrow, right_eyebrow, nose_upper, forehead, mouth_left, mouth_right
        self.local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose',
                            'forehead', 'sidemouth', 'sidemouth_']

        '''
             dim: Any,
             style_dim: {__truediv__},
             n_downsample: Any,
             n_res: Any,
             mlp_dim: Any,
             n_componets: Any,
             input_dim: Any,
             phase: str = 'train',
             activ: str = 'relu',
             pad_type: str = 'reflect',
             ispartial: bool = False,
             isinterpolation: bool = False) -> None
        '''
        # xcj edit
        self.SCGen = SCGen(opt.ngf, opt.style_dim, opt.n_downsampling, opt.n_res, opt.mlp_dim, opt.n_componets,
                         opt.input_nc,  opt.phase,  ispartial=opt.partial, isinterpolation=opt.interpolation)
        self.D_A = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)
        self.D_B = SCDis(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm1)

        self.D_A.apply(net_utils.weights_init_xavier)
        self.D_B.apply(net_utils.weights_init_xavier)
        self.SCGen.apply(net_utils.weights_init_xavier)
        self.load_checkpoint()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()

        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.vgg = VGG()
        if self.phase == 'train':
            self.vgg.load_state_dict(torch.load('vgg_conv.pth'))
        self.criterionHis = HistogramLoss()

        self.g_optimizer = torch.optim.Adam(self.SCGen.parameters(), self.g_lr, [opt.beta1, opt.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), opt.d_lr,
                                              [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), opt.d_lr,
                                              [opt.beta1, opt.beta2])
        self.SCGen.cuda()
        self.vgg.cuda()
        self.criterionHis.cuda()
        self.criterionGAN.cuda()
        self.criterionL1.cuda()
        self.criterionL2.cuda()
        self.D_A.cuda()
        self.D_B.cuda()

        print('---------- Networks initialized -------------')
        net_utils.print_network(self.SCGen)

        # ---------------------------------------------------------xcj edit >
        # 设置局部判别器的网络
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis'+local_part.capitalize(),
                        init_net(networks.Dis_pair(opt.input_nc, opt.input_nc, 5),      # 为什么是5
                                 init_type='normal', gain=0.02))

        # 设置局部判别器的优化器
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis' + local_part.capitalize() + '_opt',
                        # xcj edit 我也加了一个过滤器
                        torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                getattr(self, 'dis' + local_part.capitalize()).parameters()),
                                         lr=0.0001,
                                         betas=(0.5, 0.999), weight_decay=0.0001))
        # ---------------------------------------------------------< xcj edit
    def load_checkpoint(self):
        G_path = os.path.join(self.snapshot_path ,'G.pth')
        if os.path.exists(G_path):
            dict=torch.load(G_path)
            self.SCGen.load_state_dict(dict)
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.snapshot_path, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        D_B_path = os.path.join(self.snapshot_path, 'D_B.pth')
        if os.path.exists(D_B_path):
            self.D_B.load_state_dict(torch.load(D_B_path))
            print('loaded trained discriminator B {}..!'.format(D_B_path))


    def set_input(self, input):
        self.mask_A=input['mask_A']
        self.mask_B=input['mask_B']
        makeup=input['makeup_img']
        nonmakeup=input['nonmakeup_img']
        makeup_seg=input['makeup_seg']
        nonmakeup_seg=input['nonmakeup_seg']
        # --------------------------------------------xcj edit
        ladn_data = input['ladn_data']
        self.ladn_data = ladn_data
        # --------------------------------------------xcj edit
        self.makeup=makeup
        self.nonmakeup=nonmakeup
        self.makeup_seg=makeup_seg
        self.nonmakeup_seg=nonmakeup_seg
        self.makeup_unchanged=input['makeup_unchanged']
        self.nonmakeup_unchanged=input['nonmakeup_unchanged']




    def to_var(self, x, requires_grad=False):
        if isinstance(x, list):
            return x
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


    def train(self):
        # forward
        self.iters_per_epoch = len(self.dataloader)
        g_lr = self.g_lr
        d_lr = self.d_lr
        start = 0

        for self.e in range(start, self.num_epochs):
            for self.i, data in enumerate(self.dataloader):

                if (len(data) == 0):
                    print("No eyes!!")
                    continue

                self.set_input(data)
                makeup, nonmakeup = self.to_var(self.makeup), self.to_var(self.nonmakeup),
                makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
                # makeup_unchanged=self.to_var(self.makeup_unchanged)
                # nonmakeup_unchanged=self.to_var(self.nonmakeup_unchanged)
                mask_makeup = {key: self.to_var(self.mask_B[key]) for key in self.mask_B}
                mask_nonmakeup = {key: self.to_var(self.mask_A[key]) for key in self.mask_A}
                # ladn_input = self.to_var(self.ladn_data)
                ladn_input = self.ladn_data

                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real
                out = self.D_A(makeup)

                d_loss_real = self.criterionGAN(out, True)

                # Fake
                fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)

                fake_makeup = Variable(fake_makeup.data).detach()
                out = self.D_A(fake_makeup)

                d_loss_fake = self.criterionGAN(out, False)


                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_A_optimizer.step()

                # Logging
                self.loss = {}
                self.loss['D-A-loss_real'] = d_loss_real.mean().item()


                # training D_B, D_B aims to distinguish class A
                # Real
                out = self.D_B(nonmakeup)
                d_loss_real = self.criterionGAN(out, True)
                # Fake

                fake_nonmakeup = self.SCGen(makeup, makeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)

                fake_nonmakeup = Variable(fake_nonmakeup.data).detach()
                out = self.D_B(fake_nonmakeup)
                d_loss_fake = self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_B_optimizer.step()

                # Logging
                self.loss['D-B-loss_real'] = d_loss_real.mean().item()


                # ----------------------------------xcj edit
                fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                fake_makeup = Variable(fake_makeup.data).detach()
                self.update_D_local_style(fake_makeup, ladn_input)
                # ----------------------------------xcj edit




                # ================== Train G ================== #
                if (self.i + 1) % self.g_step == 0:
                    # identity loss
                    assert self.lambda_idt > 0

                    # G should be identity if ref_B or org_A is fed
                    idt_A = self.SCGen(makeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                    idt_B = self.SCGen(nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg,nonmakeup, nonmakeup_seg)
                    loss_idt_A = self.criterionL1(idt_A, makeup) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL1(idt_B, nonmakeup) * self.lambda_B * self.lambda_idt
                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5
                    # loss_idt = loss_idt_A * 0.5


                    # GAN loss D_A(G_A(A))
                    # fake_A in class B,
                    fake_makeup = self.SCGen(nonmakeup, nonmakeup_seg, makeup, makeup_seg, makeup, makeup_seg)
                    pred_fake = self.D_A(fake_makeup)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)

                    # GAN loss D_B(G_B(B))
                    fake_nonmakeup = self.SCGen(makeup, makeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    pred_fake = self.D_B(fake_nonmakeup)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)



                    # histogram loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    if self.lips == True:
                        g_A_lip_loss_his = self.criterionHis(fake_makeup, makeup, mask_nonmakeup["mask_A_lip"],
                                                             mask_makeup['mask_B_lip'],
                                                             mask_nonmakeup["index_A_lip"],
                                                             nonmakeup) * self.lambda_his_lip
                        g_B_lip_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup, mask_makeup["mask_B_lip"],
                                                             mask_nonmakeup['mask_A_lip'],
                                                             mask_makeup["index_B_lip"], makeup) * self.lambda_his_lip
                        g_A_loss_his += g_A_lip_loss_his
                        g_B_loss_his += g_B_lip_loss_his
                    if self.skin == True:
                        g_A_skin_loss_his = self.criterionHis(fake_makeup, makeup, mask_nonmakeup["mask_A_skin"],
                                                              mask_makeup['mask_B_skin'],
                                                              mask_nonmakeup["index_A_skin"],
                                                              nonmakeup) * self.lambda_his_skin_1
                        g_B_skin_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup, mask_makeup["mask_B_skin"],
                                                              mask_nonmakeup['mask_A_skin'],
                                                              mask_makeup["index_B_skin"],
                                                              makeup) * self.lambda_his_skin_2
                        g_A_loss_his += g_A_skin_loss_his
                        g_B_loss_his += g_B_skin_loss_his
                    if self.eye == True:
                        g_A_eye_left_loss_his = self.criterionHis(fake_makeup, makeup,
                                                                  mask_nonmakeup["mask_A_eye_left"],
                                                                  mask_makeup["mask_B_eye_left"],
                                                                  mask_nonmakeup["index_A_eye_left"],
                                                                  nonmakeup) * self.lambda_his_eye
                        g_B_eye_left_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup,
                                                                  mask_makeup["mask_B_eye_left"],
                                                                  mask_nonmakeup["mask_A_eye_left"],
                                                                  mask_makeup["index_B_eye_left"],
                                                                  makeup) * self.lambda_his_eye
                        g_A_eye_right_loss_his = self.criterionHis(fake_makeup, makeup,
                                                                   mask_nonmakeup["mask_A_eye_right"],
                                                                   mask_makeup["mask_B_eye_right"],
                                                                   mask_nonmakeup["index_A_eye_right"],
                                                                   nonmakeup) * self.lambda_his_eye
                        g_B_eye_right_loss_his = self.criterionHis(fake_nonmakeup, nonmakeup,
                                                                   mask_makeup["mask_B_eye_right"],
                                                                   mask_nonmakeup["mask_A_eye_right"],
                                                                   mask_makeup["index_B_eye_right"],
                                                                   makeup) * self.lambda_his_eye
                        g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                        g_B_loss_his += g_B_eye_left_loss_his + g_B_eye_right_loss_his



                    # cycle loss
                    rec_A = self.SCGen(fake_makeup, nonmakeup_seg, nonmakeup, nonmakeup_seg, nonmakeup, nonmakeup_seg)
                    rec_B = self.SCGen(fake_nonmakeup, makeup_seg, makeup, makeup_seg, makeup, makeup_seg)

                    g_loss_rec_A = self.criterionL1(rec_A, nonmakeup) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, makeup) * self.lambda_B


                    # vgg loss
                    vgg_s = self.vgg(makeup, self.layers)[0]
                    vgg_s = Variable(vgg_s.data).detach()
                    vgg_fake_makeup = self.vgg(fake_makeup, self.layers)[0]
                    g_loss_A_vgg = self.criterionL2(vgg_fake_makeup, vgg_s) * self.lambda_A * self.lambda_vgg


                    vgg_r = self.vgg(nonmakeup, self.layers)[0]
                    vgg_r = Variable(vgg_r.data).detach()
                    vgg_fake_nonmakeup = self.vgg(fake_nonmakeup, self.layers)[0]
                    g_loss_B_vgg = self.criterionL2(vgg_fake_nonmakeup, vgg_r) * self.lambda_B * self.lambda_vgg
                    #local-per
                    # vgg_fake_makeup_unchanged=self.vgg(fake_makeup*nonmakeup_unchanged,self.layers)
                    # vgg_makeup_masked=self.vgg(makeup*makeup_unchanged,self.layers)
                    # vgg_nonmakeup_masked=self.vgg(nonmakeup*nonmakeup_unchanged,self.layers)
                    # vgg_fake_nonmakeup_unchanged=self.vgg(fake_nonmakeup*makeup_unchanged,self.layers)
                    # g_loss_unchanged=(self.criterionL2(vgg_fake_makeup_unchanged, vgg_nonmakeup_masked)+self.criterionL2(vgg_fake_nonmakeup_unchanged,vgg_makeup_masked))

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5


                    # Combined loss
                    g_loss = (g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his).mean()


                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=False)
                    self.g_optimizer.step()
                    # self.track("Generator backward")

                    # Logging
                    # self.loss['G-loss-unchanged']=g_loss_unchanged.mean().item()
                    self.loss['G-A-loss-adv'] = g_A_loss_adv.mean().item()
                    self.loss['G-B-loss-adv'] = g_B_loss_adv.mean().item()
                    self.loss['G-loss-org'] = g_loss_rec_A.mean().item()
                    self.loss['G-loss-ref'] = g_loss_rec_B.mean().item()
                    self.loss['G-loss-idt'] = loss_idt.mean().item()
                    self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).mean().item()
                    self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).mean().item()
                    self.loss['G-A-loss-his'] = g_A_loss_his.mean().item()

                    # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()

                # save the images
                if (self.i) % self.save_step == 0:
                    print("Saving middle output...")
                    self.imgs_save([nonmakeup, makeup, fake_makeup])

                # Save model checkpoints

            # Decay learning rate
            if (self.e + 1) % self.snapshot_step == 0:
                self.save_models()
            if (self.e + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    # ---------------------------------------------------------------------------------------------------------xcj edit
    def update_D_local_style(self, fake_A, ladn_input):
        self.input_A = self.to_var(ladn_input['img_A'], requires_grad=False)  # 按照论文中的图，这个用不到，但是fake_A,没有recks,来输入到ladn中
        # print('self.input_A难道不是一张图片吗', self.input_A.size())               # torch.Size([1, 3, 256, 256])
        self.input_B = self.to_var(ladn_input['img_B'], requires_grad=False)
        self.input_C = self.to_var(ladn_input['img_C'], requires_grad=False)

        self.rects_A = self.to_var(ladn_input['rects_A'], requires_grad=False)  # 按照论文中的图，这个用不到，但是fake_A,没有recks,来输入到ladn中
        # print('self.rects_A的值,是不是12张', self.rects_A.size())                  # torch.Size([1, 12, 4])
        self.rects_B = self.to_var(ladn_input['rects_B'], requires_grad=False)
        self.rects_C = self.to_var(ladn_input['rects_C'], requires_grad=False)

        # self.input_A = ladn_input['img_A'].to(self.device).detach()
        # self.input_B = ladn_input['img_B'].to(self.device).detach()
        # self.input_C = ladn_input['img_C'].to(self.device).detach()
        # self.rects_A = ladn_input['rects_A'].to(self.device).detach()
        # self.rects_B = ladn_input['rects_B'].to(self.device).detach()
        # self.rects_C = ladn_input['rects_C'].to(self.device).detach()

        # 得到 self.rects_transfer_encoded、self.rects_after_encoded、self.rects_blend_encoded、self.fake_B_encoded（融合了A的内容和B的特征）
        self.forward_local_style(fake_A)
        # self.forward_local_style()

        for i in range(self.n_local):
            local_part = self.local_parts[i]
            if '_' not in local_part:
                getattr(self, 'dis' + local_part.capitalize() + '_opt').zero_grad()
                # 直接调用backward_local_styleD()方法计算并返回判别器的风格损失loss_D_Style
                # print('self.rects_transfer_encoded[:, i, :]的类型', self.rects_transfer_encoded.type())    # torch.cuda.IntTensor
                # print('self.rects_after_encoded[:, i, :]的类型', self.rects_after_encoded.type())          # torch.cuda.IntTensor
                # print('self.rects_blend_encoded[:, i, :]的类型', self.rects_blend_encoded.type())          # torch.cuda.IntTensor
                loss_D_Style = self.backward_local_styleD(getattr(self, 'dis' + local_part.capitalize()), self.rects_transfer_encoded[:, i, :], self.rects_after_encoded[:, i, :], self.rects_blend_encoded[:, i, :], name=local_part)
                nn.utils.clip_grad_norm_(getattr(self, 'dis' + local_part.capitalize()).parameters(), 5)
                getattr(self, 'dis' + local_part.capitalize() + '_opt').step()
                setattr(self, 'dis' + local_part.capitalize() + 'Style_loss', loss_D_Style.item())
            else:
                local_part = local_part.split('_')[0]
                getattr(self, 'dis' + local_part.capitalize() + '_opt').zero_grad()
                loss_D_Style_ = self.backward_local_styleD(getattr(self, 'dis' + local_part.capitalize()), self.rects_transfer_encoded[:, i, :], self.rects_after_encoded[:, i, :], self.rects_blend_encoded[:, i, :], name=local_part + '2', flip=True)
                nn.utils.clip_grad_norm_(getattr(self, 'dis' + local_part.capitalize()).parameters(), 5)  # 对梯度进行裁剪,将梯度进行归一化使得其绝对值不超过5  ? ? ?
                getattr(self, 'dis' + local_part.capitalize() + '_opt').step()  # 优化
                loss_D_Style = getattr(self, 'dis' + local_part.capitalize() + 'Style_loss')
                # 最后将左右眼的损失都存储到 disEyeStyle_loss 中
                setattr(self, 'dis' + local_part.capitalize() + 'Style_loss', loss_D_Style + loss_D_Style_.item())
                self.loss['dis' + local_part.capitalize() + 'Style_loss'] = getattr(self, 'dis' + local_part.capitalize() + 'Style_loss')
    def forward_local_style(self, fake_A):
        self.forward_style(fake_A)    # 得到的self.fake_B_encoded没用到
        # half_size = self.batch_size//2      # 1
        # print()
        self.rects_transfer_encoded = self.rects_A[0:1]          # [0:1] 是什么作用    没有变化
        self.rects_after_encoded = self.rects_B[0:1]
        self.rects_blend_encoded = self.rects_C[0:1]

        # print('rects_A的值', self.rects_A)
        # print('self.rects_transfer_encoded的值', self.rects_transfer_encoded)
        '''
                    rects_A的值 
            tensor([[[267, 369, 142, 244],
                     [267, 369, 262, 364],
                     [372, 474, 203, 305],
                     [301, 403, 201, 303],
                     [327, 429, 145, 247],
                     [327, 429, 260, 362],
                     [224, 326, 146, 248],
                     [223, 325, 254, 356],
                     [321, 423, 202, 304],
                     [223, 325, 199, 301],
                     [349, 451, 154, 256],
                     [349, 451, 251, 353]]], device='cuda:0', dtype=torch.int32)
            self.rects_transfer_encoded的值 
            tensor([[[267, 369, 142, 244],
                     [267, 369, 262, 364],
                     [372, 474, 203, 305],
                     [301, 403, 201, 303],
                     [327, 429, 145, 247],
                     [327, 429, 260, 362],
                     [224, 326, 146, 248],
                     [223, 325, 254, 356],
                     [321, 423, 202, 304],
                     [223, 325, 199, 301],
                     [349, 451, 154, 256],
                     [349, 451, 251, 353]]], device='cuda:0', dtype=torch.int32)
            Traceback (most recent call last):
        '''

    '''
            输入的数据按照一定规则处理，并通过生成器模型生成一些输出
        '''
    # 看一下这个函数到底是干什么的
    def forward_style(self, fake_A):  # 得到的self.fake_B_encoded没用到
        # half_size = self.batch_size//2

        # print('half_size的值', half_size)
        # print('input_A的值', self.input_A)

        self.real_A_encoded = self.input_A[0:1]  # 这个half_size的作用是什么？弄明白，然后替换生成器，维度一致就够了吧
        self.real_B_encoded = self.input_B[0:1]
        self.real_C_encoded = self.input_C[0:1]
        # print('real_A_encoded的值', self.real_A_encoded)

        #     # get encoded z_c
        #
        #         enc_c 内容
        #         self.enc_c.forward_a:   经历以下操作
        #
        #                 # xa (3, 216, 216)
        #                 nn.Conv2d           e1_1_A ： (ngf, 108, 108)        ngf = 64
        #                 nn.LeakyReLU
        #                 nn.Conv2d
        #                 nn.BatchNorm2d      e1_2_A ： (ngf*2, 54, 54)
        #
        #         enc_a  样式
        #         同上👆
        '''self.z_content_a = self.enc_c.forward_a(self.real_A_encoded)
        self.z_content_a = (self.z_content_a[0].to(self.device), self.z_content_a[1].to(self.device))

        # get encoded z_a
        self.z_attr_b = self.enc_a.forward_b(self.real_B_encoded.to(self.backup_device))
        self.z_attr_b = self.z_attr_b.to(self.device)

        # first cross translation
        self.fake_B_encoded = self.gen.forward_b(*self.z_content_a, self.z_attr_b)      # 生成假图，应该换成我自己的生成器  ! ! !
        print('self.fake_B_encoded的维度', self.fake_B_encoded.size())'''

        # xcj edit --->
        # idt_A = self.G(image_s, image_s, mask_s, mask_s, dist_s, dist_s)
        # self.fake_B_encoded = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r)
        self.fake_B_encoded = fake_A
        # print()
        # print('self.self.fake_B_encoded的维度', self.fake_B_encoded.size())
        # xcj edit <---

    '''loss_D_Style = self.backward_local_styleD(getattr(self, 'dis' + local_part.capitalize()),
                                              self.rects_transfer_encoded[:, i, :], self.rects_after_encoded[:, i, :],
                                              self.rects_blend_encoded[:, i, :], name=local_part+'2', flip=True)'''
    def backward_local_styleD(self, netD, rects_transfer, rects_after, rects_blend, name='', flip=False):
        N = self.real_B_encoded.size(0)
        # print('N的值', N)
        C = self.real_B_encoded.size(1)
        H = rects_transfer[0][1] - rects_transfer[0][0]     # self.rects_transfer_encoded = self.rects_A[0:half_size]
        W = rects_transfer[0][3] - rects_transfer[0][2]

        # transfer_crop = torch.empty((N, C, H, W)).to(self.device)
        # after_crop = torch.empty((N, C, H, W)).to(self.device)
        # blend_crop = torch.empty((N, C, H, W)).to(self.device)
        transfer_crop = torch.empty((N, C, H, W)).cuda()
        after_crop = torch.empty((N, C, H, W)).cuda()
        blend_crop = torch.empty((N, C, H, W)).cuda()
        # print('transfer_crop的形状', transfer_crop.size())     # torch.Size([1, 3, 102, 102])
        # print('after_crop的形状', after_crop.size())           # torch.Size([1, 3, 102, 102])
        # print('blend_crop的形状', blend_crop.size())           # torch.Size([1, 3, 102, 102])

        for i in range(N):
            # global x1_c

            x1_t, x2_t, y1_t, y2_t = rects_transfer[i]
            # x1 = x1_t
            # x2 = x2_t
            # print('x1_t的值', x1_t)
            # print('x2_t的值', x2_t)
            # print('y1_t的值', y1_t)
            # print('y2_t的值', y2_t)
            x1_a, x2_a, y1_a, y2_a = rects_after[i]
            # print('x1_a的值', x1_a)
            # print('x2_a的值', x2_a)
            # print('y1_a的值', y1_a)
            # print('y2_a的值', y2_a)
            x1_b, x2_b, y1_b, y2_b = rects_blend[i]
            # print('x1_b的值', x1_b)
            # print('x2_b的值', x2_b)
            # print('y1_b的值', y1_b)
            # print('y2_b的值', y2_b)
            '''
                x1_t的值 tensor(271, device='cuda:0', dtype=torch.int32)
                x2_t的值 tensor(373, device='cuda:0', dtype=torch.int32)
                y1_t的值 tensor(145, device='cuda:0', dtype=torch.int32)
                y2_t的值 tensor(247, device='cuda:0', dtype=torch.int32)
                x1_a的值 tensor(273, device='cuda:0', dtype=torch.int32)
                x2_a的值 tensor(375, device='cuda:0', dtype=torch.int32)
                y1_a的值 tensor(133, device='cuda:0', dtype=torch.int32)
                y2_a的值 tensor(235, device='cuda:0', dtype=torch.int32)
                x1_b的值 tensor(271, device='cuda:0', dtype=torch.int32)
                x2_b的值 tensor(373, device='cuda:0', dtype=torch.int32)
                y1_b的值 tensor(145, device='cuda:0', dtype=torch.int32)
                y2_b的值 tensor(247, device='cuda:0', dtype=torch.int32)
            '''

            if not flip:
                # x1_t:x2_t的范围超过0~256
                # print('self.fake_B_encoded的形状', self.fake_B_encoded[i, :, x1_t:x2_t, y1_t:y2_t].size())     # torch.Size([1, 3, 256, 256])
                # print('x1_t的值', x1_t)
                # print('x2_t的值', x2_t)
                # print('self.real_B_encoded的形状', self.real_B_encoded.size())     # torch.Size([1, 3, 256, 256])
                # print('self.real_C_encoded', self.real_C_encoded.size())     # torch.Size([1, 3, 256, 256])
                # print('self.fake_B_encoded的形状', self.fake_B_encoded[i].size())         # torch.Size([1, 3, 256, 256])
                # print('self.real_B_encoded的形状', self.real_B_encoded.size())         torch.Size([1, 3, 256, 256])
                # print('self.real_C_encoded的形状', self.real_C_encoded.size())         torch.Size([1, 3, 256, 256])
                # print('x1_t的值', x1_t)       # 165
                # print('x2_t的值', x2_t)       # 267           范围超过 0~255    就是裁剪的时候按照512裁剪的   ! ! ! !
                # print('y1_t的值', y1_t)       # 79
                # print('y2_t的值', y2_t)       # 181

                # print('transfer_crop[i]的形状' , transfer_crop[i].size())      # torch.Size([3, 102, 102])
                # print('after_crop[i]的形状' , after_crop[i].size())            torch.Size([3, 102, 102])
                # print('blend_crop[i]的形状' , blend_crop[i].size())            torch.Size([3, 102, 102])
                # print('transfer_crop[i]的形状',  self.fake_B_encoded[i, :, x1_t:x2_t, y1_t:y2_t].size())

                transfer_crop[i] = self.fake_B_encoded[i, :, x1_t:x2_t, y1_t:y2_t].clone()      #  生成的假图 这个fake_B_encoded 要换成我自己的生成器网络  ！！ ！
                after_crop[i] = self.real_B_encoded[i, :, x1_a:x2_a, y1_a:y2_a].clone()         # 参考图
                blend_crop[i] = self.real_C_encoded[i, :, x1_b:x2_b, y1_b:y2_b].clone()         # 混合图




            else:
                id = [i for i in range(W - 1, -1, -1)]
                # idx = torch.LongTensor(id).to(self.device)
                # idx_backup = torch.LongTensor(id).to(self.device)
                idx = torch.LongTensor(id).cuda()
                idx_backup = torch.LongTensor(id).cuda()
                transfer_crop[i] = self.fake_B_encoded[i, :, x1_t:x2_t, y1_t:y2_t].index_select(2, idx).clone()
                after_crop[i] = self.real_B_encoded[i, :, x1_a:x2_a, y1_a:y2_a].index_select(2, idx_backup).clone()
                blend_crop[i] = self.real_C_encoded[i, :, x1_b:x2_b, y1_b:y2_b].index_select(2, idx_backup).clone()

        setattr(self, name + '_transfer', transfer_crop)
        setattr(self, name + '_after', after_crop)
        setattr(self, name + '_blend', blend_crop)

        # print('after_crop.detach()', after_crop.detach().type())        torch.cuda.FloatTensor
        # print('transfer_crop.detach()', transfer_crop.detach().type())      torch.cuda.FloatTensor
        pred_fake = netD.forward(after_crop.detach(), transfer_crop.detach())  # 参考图 + 生成的假图   判别为 0
        pred_real = netD.forward(after_crop.detach(), blend_crop.detach())     # 参考图 + 混合图   判别为 1
        loss_D = 0
        loss_D1 = loss_D
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            # all1 = torch.ones((out_real.size(0))).to(self.device)       # 和PSGAN一样，给一个全1的标签，来计算损失
            # all0 = torch.zeros((out_fake.size(0))).to(self.device)      # 和PSGAN一样，给一个全0的标签，来计算损失
            all1 = torch.ones((out_real.size(0))).cuda()  # 和PSGAN一样，给一个全1的标签，来计算损失
            all0 = torch.zeros((out_fake.size(0))).cuda()  # 和PSGAN一样，给一个全0的标签，来计算损失
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)   # 参考图 + 混合图   判别为 1
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)   # 参考图 + 生成的假图   判别为 0
            loss_D1 = loss_D + (ad_true_loss + ad_fake_loss)
        # loss_D2 = loss_D1 * self.style_d_ls_weight / self.n_local
        loss_D2 = loss_D1 * float(2) / self.n_local
        loss_D2.backward()
        return loss_D2

    # --------------------------------------------------------------------------------------------------------xcj edit
    def imgs_save(self, imgs_list):
        if self.phase == 'test':
            length = len(imgs_list)
            for i in range(0, length):
                imgs_list[i] = torch.cat(imgs_list[i], dim=3)
            imgs_list = torch.cat(imgs_list, dim=2)

            if not osp.exists(self.result_path):
                os.makedirs(self.result_path)
            save_path = os.path.join(self.result_path,
                                     '{}{}transferred.jpg'.format("partial_" if self.ispartial else "global_",
                                                                  "interpolation_" if self.isinterpolation else ""))
            save_image(self.de_norm(imgs_list.data), save_path, normalize=True)
        if self.phase == 'train':
            img_train_list = torch.cat(imgs_list, dim=3)
            if not osp.exists(self.result_path):
                os.makedirs(self.result_path)
            save_path = os.path.join(self.result_path, 'train/'+str(self.e)+'_'+str(self.i) + ".jpg")
            save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def log_terminal(self):
        log = " Epoch [{}/{}], Iter [{}/{}]".format(
            self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)


    def save_models(self):

        if not osp.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        torch.save(
            self.SCGen.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_A.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_B.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))

    def test(self):
        self.SCGen.eval()
        self.D_A.eval()
        self.D_B.eval()
        makeups = []
        makeups_seg = []
        nonmakeups=[]
        nonmakeups_seg = []
        for self.i, data in enumerate(self.dataloader):
            if (len(data) == 0):
                print("No eyes!!")
                continue
            self.set_input(data)
            makeup, nonmakeup = self.to_var(self.makeup), self.to_var(self.nonmakeup),
            makeup_seg, nonmakeup_seg = self.to_var(self.makeup_seg), self.to_var(self.nonmakeup_seg)
            makeups.append(makeup)
            makeups_seg.append(makeup_seg)
            nonmakeups.append(nonmakeup)
            nonmakeups_seg.append(nonmakeup_seg)
        source, ref1, ref2 = nonmakeups[0], makeups[0], makeups[1]
        source_seg, ref1_seg, ref2_seg = nonmakeups_seg[0], makeups_seg[0], makeups_seg[1]
        with torch.no_grad():
            transfered = self.SCGen(source, source_seg, ref1, ref1_seg, ref2, ref2_seg)
        if not self.ispartial and not self.isinterpolation:         # 全局 不插值
            results = [[source, ref1],
                    [source, ref2],
                    [ref1, source],
                    [ref2, source]
                    ]
            for i, img in zip(range(0, len(results)), transfered):
                results[i].append(img)
            self.imgs_save(results)
        elif not self.ispartial and self.isinterpolation:           # 全局 插值
            results = [[source, ref1],
                       [source, ref2],
                       [ref1, source],
                       [ref2, source],
                       [ref2, ref1]
                       ]
            for i, imgs in zip(range(0, len(results)-1), transfered):
                for img in imgs:
                    results[i].append(img)
            for img in transfered[-1]:
                results[-1].insert(1, img)
            results[-1].reverse()

            self.imgs_save(results)
        elif self.ispartial and not self.isinterpolation:           # 部分  不插值
            results = [[source, ref1],
                       [source, ref2],
                       [source, ref1, ref2],
                       ]
            for i, imgs in zip(range(0, len(results)), transfered):
                for img in imgs:
                    results[i].append(img)
            self.imgs_save(results)
        elif self.ispartial and self.isinterpolation:               # 部分 插值
            results = [[source, ref1],
                       [source, ref1],
                       [source, ref1],
                       [source, ref2],
                       [source, ref2],
                       [source, ref2],
                       [ref2, ref1],
                       [ref2, ref1],
                       [ref2, ref1],
                       ]
            for i, imgs in zip(range(0, len(results)-3), transfered):
                for img in imgs:
                    results[i].append(img)

            for i, imgs in zip(range(len(results)-3, len(results)), transfered[-3:]):
                for img in imgs:
                    results[i].insert(1, img)

                results[i].reverse()
            self.imgs_save(results)

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)