# OperaGAN
The code for the paper "OperaGAN: A Simultaneous Transfer Network for Opera Makeup and Complex Headwear"
## Prepare
The pre-trained model is avaiable at https://drive.google.com/drive/folders/1TDJIGVgrt46PnouJpHDvzAR2wRbJ-8iW?usp=drive_link

Environments:python=3.8, pytorch=1.6.0, Ubuntu=20.04.1 LTS

Generate the blending result of the source image and the reference image using the Poisson blending method and save it in ./Xiqu-Dataset/blend. 
## Train
Put the train-list of makeup images in "./Xiqu-Dataset/makeup.txt" and the train-list of non-makeup images in "./Xiqu-Dataset/non-makeup.txt"

Use `python sc.py --phase train` to train
## Test
### 1.Global Makeup Transfer
`python sc.py --phase test`

![Global Makeup Transfer](https://github.com/)
### 2.Global Interpolation
`python sc.py --phase test --interpolation`

![Global Interpolation](https://github.com/)

