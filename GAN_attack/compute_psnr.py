import cv2
import numpy as np
import math
import os
 
def psnr1(img1, img2):
   mse = np.mean((img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)
 
def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def main():
    real_path = '/public/liangsiyuan/dataset/voc/VOCdevkit/VOC2007/JPEGImages'
    fake_path = 'checkpoints_297/G_GAN,G_cls_min,G_bbox_min,G_obj_min,P_L2/coco_translate/detection_on_adv/110/Faster-RCNN'
    images = os.listdir(fake_path)
    psnr_total = 0
    for image in images:
        fake_b = cv2.imread(os.path.join(fake_path, image))
        real_a = cv2.imread(os.path.join(real_path, image))
        psnr = psnr1(fake_b, real_a)
        psnr_total = psnr_total + psnr
        print(psnr)
    print('psnr_total:' + str(psnr_total/len(images)))

if __name__ == "__main__":
    main()