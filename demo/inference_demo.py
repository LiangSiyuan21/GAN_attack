# coding:utf-8
# author:liangsiyuan
# @Time :2020/10/30  1:49 PM


from mmdet_v2200.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = 'configs_v2200/yolox/yolox_s_8x8_300e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/public/weight/od-black/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image
img = '/home/ubuntu/workspaces/longkangli/code/efficient_od_black/dc_mga_v5/work_dirs/taig/yolox/adv/Faster-RCNN/000000138856.jpg'
result = inference_detector(model, img)
# show the results
show_result_pyplot(model, img, result)