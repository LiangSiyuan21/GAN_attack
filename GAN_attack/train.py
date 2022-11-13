#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   iou_nes_attack_demo.py
@Time    :   2021/01/12 18:57:09
@Author  :   Siyuan Liang
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from matplotlib import transforms

from numpy.core.fromnumeric import std
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
import torchvision
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import numpy as np
import pandas as pd
import torch
import argparse
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import warnings
import cv2
# import utils
import time
import attack_demo.util as demo_utils
from datetime import datetime
import mmcv
from PIL import Image
import os.path as osp
import sys
import torch
from mmcv import Config, DictAction
from mmcv import tensor2imgs
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import matplotlib as mpl
mpl.use('Agg')
from mmdet_v2200.core import encode_mask_results
from mmdet_v2200.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet_v2200.models import build_detector
from attack_demo.util import Logger

def parse_args():
    # detector argument
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out_log', type=str, help='save cmd output as log file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--out_dir', default='/srv/hdd/results/od-black/mmdetection/sq_attack/det', help='directory where painted images will be saved')      
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=0)

    # attack argument
    parser.add_argument('--attack', type=str, default='IoUnes_linf', choices=['IoUnes_linf', 'IoUnes_l2'], help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='/srv/hdd/results/od-black/mmdetection/sign_attack/adv', help='Experiment folder to store all output.')
    parser.add_argument('--n_iter', type=str, default='10000')
    parser.add_argument('--p', type=str, default='inf', choices=['inf', 'l2'])
    parser.add_argument('--targeted', default=False)
    parser.add_argument('--model', default='Faster-RCNN')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.0.05*4.52=0.22')
    parser.add_argument('--loss', type=str, default='cw_loss')
    parser.add_argument('--fd_eta', type=float, default=0.01)
    parser.add_argument('--attack_logistics', default=None)
    parser.add_argument('--vis_attack_step', default=None)
    parser.add_argument('--zeta', type=float, default='0.5')
    parser.add_argument('--lambda1', type=float, default='1.0')
    parser.add_argument('--patch_attack', default=None)
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--q', default=50)
    parser.add_argument('--name', default="coco_pix2pix")
    parser.add_argument('--GAN_model', default="pix2pix")
    parser.add_argument('--direction', default="AtoB")
    parser.add_argument('--dataset_mode', default="aligned")
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='frequency of saving the latest results')
    parser.add_argument('--loss_names', type=str, default='loss_G_L1', help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--loss_paras', type=str, default='1', help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')

    args = parser.parse_args()
    # args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()
    if args.attack_logistics is not None:
        attack_logistics = bool(args.attack_logistics)
    else:
        attack_logistics = None

    args.zeta = float(args.zeta)
    args.lambda1 = float(args.lambda1)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    args.checkpoints_dir = args.checkpoints_dir + '/' + args.loss_names + '/'
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    cfg = Config.fromfile(args.config)

    if args.out_log is not None:
        out_log_dic = args.out_log.rsplit("/",1)[0]
        if not os.path.exists(out_log_dic):
            os.makedirs(out_log_dic)
        if os.path.exists(args.out_log):
            os.remove(args.out_log)
        log = Logger(args.out_log, level='info')
    
    log.logger.info(args)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    
    if 'img' in cfg.data.train.pipeline[7]['keys']:
        cfg.data.train.pipeline[7]['keys'].append('adv_img')
    dataset = build_dataset(cfg.data.train, {'have_ad_img':True})
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=args.num_gpus,
        # workers_per_gpu=0,
        dist=distributed,
        shuffle=True)
    log.logger.info('Load data from the path:{}'.format(cfg.data.test['img_prefix']))
    
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    log.logger.info('checkpoints have been finishe into object detection model...')

    # initilaze object detector
    # model = MMDataParallel(model, device_ids=[0])
    # model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False,find_unused_parameters=find_unused_parameters)
    # model.eval()

    # prepare dataset (test dataset: 991 images)
    dataset = data_loader.dataset

    # define the GAN
    opt = TrainOptions().parse()   # get training options
    visualizer = Visualizer(opt)
    print('The number of training images = %d' % len(dataset))

    GAN_model = create_model(opt)      # create a model given opt.model and other options
    GAN_model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0   

    timestamp = str(datetime.now())[:-7]

    # init log file
    # hps_str = '{} model={} dataset={} attack={} eps={} p={} n_iter={}'.format(timestamp, args.model, dataset, args.attack, args.eps, args.p, args.n_iter)
    # log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
    # log = sq_utils.Logger(log_path)
    # log = sq_utils.Logger('')
    # log.print('All hps: {}'.format(hps_str))
    # if args.attack.split('_')[0] == 'IoUzosignsgd':
    #     results_records_iter_list = demo_utils.results_records_iter_list_for_zosignsgd
    # elif args.attack.split('_')[0] == 'IoUsquare':
    #     results_records_iter_list = demo_utils.results_records_iter_list_for_square
    # elif args.attack.split('_')[0] == 'IoUsign':
    #     results_records_iter_list = demo_utils.results_records_iter_list_for_signhunter
    # elif args.attack.split('_')[0] == 'IoUnes':
    #     results_records_iter_list = demo_utils.results_records_iter_list_for_nes
    
    # results = []
    # total_quires = []
    # total_times = []

    # prepare target img
    file_client = mmcv.FileClient(**{'backend':'disk'})
    target_img_path = 'GAN_attack/target_img.jpg'
    tar_img_bytes = file_client.get(target_img_path)
    tar_img = mmcv.imfrombytes(tar_img_bytes, flag='color')
    tar_img = mmcv.imnormalize(tar_img, np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]), True)
    tar_img = torch.from_numpy(tar_img.transpose(2, 0, 1)).cuda().unsqueeze(0)

    # prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    # results_records = [ [] for _ in range(len(results_records_iter_list))]
    # quires_records = [ [] for _ in range(len(results_records_iter_list))]
    gt_Anns = dataset.coco.imgToAnns
    cat2label = dataset.cat2label
    opt.batch_size = args.num_gpus
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0
        # visualizer.reset()
        results = []
        for index, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time


            # x_numpy = data['img'][0].numpy().transpose(0, 2, 3, 1)
            x_numpy = data['img'].data[0].numpy().transpose(0, 2, 3, 1)
            # ori_img = torch.FloatTensor(x_numpy.copy().transpose(0, 3, 1, 2))
            ori_img = torch.FloatTensor(x_numpy.transpose(0, 3, 1, 2)).cuda()
            lb, ub = x_numpy.min(), x_numpy.max()

            torch_size = torchvision.transforms.Resize((ori_img.shape[2], ori_img.shape[3]), interpolation=2)
            tar_img = torch_size(tar_img)

            # torch_size = torchvision.transforms.Resize((ori_img.shape[2], ori_img.shape[3]), interpolation=2)
            # # tar_img = tar_img.cpu().clone()
            # # tar_img = tar_img.squeeze(0)
            # # tar_img = transforms.ToPILImage(tar_img)
            # tar_img = torch_size(tar_img)

            with torch.no_grad():
                result = model(return_loss=False, rescale=False, attack_mode=True, attack_logistics=attack_logistics, **{'img_metas':data['img_metas'].data, 'img':data['img'].data})
                gt_bboxes, _, gt_labels = demo_utils.get_bboxes_scores_and_labels(result, ncls=len(dataset.CLASSES), to_Tensor=True)

                tar_features = model(return_loss=False, rescale=False, attack_mode=True, attack_logistics=attack_logistics, get_features=True, **{'img_metas':[{'file_path':target_img_path}], 'img':[tar_img]})

    
            # result = model(return_loss=False, rescale=False, attack_mode=True, attack_logistics=attack_logistics, **{'img_metas':data['img_metas'].data, 'img':data['img'].data})
            # gt_bboxes, _, gt_labels = demo_utils.get_bboxes_scores_and_labels(result, ncls=len(dataset.CLASSES), to_Tensor=True)

            # tar_features = model(return_loss=False, rescale=False, attack_mode=True, attack_logistics=attack_logistics, get_features=True, **{'img_metas':[{'file_path':target_img_path}], 'img':[tar_img]})
            
            ori_filename = data['img_metas'].data[0][0]['ori_filename']
            scale_factor = data['img_metas'].data[0][0]['scale_factor']


            gt_bboxes, gt_scores, gt_labels = demo_utils.get_gt_bboxes_scores_and_labels(gt_Anns, cat2label, ori_filename, scale_factor, len(dataset.CLASSES), scale_flag=True)
            gt_bboxes_tor = torch.Tensor(gt_bboxes).cuda()
            gt_labels_tor = torch.LongTensor(gt_labels).cuda()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            GAN_model.set_input(data, {'model':model, 'tar_features':tar_features, 'gt_bboxes':gt_bboxes_tor, 'gt_labels':gt_labels_tor, 'data':data})         # unpack data from dataset and apply preprocessing
            GAN_model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                GAN_model.compute_visuals()
                visualizer.display_current_results(GAN_model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = GAN_model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(data_loader.dataset), losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                GAN_model.save_networks(save_suffix)

            iter_data_time = time.time()
                        
            # save the result on the adversarial example
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **{'img_metas':data['img_metas'].data, 'img':[GAN_model.fake_B]})
                
                batch_size = len(result)
                out_dir = args.checkpoints_dir + '/' + args.name + '/detection_on_adv' 
                if out_dir:
                    if batch_size == 1 and isinstance(GAN_model.fake_B, torch.Tensor):
                        img_tensor = GAN_model.fake_B
                    else:
                        img_tensor = GAN_model.fake_B.data[0]
                    img_metas = data['img_metas'].data[0]
                    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                    assert len(imgs) == len(img_metas)

                    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                        if args.model == "CornerNet":
                            h, w, _ = img_meta['pad_shape']
                        else:
                            h, w, _ = img_meta['img_shape']
                        img_show = img[:h, :w, :]

                        ori_h, ori_w = img_meta['ori_shape'][:-1]
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                        if out_dir:
                            out_file = osp.join(out_dir, args.model, img_meta['filename'].split('/')[-1])
                        else:
                            out_file = None
                        # cv2.imwrite(out_file, img_show)
                        model.module.show_result(
                            img_show,
                            result[i],
                            show=True,
                            out_file=out_file,
                            score_thr=0.7)
            
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            results.extend(result)

            iter_data_time = time.time()

            # for _ in range(opt.batch_size):
            #     prog_bar.update()
        
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                # print('writing results to {args.out}')
                log.logger.info('writing results to {}'.format(args.out))
                mmcv.dump(results, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(results, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                print(dataset.evaluate(results, **eval_kwargs))

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            GAN_model.save_networks('latest')
            GAN_model.save_networks(epoch)

        GAN_model.update_learning_rate()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

if __name__ == "__main__":
    main()
