#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   iou_square_attack_demo.py
@Time    :   2021/01/12 18:57:09
@Author  :   Siyuan Liang
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import time
import sys
import os
sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.getcwd()+'/vertical_ss_tp')
import numpy as np
import pandas as pd
import torch
import argparse
import os
import numpy as np
import warnings
import cv2
# import utils
import time
import util as demo_utils
from datetime import datetime
import blackbox_attack.square_attack.utils as sq_utils
from blackbox_attack.utils.criterion import loss_fct, early_stop_crit_fct, early_stop_crit_fct_with_iou, loss_fct_with_FP

# os.environ['CUDA_VISIBLE_DEVICES']='0'
import mmcv
import os.path as osp
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet_v2200.apis import multi_gpu_test, single_gpu_test
from mmdet_v2200.core import encode_mask_results, tensor2imgs
from mmdet_v2200.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet_v2200.models import build_detector
from blackbox_attack.iou_ss_attack import IoUSSAttack 
from util import Logger


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
        '--vis_step_out_dir', default='/srv/hdd/results/od-black/mmdetection/sq_attack/det', help='directory where painted images will be saved')        
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

    # attack argument
    parser.add_argument('--attack', type=str, default='IoUsquare_linf', help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='/srv/hdd/results/od-black/mmdetection/sign_attack/adv', help='Experiment folder to store all output.')
    parser.add_argument('--n_iter', type=str, default='10000')
    parser.add_argument('--p', type=str, default='inf', choices=['inf', 'l2'])
    parser.add_argument('--targeted', default=False)
    parser.add_argument('--model', default='Faster-RCNN')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.0.05*4.52=0.22')
    parser.add_argument('--loss', type=str, default='cw_loss')
    parser.add_argument('--p_init', type=float, default=0.05)
    parser.add_argument('--attack_logistics', default=None)
    parser.add_argument('--vis_attack_step', default=None)
    parser.add_argument('--zeta', type=float, default='0.5')
    parser.add_argument('--lambda1', type=float, default='1.0')
    parser.add_argument('--patch_attack', type=str, default=None)
    parser.add_argument('--attack_parallel', type=str, default='1')
    parser.add_argument('--square_expansion', type=str, default=None)
    parser.add_argument('--square_init', type=str, default=None)
    parser.add_argument('--visual_iters_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)

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
    server_checkpoint_path = args.checkpoint.split('checkpoint')[0] + 'checkpoints/'
    if args.attack_logistics is not None:
        attack_logistics = bool(args.attack_logistics)
    else:
        attack_logistics = None

    if args.vis_attack_step is not None:
            vis_attack_step = bool(args.vis_attack_step)
    else:
        vis_attack_step = None
    
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

    cfg.data.test.test_mode = False
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
    if args.dataset_path is not None:
        new_anno_file = args.dataset_path + '/annotations/instances_val2017.json'
        new_img_prefix = args.dataset_path + '/val2017/'
        cfg.data.test['ann_file'] = new_anno_file
        cfg.data.test['img_prefix'] = new_img_prefix
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        # workers_per_gpu=0,
        dist=distributed,
        shuffle=False)
    log.logger.info('Load data from the path:{}'.format(cfg.data.test['img_prefix']))

    # build the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
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
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # prepare dataset (test dataset: 991 images)
    dataset = data_loader.dataset

    timestamp = str(datetime.now())[:-7]
    if args.attack.split('_')[0] == 'IoUzosignsgd':
        results_records_iter_list = demo_utils.results_records_iter_list_for_zosignsgd
    elif args.attack.split('_')[0] == 'IoUss' or args.attack.split('_')[0] == 'IoUssGT' \
        or args.attack.split('_')[0] == 'IoUssGTFPiter' or args.attack.split('_')[0] == 'IoUssGTTPADDFPiter':
        results_records_iter_list = demo_utils.results_records_iter_list_for_square
    elif args.attack.split('_')[0] == 'IoUsign':
        results_records_iter_list = demo_utils.results_records_iter_list_for_signhunter
    elif args.attack.split('_')[0] == 'IoUnes':
        results_records_iter_list = demo_utils.results_records_iter_list_for_nes

    # init log file
    # hps_str = '{} model={} dataset={} attack={} eps={} p={} n_iter={}'.format(timestamp, args.model, dataset, args.attack, args.eps, args.p, args.n_iter)
    # log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
    # # log = sq_utils.Logger(log_path)
    # log = sq_utils.Logger('')
    # log.print('All hps: {}'.format(hps_str))

    # init key points detector
    keypoints_models = {}
    if args.patch_attack is not None:
        for patch_attack in args.patch_attack.split(','):
            if patch_attack == 'bbox':
                log.logger.info('The keypoints get from the prediction box.')
                init_keypoints_model = None
            elif patch_attack == 'maskrcnn':
                keypoints_cfg = Config.fromfile('configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py')
                keypoints_checkpoint = server_checkpoint_path + 'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'        
                log.logger.info('The keypoints get from the detector: {}'.format(patch_attack))    
                init_keypoints_model = demo_utils.init_keypoints_model(keypoints_cfg, keypoints_checkpoint, args.fuse_conv_bn)  
            elif patch_attack == 'reppoints':
                keypoints_cfg = Config.fromfile('configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py')
                keypoints_checkpoint = server_checkpoint_path + 'reppoints_moment_r50_fpn_gn-neck+head_1x_coco_20200329-4b38409a.pth'
                log.logger.info('The keypoints get from the detector: {}'.format(patch_attack))    
                init_keypoints_model = demo_utils.init_keypoints_model(keypoints_cfg, keypoints_checkpoint, args.fuse_conv_bn)
            elif patch_attack == 'proposal':
                keypoints_cfg = Config.fromfile('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
                keypoints_checkpoint = server_checkpoint_path + 'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'        
                log.logger.info('The keypoints get from the detector: {}'.format(patch_attack))
                init_keypoints_model = demo_utils.init_keypoints_model(keypoints_cfg, keypoints_checkpoint, args.fuse_conv_bn)
            keypoints_models[patch_attack] = init_keypoints_model
    else:
        keypoints_models = None

    results = []
    total_quires = []
    total_times = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    results_records = [ [] for _ in range(len(results_records_iter_list))]
    quires_records = [ [] for _ in range(len(results_records_iter_list))]
    gt_Anns = dataset.coco.imgToAnns
    cat2label = dataset.cat2label
    if args.visual_iters_path is not None:
        args.visual_iters_path = args.visual_iters_path + args.model + '/'
    for index, data in enumerate(data_loader):
    #     continue
        # get the images bound and ori_img
        if isinstance(data['img'][0], torch.Tensor):
            x_numpy = data['img'][0].numpy().transpose(0, 2, 3, 1)
        else:
            x_numpy = data['img'][0].data[0].numpy().transpose(0, 2, 3, 1)
        # ori_img = torch.FloatTensor(x_numpy.copy().transpose(0, 3, 1, 2))
        ori_img = torch.FloatTensor(x_numpy.transpose(0, 3, 1, 2))
        lb, ub = x_numpy.min(), x_numpy.max() 
        
        ori_filename = data['img_metas'][0].data[0][0]['filename']
        if '/' in ori_filename:
            ori_filename = ori_filename.split('/')[-1]
        scale_factor = data['img_metas'][0].data[0][0]['scale_factor']
        gt_bboxes, gt_scores, gt_labels = demo_utils.get_gt_bboxes_scores_and_labels(gt_Anns, cat2label, ori_filename, scale_factor, len(dataset.CLASSES))
        if not args.targeted:
            labels_dic = demo_utils.get_untargeted_label_dic(gt_labels, len(dataset.CLASSES))
        # if attack_logistics is not None and attack_logistics:
        #     with torch.no_grad():
        #     # get logistic scores
        #         result = model(return_loss=False, rescale=True, attack_mode=True, attack_logistics=attack_logistics, **data)
        #     _, labels_clean = demo_utils.get_scores_and_labels(result, ncls=len(dataset.CLASSES))
        #     scores_logit_clean = result[2][0].cpu().detach().numpy()
        # else:
        #     with torch.no_grad():
        #     # get the bbox_scores:[n], bbox_lables[n] 
        #         result = model(return_loss=False, rescale=True, attack_mode=True, **data)
        #     # get softmax scores and labels
        #     scores_smooth_clean, labels_clean = demo_utils.get_scores_and_labels(result, ncls=len(dataset.CLASSES))
   
        # define the targeted label
        # labels_target  = sq_utils.random_classes_except_current(labels_clean, len(dataset.CLASSES)) if args.targeted else labels_clean
        # labels_target_onehot = sq_utils.dense_to_onehot(labels_target, n_cls=len(dataset.CLASSES))

        # define the attacker 
        # print(args.attack.split('_')[0])
        if args.visual_iters_path is not None:
            if not os.path.exists(args.visual_iters_path):
                os.makedirs(args.visual_iters_path)
            visual_iters_path = args.visual_iters_path + ori_filename.split('.')[0] + '_iters_process.jpg'
        else:
            visual_iters_path = None
        attacker = eval('IoUSSAttack')(
            max_loss_queries=int(args.n_iter),
            epsilon=args.eps,
            p=args.p,
            lb=lb,
            ub=ub,
            p_init=args.p_init,
            name=args.attack.split('_')[0],
            attack_model=model,
            attack_logistics=args.attack_logistics,
            attack_mode=True,
            loss=args.loss,
            targeted=args.targeted,
            ori_img=ori_img,
            model_name=args.model,
            zeta=args.zeta,
            lambda1=args.lambda1,
            patch_attack=args.patch_attack,
            keypoints_models=keypoints_models,
            attack_parallel=args.attack_parallel,
            square_expansion=args.square_expansion,
            square_init=args.square_init,
            visual_iters_path=visual_iters_path
        )
        
        time_start = time.time()

        # attack optimize
        if vis_attack_step is not None and vis_attack_step:
            queries, vis_result, results_records, queries_records = attacker.run(data, loss_fct_with_FP, early_stop_crit_fct_with_iou, vis_attack_step=vis_attack_step, results_records=results_records, quires_records=quires_records, gt_info=[gt_bboxes, gt_scores, gt_labels, labels_dic])
        else:
            queries = attacker.run(data, loss_fct, early_stop_crit_fct)
        # queries = [0.0]

        if attacker.visual_iters_path is not None:
            demo_utils.visual_iter_process(attacker, attacker.iters_list, attacker.losses_list, attacker.visual_iters_path)
        # save the adversarial example
        if args.exp_folder:
            if isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
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

                out_path = osp.join(args.exp_folder, args.model)

                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                if args.exp_folder:
                    out_file = osp.join(out_path, img_meta['filename'].split('/')[-1])
                else:
                    out_file = None
            

                cv2.imwrite(out_file, img_show)

        # save the opt result on the adversarial example
        if vis_attack_step is not None and vis_attack_step:
            color_map = ['red', 'blue', 'cyan', 'yellow', 'magenta']
            batch_size = 1
            vis_step_out_dir = args.vis_step_out_dir
            if vis_step_out_dir:
                out_file = osp.join(vis_step_out_dir, args.model, img_meta['filename'].split('/')[-1])

                # save the result on clean image and get clean img
                if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                        img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    if args.model == "CornerNet":
                        h, w, _ = img_meta['pad_shape']
                    else:
                        h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    img_show = model.module.show_result(
                        img_show,
                        vis_result[0][0],
                        bbox_color=color_map[0],
                        text_color=color_map[0],
                        score_thr=0.5)
                
                # draw the opt result on the adversarial image 
                for i in range(1, len(vis_result)-1):
                    img_show = model.module.show_result(
                        img_show,
                        vis_result[i][0],
                        bbox_color=color_map[i],
                        text_color=color_map[i],
                        score_thr=0.5
                    )

                # draw the last opt result and save the result
                model.module.show_result(
                    img_show,
                    vis_result[-1][0],
                    bbox_color=color_map[-1],
                    text_color=color_map[-1],
                    show=True,
                    out_file=out_file,
                    score_thr=0.5
                )
                    

        # save the result on the adversarial example
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            batch_size = len(result)
            out_dir = args.out_dir
            if out_dir:
                if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
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

        for _ in range(batch_size):
            prog_bar.update()

        time_total = time.time() - time_start
        log.logger.info('IMAGE {}: quries {}, times {}'.format(index, queries, time_total))
        total_quires.append(queries)
        total_times.append(time_total)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print('writing results to {args.out}')
            mmcv.dump(results, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(results, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'dynamic_intervals']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            for index in range(len(results_records)):
                results_record = results_records[index]
                if len(results_record) != 0:
                    # if isinstance(result[0], tuple):
                    #     result = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in result]
                    #     results.extend(result)
                    log.logger.info('-------------The attack {} iters result----------------'.format(str(results_records_iter_list[index])))
                    queries_record = quires_records[index]
                    mean_quries = np.mean(queries_record)
                    median_quries = np.median(queries_record)
                    
                    if isinstance(results_record[0], tuple):
                        results_record = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in results_record]
                    
                    from scipy import stats
                    # mode_quries = stats.mode(queries_record)[1][0]
                    log.logger.info('All {} images: mean quries {}, median_quries {}'.format(len(data_loader), mean_quries, median_quries)) 
                    log.logger.info(dataset.evaluate(results_record, **eval_kwargs))

                else:
                    log.logger.info('-------------The attack {} iters result is None----------------'.format(str(results_records_iter_list[index])))
                    log.logger.info('All {} images: mean quries {}, median_quries {}, mode_quries {}'.format(None, None, None, None)) 
    
    total_quires = np.array(total_quires).squeeze()
    total_times = np.array(total_times).squeeze()    
    mean_quries = np.mean(total_quires)
    median_quries = np.median(total_quires)
    
    from scipy import stats
    mode_quries = stats.mode(total_quires)[0][0]
    
    mean_times =  np.mean(total_times)
    log.logger.info('All {} images: mean quries {}, median_quries {}, mode_quries {}, mean times {}'.format(len(data_loader), mean_quries, median_quries, None, mean_times)) 



if __name__ == "__main__":
    main()