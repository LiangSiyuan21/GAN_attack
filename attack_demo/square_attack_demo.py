# coding:utf-8
# author:liangsiyuan
# @Time :2020/11/3  3:22 PM

import argparse
import os
import numpy as np
import warnings
import cv2
import utils
import time
import util as demo_utils
from datetime import datetime
import blackbox_attack.square_attack.utils as sq_utils

os.environ['CUDA_VISIBLE_DEVICES']='1'
import mmcv
import os.path as osp
import sys
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
from blackbox_attack.square_attack import square_attack as sq

def label_smooth(scores, labels, n_cls):
    result = np.ones([len(labels), n_cls])
    for i in range(len(labels)):
        result[i] = (1.0 - scores[i]) / (n_cls - 1) * result[i]
    result[np.arange(len(labels)), labels] = scores
    return result

    # y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    # y_test_onehot[np.arange(len(y_test)), y_test] = True
    # return y_test_onehot

def parse_args():
    # detector argument
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    parser.add_argument('--vis_step_out_dir', default=None)
    parser.add_argument('--vis_attack_step', default=None)

    # attack argument
    parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2'], help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='/srv/hdd/results/od-black/mmdetection/sq_attack/adv', help='Experiment folder to store all output.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--n_iter', type=str, default='1000')
    parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
    parser.add_argument('--model', default='Faster-RCNN')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.0.05*4.52=0.22')
    parser.add_argument('--loss', type=str)
    parser.add_argument('--attack_logistics', default=None)
    args = parser.parse_args()
    args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
    
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
    
    if args.vis_attack_step is not None:
        vis_attack_step = bool(args.vis_attack_step)
    else:
        vis_attack_step = None

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
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
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

    # initilaze object detector
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # prepare dataset (test dataset: 991 images)
    dataset = data_loader.dataset

    timestamp = str(datetime.now())[:-7]

    # init log file
    hps_str = '{} model={} dataset={} attack={} eps={} p={} n_iter={}'.format(timestamp, args.model, dataset, args.attack, args.eps, args.p, args.n_iter)
    log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
    log = sq_utils.Logger(log_path)
    log = sq_utils.Logger('')
    # log.print('All hps: {}'.format(hps_str))
    
    results = []
    total_quires = []
    total_times = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    # optimize the clean images by square attack
    for index, data in enumerate(data_loader):
        # if i != 907:
        #     continue
        if attack_logistics is not None and attack_logistics:
            with torch.no_grad():
            # get logistic scores
                result = model(return_loss=False, rescale=True, attack_mode=True, attack_logistics=attack_logistics, **data)
            _, labels_clean = demo_utils.get_scores_and_labels(result, ncls=len(dataset.CLASSES))
            scores_logit_clean = result[2][0].cpu().detach().numpy()
        else:
            with torch.no_grad():
            # get the bbox_scores:[n], bbox_lables[n] 
                result = model(return_loss=False, rescale=True, attack_mode=True, **data)    
            # get softmax scores and labels
            scores_smooth_clean, labels_clean = demo_utils.get_scores_and_labels(result, ncls=len(dataset.CLASSES))

        # define the attack types
        square_attack = sq.square_attack_linf if args.attack == 'square_linf' else sq.square_attack_l2
        # define the targeted label
        labels_target  = sq_utils.random_classes_except_current(labels_clean, len(dataset.CLASSES)) if args.targeted else labels_clean
        labels_target_onehot = sq_utils.dense_to_onehot(labels_target, n_cls=len(dataset.CLASSES))

        time_start = time.time()
        
        # attack optimize
        if vis_attack_step is not None and vis_attack_step:
            queries, vis_result = square_attack(model, data, (scores_smooth_clean, labels_clean), args.eps, int(args.n_iter),
                                     args.p, args.targeted, args.loss, ncls=len(dataset.CLASSES), vis_attack_step=vis_attack_step)
        else:
            if attack_logistics is not None and attack_logistics:
                queries = square_attack(model, data, (scores_logit_clean, labels_clean), args.eps, int(args.n_iter),
                                        args.p, args.targeted, args.loss, attack_logistics=attack_logistics, ncls=len(dataset.CLASSES))
            else:
                queries = square_attack(model, data, (scores_smooth_clean, labels_clean), args.eps, int(args.n_iter),
                                        args.p, args.targeted, args.loss, ncls=len(dataset.CLASSES))            
        
        # save the adversarial example    
        if args.exp_folder:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
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
        log.print('IMAGE {}: quries {}, times {}'.format(index, queries, time_total))
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
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(results, **eval_kwargs))


    mean_quries = np.array(total_quires).sum(0) / len(data_loader)
    mean_times =  np.array(total_times).sum(0) / len(data_loader)
    log.print('{} images: mean quries {}, mean times {}'.format(len(data_loader), mean_quries, mean_times))

if __name__ == '__main__':
    main()
