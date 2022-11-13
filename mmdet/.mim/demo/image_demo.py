import sys
import os
sys.path.insert(0, os.getcwd())
from argparse import ArgumentParser

from mmdet_v2200.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('save_path', help='Save file dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    file_name = args.img.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.save_path, file_name+'_'+str(args.score_thr)+'.jpg')
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr, save_path=save_path)


if __name__ == '__main__':
    main()
