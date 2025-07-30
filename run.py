import torch
import cv2
import time
import os

from multiprocessing.pool import ThreadPool
from queue import Queue
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import argparse
from collections import defaultdict

from yolact_edge.data import cfg, set_cfg
from yolact_edge.utils import timer
from yolact_edge.yolact import Yolact
from yolact_edge.utils.functions import MovingAverage
from yolact_edge.utils.augmentations import FastBaseTransform
from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.data import  COLORS
from yolact_edge.utils.functions import SavePath

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--trained_model',
                        default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
parser.add_argument('--trt_batch_size', default=1, type=int,
                        help='Maximum batch size to use during TRT conversion. This has to be greater than or equal to the batch size the model will take during inferece.')
parser.add_argument('--disable_tensorrt', default=False, dest='disable_tensorrt', action='store_true',
                        help='Don\'t use TensorRT optimization when specified.')
parser.add_argument('--use_fp16_tensorrt', default=False, dest='use_fp16_tensorrt', action='store_true',
                        help='This replaces all TensorRT INT8 optimization with FP16 optimization when specified.')
parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
parser.add_argument('--config', required=True, help='The config object to use.')                        
parser.add_argument('--verbose', default=True, type=str2bool,
                        help='show debug print')

## used from yolact_edge/yolact.py
parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true',
                        help='Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
parser.add_argument('--drop_weights', default=None, type=str,
                        help='Drop specified weights (split by comma) from existing model.')
parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true',
                        help='[Deprecated] Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')

parser.set_defaults(
    trained_model="../turnip_detect/weights/yolact_edge_mobilenetv2_54_800000.pth",
    top_k=30,
    cuda=False,
    video=0,
    video_multiframe=2,
    score_threshold=0.3,
    trt_batch_size=2,
    use_fp16_tensorrt=True,
    disable_tensorrt=True,
    crop=True,
    fast_eval=True
)
args = parser.parse_args()

color_cache = defaultdict(lambda: {})


if __name__ == '__main__':
    model_path = SavePath.from_str(args.trained_model)
    set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    from yolact_edge.utils.logging_helper import setup_logger
    setup_logger(logging_level=logging.INFO)
    logger = logging.getLogger("yolact.eval")

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            #if args.deterministic:
            #    cudnn.deterministic = True
            #    cudnn.benchmark = False
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    logger.info('Loading model...')
    net = Yolact(training=False)
    net.load_weights(args.trained_model, args=args)
    net.eval()
    logger.info('Model loaded.')

    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    evalvideo(net, args.video)

