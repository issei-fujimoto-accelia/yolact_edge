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
    score_threshold=0.3,
    trt_batch_size=2,
    use_fp16_tensorrt=True,
    disable_tensorrt=True
)
args = parser.parse_args()


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        torch.cuda.synchronize()

    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][:args.top_k]
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break
    
    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(net:Yolact, path:str):
    vid = cv2.VideoCapture(int(path))
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)
    if args.cuda:
        net = CustomDataParallel(net).cuda()
    else:
        logger.warning('CustomDataParallel not use cuda...')
        net = CustomDataParallel(net)

    if args.cuda:
        transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    else:
        logger.warning('DataParallel not use cuda...')
        transform = torch.nn.DataParallel(FastBaseTransform())
    frame_times = MovingAverage(400)
    fps = 0
    # The 0.8 is to account for the overhead of time.sleep
    frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)
    running = True
    
    frame_idx = 0
    every_k_frames = 5
    moving_statistics = {"conf_hist": []}

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        return [vid.read()[1] for _ in range(args.video_multiframe)]

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        nonlocal frame_idx
        with torch.no_grad():
            frames, imgs = inp
            if frame_idx % every_k_frames == 0 or cfg.flow.warp_mode == 'none':
                extras = {"backbone": "full", "interrupt": False, "keep_statistics": True,
                        "moving_statistics": moving_statistics}

                with torch.no_grad():
                    net_outs = net(imgs, extras=extras)

                moving_statistics["feats"] = net_outs["feats"]
                moving_statistics["lateral"] = net_outs["lateral"]

            else:
                extras = {"backbone": "partial", "interrupt": False, "keep_statistics": False,
                        "moving_statistics": moving_statistics}

                with torch.no_grad():
                    net_outs = net(imgs, extras=extras)
            frame_idx += 1

            return frames, net_outs["pred_outs"]

    def prep_frame(inp):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that 
    def play_video():
        nonlocal frame_buffer, running, video_fps

        video_frame_times = MovingAverage(100)
        frame_time_stabilizer = frame_time_target
        last_time = None
        stabilizer_step = 0.0005

        while running:
            frame_time_start = time.time()

            if not frame_buffer.empty():
                next_time = time.time()
                if last_time is not None:
                    video_frame_times.add(next_time - last_time)
                    video_fps = 1 / video_frame_times.get_avg()
                cv2.imshow(path, frame_buffer.get())
                last_time = next_time

            if cv2.waitKey(1) == 27: # Press Escape to close
                running = False

            buffer_size = frame_buffer.qsize()
            if buffer_size < args.video_multiframe:
                frame_time_stabilizer += stabilizer_step
            elif buffer_size > args.video_multiframe:
                frame_time_stabilizer -= stabilizer_step
                if frame_time_stabilizer < 0:
                    frame_time_stabilizer = 0

            new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)

            next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
            target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
            # This gives more accurate timing than if sleeping the whole amount at once
            while time.time() < target_time:
                time.sleep(0.001)


    extract_frame = lambda x, i: (x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    n_threads = len(sequence) + args.video_multiframe + 2
    n_threads = 4
    pool = ThreadPool(processes=n_threads)
    print("Number of threads: {}".format(n_threads))
    pool.apply_async(play_video)

    active_frames = []
    inference_times = []

    print()
    while vid.isOpened() and running:
        start_time = time.time()

        # Start loading the next frames from the disk
        next_frames = pool.apply_async(get_next_frame, args=(vid,))
        
        # For each frame in our active processing queue, dispatch a job
        # for that frame using the current function in the sequence
        for frame in active_frames:
            frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))
        
        # For each frame whose job was the last in the sequence (i.e. for all final outputs)
        for frame in active_frames:
            if frame['idx'] == 0:
                frame_buffer.put(frame['value'].get())

        # Remove the finished frames from the processing queue
        active_frames = [x for x in active_frames if x['idx'] > 0]

        # Finish evaluating every frame in the processing queue and advanced their position in the sequence
        for frame in list(reversed(active_frames)):
            frame['value'] = frame['value'].get()
            frame['idx'] -= 1

            if frame['idx'] == 0:
                # Split this up into individual threads for prep_frame since it doesn't support batch size
                active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
                frame['value'] = extract_frame(frame['value'], 0)

        
        # Finish loading in the next frames and add them to the processing queue
        active_frames.append({'value': next_frames.get(), 'idx': len(sequence)-1})
        
        # Compute FPS
        inference_time = time.time() - start_time
        frame_times.add(inference_time)
        inference_times.append(inference_time)
        fps = args.video_multiframe / frame_times.get_avg()
        np.save(args.video, np.asarray(inference_times))

        print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' % (fps, video_fps, frame_buffer.qsize()), end='')
    
    cleanup_and_exit()




if __name__ == '__main__':
    model_path = SavePath.from_str(args.trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
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
            if args.deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
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

