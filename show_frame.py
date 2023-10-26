import torch
import cv2
import time

from multiprocessing.pool import ThreadPool
from queue import Queue
import numpy as np
from collections import defaultdict

from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils import timer
from yolact_edge.data import  COLORS
from yolact_edge.yolact import Yolact
from yolact_edge.utils.functions import MovingAverage
from yolact_edge.utils.augmentations import FastBaseTransform
from yolact_edge.utils.augmentations import BaseTransform

COLORS = dict(
    blue=(255, 0, 0),
    red=(0, 0, 255),
    green=(0, 255, 0),
)
## (255, 0, 165), # puple  
## (128, 0, 128), # puple


HEIGHT=1080
WIDTH=1920

HEIGHT=1080/2
WIDTH=1920/2

FPS=20

SIZE_SMALL= 5000
SIZE_MIDIUM = 20000

DOT_RAD=30

TURNIP_LABEL_IDX = 0

color_cache = defaultdict(lambda: {})
def prep_display(args, cfg, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if args.hide_back:
        img = torch.zeros(img.shape, dtype=torch.int8)
        img = img + 255
        if args.cuda:
            img = img.cuda()

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        keep_class_idx = TURNIP_LABEL_IDX if args.only_turnip else -1        
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold,
                                        keep_class_idx    = keep_class_idx
                                        )
        if args.cuda:
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
            k = list(COLORS.keys())[color_idx]  
            color = COLORS[k]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
        
    def get_color_by_size(mask, on_gpu=True):
        size = torch.count_nonzero(mask)
        if size < SIZE_SMALL:
          select_color = "blue"
        elif size < SIZE_MIDIUM:
          select_color = "red"
        else:  
          select_color = "green"
        color = COLORS[select_color]
        if on_gpu is not None:
          color = torch.Tensor(color).to(on_gpu).float() / 255.
          ## color_cache[on_gpu][color_idx] = color
        return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        # colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        ## color by size
        colors = torch.cat([get_color_by_size(masks[j], on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        
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
    
    if args.display_text or args.display_bboxes or args.display_size:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]         

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_size:
                size = torch.count_nonzero(masks[j])
                text_str = 'size: %d' % size

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_pt = (x1, y1 - 3)
                # text_color = [255, 255, 255]
                text_color = [0, 0, 0]
                
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)                
                if args.display_dot:
                    _x = int((x2 - x1) // 2 + x1)
                    _y = int((y2 - y1) // 2 + y1)
                    cv2.circle(img_numpy,
                        center=(_x, _y),
                        radius=DOT_RAD,
                        color=color,
                        thickness=-1,
                        lineType=cv2.LINE_4,
                        shift=0
                    )


            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                size = torch.count_nonzero(masks[j])
                # text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                text_str = '%s, score: %.2f, size: %d' % (_class, score, size) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if args.display_ajuster:        
        _h, _w, _ = img_numpy.shape
        cv2.circle(img_numpy, (20, 20), 40, COLORS["red"], thickness=-1)

    return img_numpy


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def show_ajuster(vid):
    ret, frame = vid.read()
    _h, _w, _ = frame.shape
    _r = 40
    cv2.circle(frame, (int(_w/2), int(_h/2)), _r, COLORS["red"], thickness=-1) ## 真ん中
    cv2.circle(frame, (_w-_r, _r), _r, COLORS["red"], thickness=-1) ## 右上
    cv2.circle(frame, (0+_r, 0+_r), _r, COLORS["red"], thickness=-1) ## 左上     
    cv2.circle(frame, (_w-_r, _h-_r), _r, COLORS["red"], thickness=-1) ## 右下
    cv2.circle(frame, (_r, _h-_r), _r, COLORS["red"], thickness=-1) ## 左下
    cv2.imshow("frame", frame)


def evalvideo_show_frame(net:Yolact, path:str, cuda: bool, args, cfg):    
    vid = cv2.VideoCapture(int(path))
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    vid.set(cv2.CAP_PROP_FPS, FPS)

    print("width: ", cv2.CAP_PROP_FRAME_WIDTH)
    print("height: ", cv2.CAP_PROP_FRAME_HEIGHT)

    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)
    
    net = CustomDataParallel(net)
    if cuda:
        net.cuda()
    
    transform = torch.nn.DataParallel(FastBaseTransform(cuda))
    
    if cuda:
        transform.cuda()
        
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
          if cuda:
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
          else:
            frames = [torch.from_numpy(frame).float() for frame in frames]
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
            return prep_display(args, cfg, preds, frame, None, None, undo_transform=False, class_color=True, mask_alpha=1)

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
                # cv2.imshow(path, frame_buffer.get())
                cv2.imshow("frame", frame_buffer.get())
                last_time = next_time
            if args.display_ajuster:
                show_ajuster(vid)
                
            if cv2.waitKey(1) == 27: # Press Escape to close
                running = False

            buffer_size = frame_buffer.qsize()
            if buffer_size < args.video_multiframe:
                frame_time_stabilizer += stabilizer_step
            elif buffer_size > args.video_multiframe:
                frame_time_stabilizer -= stabilizer_step
                if frame_time_stabilizer < 0:
                    frame_time_stabilizer = 0

            # new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
            new_target = frame_time_stabilizer

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
        # np.save(args.video, np.asarray(inference_times))
        # np.save("./tmp/tmp", np.asarray(inference_times))

        print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' % (fps, video_fps, frame_buffer.qsize()), end='')
    
    cleanup_and_exit()

