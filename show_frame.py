import torch
import cv2
import time

from multiprocessing.pool import ThreadPool

# from multiprocessing import Queue
from queue import Queue
import numpy as np
from collections import defaultdict

from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils import timer
from yolact_edge.data import COLORS
from yolact_edge.yolact import Yolact
from yolact_edge.utils.functions import MovingAverage
from yolact_edge.utils.augmentations import FastBaseTransform
from yolact_edge.utils.augmentations import BaseTransform

## --- 出力するサイズ ---
## full hd 16:9
WIDTH = 1920
HEIGHT = 1080 - 100  ## プロジェクターだと高さがカットされてる
HEIGHT = 1080 - 48  ## プロジェクターだと高さがカットされてる

## hd 16:9
# WIDTH=1280
# HEIGHT=720

# WIDTH=int(WIDTH/2)
# HEIGHT=int(HEIGHT/2)

# WIDTH=1920
# HEIGHT=1080
## --- 出力するサイズ ---


## --- 取り込むカメラのサイズ ---
## 4:3
# CAM_WIDTH=640
# CAM_HEIGHT=480

## 16:9
# CAM_WIDTH = 640
# CAM_HEIGHT = 360
CAM_WIDTH = 1280
CAM_HEIGHT = 720
## --- 取り込むカメラのサイズ ---

FPS = 10
FPS = 5

# COLOR_SET = dict(
#     green=(0, 255, 0),
#     red=(0, 0, 255),
#     blue=(255, 0, 0),
#     pink=(255, 0, 165),
#     puple=(128, 0, 128),
# )

# COLORS = dict(
#     L4=COLOR_SET["green"],
#     L3=COLOR_SET["red"],
#     L2=COLOR_SET["blue"],
#     L=COLOR_SET["pink"],
#     M=COLOR_SET["puple"],
# )
COLORS = dict()

## --- size ---
# SIZE_SMALL= 5000
# SIZE_MIDIUM = 20000

# SIZE_SMALL= 500
# SIZE_MIDIUM = 1000
# SIZE_L4=else
SIZE_L3 = 1600
SIZE_L2 = 1300
SIZE_L = 1200
SIZE_M = 1000
## --- size ---

## -- convert to ---

## check_cam.pyで４点を選択する
# PTS = [[84, 35], [502, 49], [507, 280], [73, 279]]
PTS = [[141, 82], [424, 88], [431, 243], [138, 245]]
PTS = [[179, 78], [501, 53], [518, 238], [187, 254]]


## -- convert to ---


DOT_RAD = 5

TURNIP_LABEL_IDX = 0

color_cache = defaultdict(lambda: {})


def override_global_color_size_pts(sizes, colors, points):
    global COLORS
    global SIZE_M, SIZE_L, SIZE_L2, SIZE_L3
    global PTS

    SIZE_M = sizes[0]
    SIZE_L = sizes[1]
    SIZE_L2 = sizes[2]
    SIZE_L3 = sizes[3]

    for i, v in enumerate(["M", "L", "L2", "L3", "L4"]):
        _idx = i * 3
        # COLORS[v] = (colors[_idx], colors[_idx+1], colors[_idx+2])
        COLORS[v] = (colors[_idx + 2], colors[_idx + 1], colors[_idx])
        ## b,g,rの順番

    # COLORS["M"] = colors[4]
    # COLORS["L"] = colors[3]
    # COLORS["L2"] = colors[2]
    # COLORS["L3"] = colors[1]
    # COLORS["L4"] = colors[0]

    print("points", points[:])
    for i in range(4):
        PTS[i] = [points[i * 2], points[i * 2 + 1]]
    if all([v == 0 for v in points]):
        PTS = None

    # print("COLOR", COLORS)
    # print("sizes", sizes[:])
    print("PTS", PTS)


def prep_display(
    args,
    cfg,
    dets_out,
    img,
    h,
    w,
    undo_transform=True,
    class_color=False,
    mask_alpha=0.45,
):
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

    with timer.env("Postprocess"):
        keep_class_idx = TURNIP_LABEL_IDX if args.only_turnip else -1
        t = postprocess(
            dets_out,
            w,
            h,
            visualize_lincomb=args.display_lincomb,
            crop_masks=args.crop,
            score_threshold=args.score_threshold,
            keep_class_idx=keep_class_idx,
        )
        if args.cuda:
            torch.cuda.synchronize()

    with timer.env("Copy"):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][: args.top_k]
        classes, scores, boxes = [x[: args.top_k].cpu().numpy() for x in t[:3]]

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
                color = torch.Tensor(color).to(on_gpu).float() / 255.0
                color_cache[on_gpu][color_idx] = color
            return color

    def get_color_by_size(mask, on_gpu=True):
        size = torch.count_nonzero(mask)
        if size < SIZE_M:
            select_color = "M"
        elif size < SIZE_L:
            select_color = "L"
        elif size < SIZE_L2:
            select_color = "L2"
        elif size < SIZE_L3:
            select_color = "L3"
        else:
            select_color = "L4"
        color = COLORS[select_color]
        if on_gpu:
            color = torch.Tensor(color).to("cuda").float() / 255.0
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
        colors = torch.cat(
            [
                get_color_by_size(masks[j], on_gpu=img_gpu.device.index == 0).view(
                    1, 1, 1, 3
                )
                for j in range(num_dets_to_consider)
            ],
            dim=0,
        )
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[: (num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if (
        args.display_text
        or args.display_bboxes
        or args.display_size
        or args.display_dot
    ):
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            # color = get_color(j)
            color = get_color_by_size(masks[j], on_gpu=False)

            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_size:
                size = torch.count_nonzero(masks[j])
                text_str = "size: %d" % size

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_pt = (x1, y1 - 3)
                # text_color = [255, 255, 255]
                text_color = [0, 0, 0]

                cv2.putText(
                    img_numpy,
                    text_str,
                    text_pt,
                    font_face,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

            if args.display_dot:
                _x = int((x2 - x1) // 2 + x1)
                _y = int((y2 - y1) // 2 + y1)
                cv2.circle(
                    img_numpy,
                    center=(_x, _y),
                    radius=DOT_RAD,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_4,
                    shift=0,
                )

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                size = torch.count_nonzero(masks[j])
                # text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                text_str = (
                    "%s, score: %.2f, size: %d" % (_class, score, size)
                    if args.display_scores
                    else _class
                )

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(
                    text_str, font_face, font_scale, font_thickness
                )[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(
                    img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1
                )
                cv2.putText(
                    img_numpy,
                    text_str,
                    text_pt,
                    font_face,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

    # if args.display_ajuster:
    #    _h, _w, _ = img_numpy.shape
    #    cv2.circle(img_numpy, (20, 20), 40, COLORS["red"], thickness=-1)
    return img_numpy


def prep_display_for_ui(
    args,
    cfg,
    dets_out,
    img,
    raw_frame,
    h,
    w,
    undo_transform=True,
    class_color=False,
    mask_alpha=0.45,
    depth=None,
):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    show_img = torch.zeros(img.shape, dtype=torch.int8)
    show_img = show_img + 255
    if args.cuda:
        show_img = show_img.cuda()

    if undo_transform:
        # preview_img = undo_image_transformation(raw_frame, w, h)
        # preview_img_gpu = torch.Tensor(preview_img).cuda()
        preview_img_gpu = undo_image_transformation(img, w, h)

        show_img = undo_image_transformation(show_img, w, h)
        show_img_gpu = torch.Tensor(show_img).cuda()
    else:
        preview_img_gpu = img / 255.0
        show_img_gpu = show_img / 255.0
        h, w, _ = img.shape

    preview_img_numpy = (preview_img_gpu * 255).byte().cpu().numpy()
    # preview_img_numpy = raw_frame / 255.0
    # preview_img_numpy = raw_frame

    with timer.env("Postprocess"):
        keep_class_idx = TURNIP_LABEL_IDX if args.only_turnip else -1
        t = postprocess(
            dets_out,
            w,
            h,
            visualize_lincomb=args.display_lincomb,
            crop_masks=args.crop,
            score_threshold=args.score_threshold,
            keep_class_idx=keep_class_idx,
        )
        if args.cuda:
            torch.cuda.synchronize()

    with timer.env("Copy"):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][: args.top_k]
        classes, scores, boxes = [x[: args.top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return preview_img_numpy * 255, (show_img_gpu * 255).byte().cpu().numpy()

    def get_color_by_size(mask, on_gpu=True):
        size = torch.count_nonzero(mask)
        if size < SIZE_M:
            select_color = "M"
        elif size < SIZE_L:
            select_color = "L"
        elif size < SIZE_L2:
            select_color = "L2"
        elif size < SIZE_L3:
            select_color = "L3"
        else:
            select_color = "L4"
        color = COLORS[select_color]
        if on_gpu:
            color = torch.Tensor(color).to("cuda").float() / 255.0
        ## color_cache[on_gpu][color_idx] = color
        return color

    show_img_numpy = (show_img_gpu * 255).byte().cpu().numpy()
    # preview_img_numpy = (preview_img_gpu * 255).byte().cpu().numpy()

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        # color = get_color(j)
        color = get_color_by_size(masks[j], on_gpu=False)

        _x = int((x2 - x1) // 2 + x1)
        _y = int((y2 - y1) // 2 + y1)
        cv2.circle(
            show_img_numpy,
            center=(_x, _y),
            radius=DOT_RAD,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_4,
            shift=0,
        )

        ## bbox
        cv2.rectangle(preview_img_numpy, (x1, y1), (x2, y2), color, 1)

        ## size
        size = torch.count_nonzero(masks[j])
        text_str = "size: %d" % size

        ## depth
        if depth:
            depth_value = min(
                [
                    depth[_y + 2][_x],
                    depth[_y - 2][_x],
                    depth[_y][_x + 2],
                    depth[_y][_x - 2],
                    depth[_y][_x],
                ]
            )  # 周辺の最小値
            # depth_value = np.mean([depth[_y+2][_x], depth[_y-2][_x], depth[_y][_x+2], depth[_y][_x-2], depth[_y][_x]]) # 周辺の平均
            text_str += ",depth %.2f" % depth_value

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_pt = (x1, y1 - 3)
        # text_color = [255, 255, 255]
        text_color = [0, 0, 0]
        cv2.putText(
            preview_img_numpy,
            text_str,
            text_pt,
            font_face,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return preview_img_numpy, show_img_numpy


class CustomDataParallel(torch.nn.DataParallel):
    """A Custom Data Parallel class that properly gathers lists of dictionaries."""

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def show_ajuster(vid, zoom_rate):
    ret, frame = vid.read()
    _h, _w, _ = frame.shape
    _r = 40
    cv2.circle(
        frame, (int(_w / 2), int(_h / 2)), _r, COLORS["red"], thickness=-1
    )  ## 真ん中
    cv2.circle(frame, (_w - _r, _r), _r, COLORS["red"], thickness=-1)  ## 右上
    cv2.circle(frame, (0 + _r, 0 + _r), _r, COLORS["red"], thickness=-1)  ## 左上
    cv2.circle(frame, (_w - _r, _h - _r), _r, COLORS["red"], thickness=-1)  ## 右下
    cv2.circle(frame, (_r, _h - _r), _r, COLORS["red"], thickness=-1)  ## 左下
    frame = _zoom(frame, zoom_rate)
    cv2.imshow("frame", frame)


def _zoom(frame, rate=1.0):
    """
    frame: np.array
    rate: zoom rate. 1.0より大きい値を期待、1.0より小さい場合うまく動かない

    画面の中央を切り取り拡大する
    """
    h, w, _ = frame.shape
    crop_h = int((h - h / rate) / 2)
    crop_h_to = int(h / rate)
    crop_w = int((w - w / rate) / 2)
    crop_w_to = int(w / rate)
    return cv2.resize(
        frame[crop_h:crop_h_to, crop_w:crop_w_to, :], dsize=(WIDTH, HEIGHT)
    )


def evalvideo_show_frame(
    net: Yolact,
    path: str,
    cuda: bool,
    args,
    cfg,
    set_frame=None,
    sizeArray=None,
    colorArray=None,
    pointArray=None,
):
    # vid = cv2.VideoCapture(int(path))
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # vid.set(cv2.CAP_PROP_FPS, FPS)

    # print("width: ", vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print("height: ", vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print("fps: ", vid.get(cv2.CAP_PROP_FPS))

    # if not vid.isOpened():
    #     print('Could not open video "%s"' % path)
    #     exit(-1)
    # vid = CV2Video(CAM_WIDTH, CAM_HEIGHT, fps=FPS, path=path)

    # vid = CV2VideoWithMiDas(CAM_WIDTH, CAM_HEIGHT, fps=FPS, path=path)
    vid = RealSense(CAM_WIDTH, CAM_HEIGHT, FPS)
    vid.start()

    net = CustomDataParallel(net)
    if cuda:
        net.cuda()

    transform = torch.nn.DataParallel(FastBaseTransform(cuda))

    if cuda:
        transform.cuda()

    frame_times = MovingAverage(400)
    fps = 0
    # The 0.8 is to account for the overhead of time.sleep
    frame_time_target = 1 / vid.get_fps()
    running = True

    frame_idx = 0
    every_k_frames = 5
    moving_statistics = {"conf_hist": []}

    def cleanup_and_exit():
        pool.terminate()
        vid.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for _ in range(args.video_multiframe):
            frames.append(vid.read())
        return frames

    def convert(frame):
        """
        四角形を長方形に変換する
        """
        if PTS is None:
            return frame
        # h, w = frame.shape
        h = frame.shape[0]
        w = frame.shape[1]
        h, w, _ = frame.shape
        from_pts = np.array(PTS)
        dst_pts = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
        )
        M = cv2.getPerspectiveTransform(from_pts.astype("float32"), dst_pts)
        t_frame = cv2.warpPerspective(frame, M, (w, h))
        return t_frame

    def transform_frame(frames):
        raw_frames = frames.copy()
        frames = [convert(frame) for frame in frames]
        with torch.no_grad():
            if cuda:
                frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            else:
                frames = [torch.from_numpy(frame).float() for frame in frames]
            return raw_frames, frames, transform(torch.stack(frames, 0))

    def transform_frame_color(frames):
        """
        frame[0] = civ_ret
        frame[1] = color image
        """
        raw_frames = frames.copy()
        frames = [convert(frame[1]) for frame in frames]
        with torch.no_grad():
            if cuda:
                frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            else:
                frames = [torch.from_numpy(frame).float() for frame in frames]
            return raw_frames, frames, transform(torch.stack(frames, 0)), None

    def transform_frame_with_depth(frames):
        """
        frame[0] = depth image
        frame[1] = color image
        """
        raw_frames = frames.copy()
        depth_frames = [convert(frame[0]) for frame in frames]
        color_frames = [convert(frame[1]) for frame in frames]

        with torch.no_grad():
            if cuda:
                color_frames = [
                    torch.from_numpy(frame).cuda().float() for frame in color_frames
                ]
            else:
                color_frames = [
                    torch.from_numpy(frame).float() for frame in color_frames
                ]
            return (
                raw_frames,
                color_frames,
                transform(torch.stack(color_frames, 0)),
                depth_frames,
            )

    def eval_network(inp):
        nonlocal frame_idx
        with torch.no_grad():
            raw_frames, frames, imgs, depth_frame = inp
            if frame_idx % every_k_frames == 0 or cfg.flow.warp_mode == "none":
                extras = {
                    "backbone": "full",
                    "interrupt": False,
                    "keep_statistics": True,
                    "moving_statistics": moving_statistics,
                }

                with torch.no_grad():
                    net_outs = net(imgs, extras=extras)

                moving_statistics["feats"] = net_outs["feats"]
                moving_statistics["lateral"] = net_outs["lateral"]

            else:
                extras = {
                    "backbone": "partial",
                    "interrupt": False,
                    "keep_statistics": False,
                    "moving_statistics": moving_statistics,
                }

                with torch.no_grad():
                    net_outs = net(imgs, extras=extras)
            frame_idx += 1
            return raw_frames, frames, net_outs["pred_outs"], depth_frame

    def prep_frame(inp):
        with torch.no_grad():
            raw_frame, frame, preds, depth_image = inp
            return (
                prep_display_for_ui(
                    args,
                    cfg,
                    preds,
                    frame,
                    raw_frame,
                    None,
                    None,
                    undo_transform=False,
                    class_color=True,
                    mask_alpha=1,
                    depth=depth_image,
                ),
                depth_image,
            )

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
                # _resized_frame = _zoom(frame_buffer.get(), args.zoom_rate)
                # preview_img, show_img = frame_buffer.get()
                show_img, _ = frame_buffer.get()
                show_frame = cv2.resize(show_img, dsize=(WIDTH, HEIGHT))
                cv2.imshow("frame", show_frame)
                last_time = next_time
            ## frame更新のタイミングで上書きするが更新頻度が高すぎる
            override_global_color_size_pts(sizeArray, colorArray, pointArray)

            if args.display_ajuster:
                show_ajuster(vid, args.zoom_rate)

            if cv2.waitKey(1) == 27:  # Press Escape to close
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
            target_time = (
                frame_time_start + next_frame_target - 0.001
            )  # Let's just subtract a millisecond to be safe
            # This gives more accurate timing than if sleeping the whole amount at once
            while time.time() < target_time:
                time.sleep(0.001)

    def extract_frame(x, i):
        if vid.with_depth:
            color, preds, depth = x
            if preds[i] is None:
                return (color[i], [preds[i]], depth[i])
            else:
                return (color[i].to(preds[i]["box"].device), [preds[i]], depth[i])
        else:
            return (
                x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]["box"].device),
                [x[1][i]],
                None,
            )

    # extract_frame = lambda x, i: (
    #     x[0][i],
    #     x[1][i] if x[2][i] is None else x[1][i].to(x[2][i]["box"].device),
    #     [x[2][i]],
    # )

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print("Initializing model... ", end="")
    _transform_frame = (
        transform_frame_with_depth if vid.with_depth else transform_frame_color
    )
    eval_network(_transform_frame(get_next_frame(vid)))
    print("Done.")

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    n_threads = len(sequence) + args.video_multiframe + 2
    # n_threads = 4
    pool = ThreadPool(processes=n_threads)
    print("Number of threads: {}".format(n_threads))
    pool.apply_async(play_video)

    active_frames = []
    inference_times = []

    while vid.isOpened() and running:
        start_time = time.time()

        # Start loading the next frames from the disk
        next_frames = pool.apply_async(get_next_frame, args=(vid,))

        # For each frame in our active processing queue, dispatch a job
        # for that frame using the current function in the sequence
        for frame in active_frames:
            frame["value"] = pool.apply_async(
                sequence[frame["idx"]], args=(frame["value"],)
            )

        # For each frame whose job was the last in the sequence (i.e. for all final outputs)
        for frame in active_frames:
            if frame["idx"] == 0:
                preview, show = frame["value"].get()
                set_frame(preview)
                frame_buffer.put(show)

        # Remove the finished frames from the processing queue
        active_frames = [x for x in active_frames if x["idx"] > 0]

        # Finish evaluating every frame in the processing queue and advanced their position in the sequence
        for frame in list(reversed(active_frames)):
            frame["value"] = frame["value"].get()
            frame["idx"] -= 1

            if frame["idx"] == 0:
                # Split this up into individual threads for prep_frame since it doesn't support batch size
                active_frames += [
                    {"value": extract_frame(frame["value"], i), "idx": 0}
                    for i in range(1, args.video_multiframe)
                ]
                frame["value"] = extract_frame(frame["value"], 0)

        # Finish loading in the next frames and add them to the processing queue
        active_frames.append({"value": next_frames.get(), "idx": len(sequence) - 1})

        # Compute FPS
        inference_time = time.time() - start_time
        frame_times.add(inference_time)
        inference_times.append(inference_time)
        fps = args.video_multiframe / frame_times.get_avg()
        # np.save(args.video, np.asarray(inference_times))
        # np.save("./tmp/tmp", np.asarray(inference_times))

        print(
            "\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    "
            % (fps, video_fps, frame_buffer.qsize()),
            end="",
        )

    cleanup_and_exit()
