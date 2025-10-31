import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pickle
import torch


class VideoSrc:
    def __init__(self, width, height, fps=30):
        pass

    def start(self):
        raise NotImplementedError("not imple")

    def release(self):
        raise NotImplementedError("not imple")

    def get_fps(self):
        raise NotImplementedError("not imple")

    def read(self):
        raise NotImplementedError("not imple")

    def isOpened(self):
        raise NotImplementedError("not imple")


class RealSense(VideoSrc):
    def __init__(self, width, height, fps=15):
        self.type = "real_sense"
        # ストリームの設定
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.with_depth = True

        self.width = width
        self.height = height


        print("width: ", width)
        print("height: ", height)
        print("fps: ", fps)
        # カラーストリームを設定
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # デプスストリームを設定
        # self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        ## depthは640:480の4:3のみ
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.fps = fps

        ## preset
        PRESET = "High Density"
        # PRESET = "High Accuracy"
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            print('%02d: %s' %(i,visulpreset))
            if visulpreset == PRESET:
                depth_sensor.set_option(rs.option.visual_preset, i)
                print(f"set {PRESET}")

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def start(self):
        self.pipeline.start(self.config)

    def _get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("no frame")
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        resized_depth_image = cv2.resize(
            depth_image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

        return resized_depth_image, color_image

    def depth2colormap(self, depth):
        modified_array = depth.copy()
        threshold = 0
        modified_array[modified_array <= threshold] = threshold

        non_zero_values = modified_array[modified_array > 0]
        if non_zero_values.size == 0:
            return np.zeros_like(
                modified_array, dtype=np.uint8
            )  # Return zeros if all values are below threshold

        min_val = np.min(non_zero_values)
        max_val = np.max(non_zero_values)

        # Scale the remaining values to the range [0, 255]
        scaled_array = np.zeros_like(
            modified_array, dtype=np.float64
        )  # Initialize with zeros
        scaled_array[modified_array > 0] = (
            (non_zero_values - min_val) / (max_val - min_val)
        ) * 255

        array_2d = scaled_array.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(array_2d, cv2.COLORMAP_JET)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.08), cv2.COLORMAP_JET)
        # depth_colormap = cv2.applyColorMap(scaled_array, cv2.COLORMAP_JET)

        return depth_colormap

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.08), cv2.COLORMAP_JET)
        # return depth_colormap

    def release(self):
        self.pipeline.stop()

    def get_fps(self):
        return self.fps

    def isOpened(self):
        return True

    def read(self):
        depth, color = self._get_frames()
        return [depth, color]


class CV2Video(VideoSrc):
    def __init__(self, width, height, path=0, fps=30, enable_calib=False):
        self.type = "cv2"
        self.with_depth = False

        self.path = path
        self.vid = cv2.VideoCapture(int(self.path))

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.vid.set(cv2.CAP_PROP_FPS, fps)
        print("width: ", self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("height: ", self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("fps: ", self.vid.get(cv2.CAP_PROP_FPS))

        self.is_calibed = False

        if enable_calib:
            base_path = "./calibration"
            calib_file = f"{base_path}/calib_files/calibration.pkl"
            if os.path.isfile(calib_file):
                print("load:", calib_file)
                with open(calib_file, "rb") as f:
                    self.cameraMatrix, self.dist = pickle.load(f)
                self.is_calibed = True

    def start(self):
        if not self.vid.isOpened():
            print(f"Could not open video {self.path}")
            exit(-1)
        print("start cv2")

    def release(self):
        self.vid.release()

    def _calibed_read(self):
        ret, frame = self.vid.read()
        undistorted_frame = cv2.undistort(frame, self.cameraMatrix, self.dist, None)

    def get_fps(self):
        return self.vid.get(cv2.CAP_PROP_FPS)

    def read(self):
        if self.is_calibed:
            ret, frame = self.vid.read()
            undistorted_frame = cv2.undistort(frame, self.cameraMatrix, self.dist, None)
            return [ret, undistorted_frame]
        else:
            ret, frame = self.vid.read()
            return [ret, frame]

    def isOpened(self):
        return self.vid.isOpened()


class CV2VideoWithMiDas(CV2Video):
    def __init__(self, width, height, path=0, fps=30, enable_calib=False):
        super().__init__(width, height, path, fps, enable_calib)
        self.type = "cv2_midas"
        self.with_depth = True

        model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def read(self):
        _, frame = super().read()
        _frame = frame.copy()
        input_batch = self.transform(_frame).to(self.device)
        prediction = self.midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = prediction.cpu().detach().numpy().squeeze()
        # output = prediction.cpu().numpy().squeeze()
        return [output, frame]

    def depth2colormap(self, depth, bits=2):
        depth_min = depth.min()
        depth_max = depth.max()
        # max_val = (2**(8*bits))
        # if depth_max - depth_min > np.finfo("float").eps:
        #     out = max_val * (depth - depth_min) / (depth_max - depth_min)
        # else:
        #     out = np.zeros(depth.shape, dtype=depth.type)
        # if bits == 1:
        #     out = out.astype("uint8")
        # elif bits == 2:
        #     out = out.astype("uint16")
        out = 255 * (depth - depth_min) / (depth_max - depth_min)
        out = out.astype(np.uint8)
        return out
