
"""
【K230/CanMV 初学者友好注释版】
—— 用摄像头和触摸屏做一个“可用手指/触摸操作的简易计算器”

【本脚本适合谁？】
    - 你只会最基础的Python（print/变量/循环），但想学K230/AI摄像头/触摸屏。
    - 你没嵌入式/硬件基础，想看“每一步都解释清楚”的代码。

【本脚本能做什么？】
 1. 打开K230摄像头，实时显示画面。
 2. 在画面上叠加一层“半透明的计算器按钮UI”。
 3. 支持两种输入方式：
        - 触摸屏直接点按钮（最推荐，最稳定）。
        - 用摄像头识别“手指”或“色块”，隔空点按钮。
 4. 实现一个能加减乘除的简易计算器。

【核心原理简述】
    - 摄像头采集画面，作为底图。
    - 触摸屏/AI手势/色块追踪，作为输入。
    - OSD（叠加层）负责把按钮、文字、光标画到屏幕上。
    - 主循环每帧采集输入→识别→绘制UI→显示。

【初学者必读】
    - 你可以把“摄像头画面”理解为一张大背景。
    - “UI图层”就是一张透明纸，画按钮/文字/光标。
    - 触摸屏输入最可靠，AI/色块追踪是“备用方案”。
    - 代码每一段都加了详细注释，遇到不懂的地方直接看注释。
    - 你可以只看“每一部分的开头注释”，先理解整体流程。
"""

# https://wiki.lckfb.com/zh-hans/lushan-pi-k230/
# 这个项目基于立创开发板 K230，运行环境是 MicroPython / CanMV。

from media.sensor import *
from media.display import *
from media.media import *
from machine import TOUCH
import image
import time
import gc
import os

try:
    import nncase_runtime as nn
    import ulab.numpy as np
    import aicube
    AI_RUNTIME_AVAILABLE = True
except Exception:
    nn = None
    np = None
    aicube = None
    AI_RUNTIME_AVAILABLE = False

# 屏幕分辨率。
# 这里的 UI 绘制、触摸坐标、相机输出尺寸，都会尽量和这个分辨率保持一致，
# 这样最容易理解，也最不容易出现“坐标对不上”的问题。
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
# AI 输入分辨率可适当降低，先优先稳定性；若你后续想更高精度可改回 800x480。
AI_FRAME_WIDTH = 640
AI_FRAME_HEIGHT = 360

HAND_DET_MODEL = "/sdcard/examples/kmodel/hand_det.kmodel"
HAND_KP_MODEL = "/sdcard/examples/kmodel/handkp_det.kmodel"
HAND_LABELS = ["hand"]
HAND_DET_INPUT_SIZE = [512, 512]
HAND_KP_INPUT_SIZE = [256, 256]
HAND_ANCHORS = [26, 27, 53, 52, 75, 71, 80, 99, 106, 82, 99, 134, 140, 113, 161, 172, 245, 276]
POINTER_GESTURES = set(["one", "gun", "yeah"])
POINTER_SMOOTH_ALPHA = 0.80

# 调试与稳定性参数
DEBUG_RUNTIME = True
DEBUG_PRINT_INTERVAL_MS = 1000
FORCE_GC_INTERVAL_FRAMES = 20
SHRINK_POOL_INTERVAL_FRAMES = 120
LOOP_HANG_WARN_MS = 800
MEM_GUARD_FREE_MIN = 2600000
KP_INFER_INTERVAL = 1
IDLE_SLEEP_S = 0.01

# 下面这些常量决定了“按钮网格”的布局。
# 这里的思路是：
# 1. 先确定总共 4 列 4 行。
# 2. 再确定整个按钮区域占屏幕多大。
# 3. 最后根据间距，反推出每个按钮自己的宽高。
GRID_COLS = 4
GRID_ROWS = 4
GRID_GAP = 12
GRID_WIDTH = int(DISPLAY_WIDTH * 0.9)
GRID_HEIGHT = int(DISPLAY_HEIGHT * 0.62)
GRID_X = (DISPLAY_WIDTH - GRID_WIDTH) // 2
GRID_Y = DISPLAY_HEIGHT - GRID_HEIGHT - 20
BTN_WIDTH = (GRID_WIDTH - (GRID_COLS - 1) * GRID_GAP) // GRID_COLS
BTN_HEIGHT = (GRID_HEIGHT - (GRID_ROWS - 1) * GRID_GAP) // GRID_ROWS

# 当没有触摸屏输入时，颜色追踪模式要求“长按 1 秒”才算点击成功。
# 这样做的原因是：颜色识别通常会抖动，如果看到手指就立刻触发，误触会很多。
PRESS_HOLD_MS = 1000

# 下面是 UI 颜色主题。
# 这些颜色主要用于 OSD 叠加层 ui_img。
# ARGB8888 可以理解为：
# A = 透明度（Alpha）
# R = 红色
# G = 绿色
# B = 蓝色
#
# 半透明面板的原理：
# 底下仍然是相机画面，上层只画一层带透明度的矩形，所以看起来像“悬浮在画面上”。
CLR_PANEL = (105, 32, 40, 52)
CLR_PANEL_BORDER = (180, 170, 190, 210)
CLR_TEXT_MAIN = (255, 240, 246, 255)
CLR_TEXT_SUB = (225, 170, 185, 205)
CLR_DIGIT = (120, 70, 100, 145)
CLR_OP = (125, 38, 165, 160)
CLR_FUNC = (125, 190, 95, 105)
CLR_EQUAL = (140, 235, 170, 70)
CLR_BTN_BORDER = (210, 210, 220, 235)
CLR_HOVER = (235, 255, 240, 120)
CLR_SHADOW = (90, 10, 16, 24)
CLR_FINGER_DOT = (130, 210, 210, 210)
CLR_FINGER_RING = (205, 235, 235, 235)

# 运算符集合，用来快速判断某个按钮是不是“运算按钮”。
OP_SET = set(["/", "*", "-", "+"])


class ScopedTiming:
    """
    【进阶说明，可跳过】
    这是一个“计时代码块用的工具类”，本例里其实没用到，只是为了兼容官方AI示例结构。
    你可以不用关心它。
    """
    def __init__(self, _name, _enable):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def align_up(value, align):
    # 把 value 向上对齐到 align 的整数倍。比如 align_up(17, 8) = 24。
    # 主要用于AI模型输入尺寸要求“16/32/64的倍数”时自动补齐。
    return ((value + align - 1) // align) * align


def center_pad_param(src_size, dst_size):
    """
    计算等比例缩放时四周需要补的边距。
    比如把 640x360 的图缩放到 512x512，四周要补多少黑边。
    """
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale = min(float(dst_w) / src_w, float(dst_h) / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    left = (dst_w - new_w) // 2
    right = dst_w - new_w - left
    top = (dst_h - new_h) // 2
    bottom = dst_h - new_h - top
    return top, bottom, left, right, scale


if AI_RUNTIME_AVAILABLE:
    # 下面是AI相关的“手部检测/关键点/手势”推理类。
    # 你可以先跳过，等用到AI输入时再细看。
    class SimpleAi2d:
        def __init__(self):
            self.ai2d = nn.ai2d()
            self.builder = None
            self.output_tensor = None
            self.output_shape = None

        def set_dtype(self):
            self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

        def pad_and_resize(self, src_size, dst_size, pad_val):
            top, bottom, left, right, _ = center_pad_param(src_size, dst_size)
            self.ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, pad_val)
            self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            if self.builder is not None:
                del self.builder
            self.builder = self.ai2d.build([1, 3, src_size[1], src_size[0]], [1, 3, dst_size[1], dst_size[0]])
            target_shape = (1, 3, dst_size[1], dst_size[0])
            if self.output_tensor is None or self.output_shape != target_shape:
                output_data = np.ones(target_shape, dtype=np.uint8)
                self.output_tensor = nn.from_numpy(output_data)
                self.output_shape = target_shape

        def crop_and_resize(self, crop_params, src_size, dst_size):
            x, y, w, h = crop_params
            self.ai2d.set_crop_param(True, x, y, w, h)
            self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            if self.builder is not None:
                del self.builder
            self.builder = self.ai2d.build([1, 3, src_size[1], src_size[0]], [1, 3, dst_size[1], dst_size[0]])
            target_shape = (1, 3, dst_size[1], dst_size[0])
            if self.output_tensor is None or self.output_shape != target_shape:
                output_data = np.ones(target_shape, dtype=np.uint8)
                self.output_tensor = nn.from_numpy(output_data)
                self.output_shape = target_shape

        def run(self, input_np):
            input_tensor = nn.from_numpy(input_np)
            self.builder.run(input_tensor, self.output_tensor)
            del input_tensor
            return self.output_tensor


    class SimpleAIBase:
        def __init__(self, kmodel_path):
            self.kpu = nn.kpu()
            self.kpu.load_kmodel(kmodel_path)

        def run_kmodel(self, tensors):
            for index, tensor in enumerate(tensors):
                self.kpu.set_input_tensor(index, tensor)
            self.kpu.run()
            outputs = []
            for index in range(self.kpu.outputs_size()):
                output_tensor = self.kpu.get_output_tensor(index)
                outputs.append(output_tensor.to_numpy())
                del output_tensor
            return outputs

        def deinit(self):
            del self.kpu
            gc.collect()
            nn.shrink_memory_pool()


    class HandDetApp(SimpleAIBase):
        def __init__(self, kmodel_path, rgb888p_size, confidence_threshold=0.2, nms_threshold=0.5):
            super().__init__(kmodel_path)
            self.rgb888p_size = [align_up(rgb888p_size[0], 16), rgb888p_size[1]]
            self.model_input_size = HAND_DET_INPUT_SIZE
            self.strides = [8, 16, 32]
            self.confidence_threshold = confidence_threshold
            self.nms_threshold = nms_threshold
            self.ai2d = SimpleAi2d()
            self.ai2d.set_dtype()
            self.ai2d.pad_and_resize(self.rgb888p_size, self.model_input_size, [114, 114, 114])

        def run(self, input_np):
            results = self.run_kmodel([self.ai2d.run(input_np)])
            dets = aicube.anchorbasedet_post_process(
                results[0],
                results[1],
                results[2],
                self.model_input_size,
                self.rgb888p_size,
                self.strides,
                len(HAND_LABELS),
                self.confidence_threshold,
                self.nms_threshold,
                HAND_ANCHORS,
                False,
            )
            del results
            return dets


    class HandKPClassApp(SimpleAIBase):
        def __init__(self, kmodel_path, rgb888p_size, display_size):
            super().__init__(kmodel_path)
            self.rgb888p_size = [align_up(rgb888p_size[0], 16), rgb888p_size[1]]
            self.display_size = [align_up(display_size[0], 16), display_size[1]]
            self.model_input_size = HAND_KP_INPUT_SIZE
            self.crop_params = [0, 0, self.rgb888p_size[0], self.rgb888p_size[1]]
            self.ai2d = SimpleAi2d()
            self.ai2d.set_dtype()

        def get_crop_param(self, det_box):
            x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
            width = int(x2 - x1)
            height = int(y2 - y1)
            length = max(width, height) / 2
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ratio = 1.26 * length
            crop_x1 = int(max(0, center_x - ratio))
            crop_y1 = int(max(0, center_y - ratio))
            crop_x2 = int(min(self.rgb888p_size[0] - 1, center_x + ratio))
            crop_y2 = int(min(self.rgb888p_size[1] - 1, center_y + ratio))
            crop_w = int(crop_x2 - crop_x1 + 1)
            crop_h = int(crop_y2 - crop_y1 + 1)
            return [crop_x1, crop_y1, crop_w, crop_h]

        def hk_vector_2d_angle(self, v1, v2):
            v1_x, v1_y = v1
            v2_x, v2_y = v2
            v1_norm = np.sqrt(v1_x * v1_x + v1_y * v1_y)
            v2_norm = np.sqrt(v2_x * v2_x + v2_y * v2_y)
            if v1_norm == 0 or v2_norm == 0:
                return 65535.
            cos_angle = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm)
            if cos_angle > 1:
                cos_angle = 1
            elif cos_angle < -1:
                cos_angle = -1
            return np.acos(cos_angle) * 180 / np.pi

        def classify_gesture(self, results):
            angle_list = []
            for index in range(5):
                angle = self.hk_vector_2d_angle(
                    [(results[0] - results[index * 8 + 4]), (results[1] - results[index * 8 + 5])],
                    [(results[index * 8 + 6] - results[index * 8 + 8]), (results[index * 8 + 7] - results[index * 8 + 9])],
                )
                angle_list.append(angle)
            thr_angle = 65.
            thr_angle_thumb = 53.
            thr_angle_small = 49.
            gesture = None
            if 65535. not in angle_list:
                if (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_small) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture = "one"
                elif (angle_list[0] < thr_angle_small) and (angle_list[1] < thr_angle_small) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture = "gun"
                elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_small) and (angle_list[2] < thr_angle_small) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture = "yeah"
                elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture = "fist"
                elif (angle_list[0] < thr_angle_small) and (angle_list[1] < thr_angle_small) and (angle_list[2] < thr_angle_small) and (angle_list[3] < thr_angle_small) and (angle_list[4] < thr_angle_small):
                    gesture = "five"
            return gesture

        def run(self, input_np, det_box):
            self.crop_params = self.get_crop_param(det_box)
            self.ai2d.crop_and_resize(self.crop_params, self.rgb888p_size, self.model_input_size)
            results = self.run_kmodel([self.ai2d.run(input_np)])
            raw = results[0].reshape(results[0].shape[0] * results[0].shape[1])
            points = np.zeros(raw.shape, dtype=np.int16)
            # 与官方示例保持一致：该模型输出的 x/y 顺序在后处理时需要交叉映射。
            points[0::2] = raw[0::2] * self.crop_params[3] + self.crop_params[0]
            points[1::2] = raw[1::2] * self.crop_params[2] + self.crop_params[1]
            gesture = self.classify_gesture(points)
            points[0::2] = points[0::2] * (self.display_size[0] / self.rgb888p_size[0])
            points[1::2] = points[1::2] * (self.display_size[1] / self.rgb888p_size[1])
            del raw
            del results
            return points, gesture


    class HandPointerTracker:
        """
        把手掌框 + 关键点结果转换成计算器可直接使用的指针坐标。
        你可以理解为“AI模式下的鼠标指针追踪器”。
        """
        def __init__(self, rgb888p_size, display_size):
            self.rgb888p_size = [align_up(rgb888p_size[0], 16), rgb888p_size[1]]
            self.display_size = [align_up(display_size[0], 16), display_size[1]]
            self.hand_det = HandDetApp(HAND_DET_MODEL, self.rgb888p_size)
            self.hand_kp = HandKPClassApp(HAND_KP_MODEL, self.rgb888p_size, self.display_size)
            self.smoothed_point = None
            self.det_only_mode = False
            self.last_kp_frame = -999
            self.cached_point = None
            self.cached_gesture = None
            self.frame_id = 0

        def deinit(self):
            self.hand_det.deinit()
            self.hand_kp.deinit()

        def is_valid_det(self, det_box):
            x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
            width = int(x2 - x1)
            height = int(y2 - y1)
            if height < (0.10 * self.rgb888p_size[1]):
                return False
            if width < (0.25 * self.rgb888p_size[0]) and ((x1 < (0.03 * self.rgb888p_size[0])) or (x2 > (0.97 * self.rgb888p_size[0]))):
                return False
            if width < (0.15 * self.rgb888p_size[0]) and ((x1 < (0.01 * self.rgb888p_size[0])) or (x2 > (0.99 * self.rgb888p_size[0]))):
                return False
            return True

        def select_main_hand(self, det_boxes):
            best_box = None
            best_area = -1
            for det_box in det_boxes:
                if not self.is_valid_det(det_box):
                    continue
                area = (det_box[4] - det_box[2]) * (det_box[5] - det_box[3])
                if area > best_area:
                    best_area = area
                    best_box = det_box
            return best_box

        def smooth_point(self, point):
            x, y = point
            x = max(0, min(DISPLAY_WIDTH - 1, int(x)))
            y = max(0, min(DISPLAY_HEIGHT - 1, int(y)))
            if self.smoothed_point is None:
                self.smoothed_point = (x, y)
                return self.smoothed_point
            prev_x, prev_y = self.smoothed_point
            delta_x = abs(x - prev_x)
            delta_y = abs(y - prev_y)
            # 移动很大时提高响应速度，小范围抖动时仍保留一定平滑。
            alpha = 0.92 if (delta_x + delta_y) > 28 else POINTER_SMOOTH_ALPHA
            new_x = int(prev_x + (x - prev_x) * alpha)
            new_y = int(prev_y + (y - prev_y) * alpha)
            self.smoothed_point = (new_x, new_y)
            return self.smoothed_point

        def det_box_to_display(self, det_box):
            x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
            box_x = int(x1 * DISPLAY_WIDTH // self.rgb888p_size[0])
            box_y = int(y1 * DISPLAY_HEIGHT // self.rgb888p_size[1])
            box_w = int((x2 - x1) * DISPLAY_WIDTH // self.rgb888p_size[0])
            box_h = int((y2 - y1) * DISPLAY_HEIGHT // self.rgb888p_size[1])
            return box_x, box_y, box_w, box_h

        def det_box_pointer(self, det_box):
            box_x, box_y, box_w, box_h = self.det_box_to_display(det_box)
            # det-only 模式下使用手掌框上半区域中心，尽量靠近手指方向。
            return (box_x + box_w // 2, box_y + box_h // 3)

        def get_pointer_state(self, input_np):
            self.frame_id += 1
            det_boxes = self.hand_det.run(input_np)
            if not det_boxes:
                self.smoothed_point = None
                self.cached_point = None
                self.cached_gesture = None
                return None
            det_box = self.select_main_hand(det_boxes)
            if not det_box:
                self.smoothed_point = None
                self.cached_point = None
                self.cached_gesture = None
                return None

            if self.det_only_mode:
                fallback_point = self.smooth_point(self.det_box_pointer(det_box))
                return {
                    "point": fallback_point,
                    "gesture": "det_only",
                    "box": self.det_box_to_display(det_box),
                }

            keypoints = None
            gesture = None
            if (self.frame_id - self.last_kp_frame) >= KP_INFER_INTERVAL:
                keypoints, gesture = self.hand_kp.run(input_np, det_box)
                self.last_kp_frame = self.frame_id

            pointer_point = None
            # 对“鼠标式指针”场景，关键点比手势分类更重要。
            # 因此只要关键点可用，就直接使用食指指尖坐标，避免因为 gesture 分类失败而失去光标。
            if keypoints is not None and len(keypoints) >= 18:
                pointer_point = self.smooth_point((int(keypoints[16]), int(keypoints[17])))
                self.cached_point = pointer_point
                self.cached_gesture = gesture if gesture else "kp"
            elif self.cached_point is not None:
                pointer_point = self.cached_point
                gesture = self.cached_gesture
            else:
                self.smoothed_point = None
            return {
                "point": pointer_point,
                "gesture": gesture,
                "box": self.det_box_to_display(det_box),
            }


# =============================
# 第 1 部分：初始化摄像头
# =============================
# 你可以把“摄像头”理解为一台能拍照的相机。
# Sensor() 就是“新建一个摄像头对象”，后面可以用它来拍照、设置分辨率等。
sensor = Sensor()
sensor.reset()  # 复位摄像头，保证参数是初始状态。
try:
    # 水平镜像：修正“你往左动，画面里的点却往右动”的左右反向问题。
    # 对交互类项目，这一步通常很重要，否则用户会觉得操作方向是反的。
    sensor.set_hmirror(True)
except Exception:
    pass  # 某些固件不支持 set_hmirror，可以忽略。

hand_tracker = None
pointer_status_text = "INPUT: TOUCH + COLOR"
pointer_hint_text = "fallback to color blob"
use_ai_pointer = False
ai_error_text = ""



def configure_color_mode(sensor_obj):
    """
    配置为“只用色块追踪”的模式。
    这种模式最兼容，适合AI模型不可用/摄像头通道不支持时。
    """
    sensor_obj.set_pixformat(Sensor.RGB565)
    sensor_obj.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, crop=True)



def configure_ai_mode(sensor_obj):
    """
    配置为“AI手部识别”模式：
    - chn0 负责显示（大分辨率，给人看）。
    - chn2 负责AI推理（小分辨率，给模型用）。
    """
    sensor_obj.set_framesize(w=DISPLAY_WIDTH, h=DISPLAY_HEIGHT, chn=CAM_CHN_ID_0)
    sensor_obj.set_pixformat(Sensor.RGB888, chn=CAM_CHN_ID_0)
    sensor_obj.set_framesize(w=AI_FRAME_WIDTH, h=AI_FRAME_HEIGHT, chn=CAM_CHN_ID_2)
    sensor_obj.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_2)


if AI_RUNTIME_AVAILABLE:
    try:
        # 初始化AI手部追踪器。
        # 只要模型和硬件支持，优先用AI识别手指。
        hand_tracker = HandPointerTracker([AI_FRAME_WIDTH, AI_FRAME_HEIGHT], [DISPLAY_WIDTH, DISPLAY_HEIGHT])
        use_ai_pointer = True
        pointer_status_text = "INPUT: TOUCH + AI POINTER"
        pointer_hint_text = "show one / gun / yeah to point"
    except Exception as e:
        # 如果AI初始化失败，自动降级为色块追踪。
        hand_tracker = None
        ai_error_text = str(e)


if use_ai_pointer:
    configure_ai_mode(sensor)
else:
    # 回退模式仍然使用原来的 RGB565 + 色块追踪逻辑。
    configure_color_mode(sensor)


try:
    # 给摄像头一点时间稳定曝光、白平衡等参数。
    # 某些固件支持 skip_frames，某些不支持，所以做兼容处理。
    sensor.skip_frames(time=2000)
except NotImplementedError:
    time.sleep_ms(2000)


# =============================
# 第 2 部分：初始化显示屏和触摸屏
# =============================
# 你可以把“显示屏”理解为一张大画布，所有内容都要画到这里。
# Display.init(...) 用来初始化 LCD。
# 这里选择的是 ST7701 屏幕驱动，如果你的板子屏幕型号不同，需要改这里。
# to_ide=True：通常表示同时把画面送到 IDE 里，便于调试观察。
# osd_num=1：启用 OSD 图层。OSD 可以理解为“覆盖在底图之上的透明绘图层”。
Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True, osd_num=1)  # 或 Display.LT9611 等，根据你的 LCD 型号调整

# 启动摄像头采集。
# 到这里以后，sensor.snapshot() 才能不断取到新的画面。
sensor.run()

# 尝试清理可能残留的 OSD 图层内容。
# 有些开发板在反复运行脚本时，图层内容可能不会自动干净地清掉，
# 所以先画一张“全透明空白图”覆盖一下，避免看到上一次程序的残影。
try:
    clear_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
    clear_img.clear()
    Display.show_image(clear_img, layer=Display.LAYER_OSD1, alpha=0)
    Display.show_image(clear_img, layer=Display.LAYER_OSD2, alpha=0)
except Exception:
    pass

# ui_img 是“专门负责画界面”的图层。
# 注意：它不是相机图像，而是一张透明背景的 ARGB 图。
# 每一帧我们都会先 clear()，然后重新把按钮、文字、光标等内容画上去。
ui_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)

# TOUCH(0) 表示初始化触摸设备。
# 触摸屏坐标通常和显示屏坐标一一对应，所以非常适合直接点按钮。
tp = TOUCH(0)


def draw_text(img, x, y, text, color=(255, 255, 255), size=24):
    """
    在图像上绘制文字。
    兼容不同固件版本的API（有的叫draw_string_advanced，有的叫draw_string）。
    """
    try:
        img.draw_string_advanced(x, y, size, text, color=color)
    except Exception:
        # 某些固件没有 draw_string_advanced，就退回到 draw_string。
        if hasattr(img, "draw_string"):
            try:
                img.draw_string(x, y, text, color=color)
            except Exception:
                pass


# =============================
# 第 3 部分：计算器状态变量
# =============================
# expression：当前用户已经输入的表达式，例如 "12+7*3"
# result：当前要显示的结果。
#
# 一个常见思路是：
# - 用户还在输入时，优先显示 expression。
# - 用户按下等号后，把计算结果保存到 result，并同步回 expression，便于继续计算。
expression = ""  # 当前输入表达式
result = "0"     # 显示结果

BUTTON_LABELS = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", "C", "=", "+"],
]


def build_buttons():
    """
    根据网格参数，生成每个按钮的矩形区域和文字标签。
    你可以理解为“把按钮排成4x4的表格，每个按钮有自己的坐标和大小”。
    """
    items = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            bx = GRID_X + c * (BTN_WIDTH + GRID_GAP)
            by = GRID_Y + r * (BTN_HEIGHT + GRID_GAP)
            items.append((bx, by, BTN_WIDTH, BTN_HEIGHT, BUTTON_LABELS[r][c]))
    return items


# buttons 里的每个元素格式是：
# (按钮左上角 x, 按钮左上角 y, 按钮宽, 按钮高, 按钮文字)
buttons = build_buttons()


# =============================
# 第 4 部分：颜色追踪参数
# =============================
# 这里使用 find_blobs 做颜色块检测。
# 原理是：
# 1. 把图像中的像素和我们设定的颜色范围进行比较。
# 2. 符合范围的像素会被归为“候选点”。
# 3. 相邻候选点会合并成一个个 blob（色块）。
# 4. 再从这些 blob 中挑出最像“手指”的那个。
#
# 这里采用的是 LAB 颜色空间阈值。
# 你可以先理解为：这是另一种表示颜色的方法，调颜色识别时常常比 RGB 更稳定。
# 这个阈值现在假设你要追踪的“手指目标”偏红色，实际使用时常常需要根据光线重新调整。
finger_threshold = (30, 100, 15, 127, 15, 127)  # L A B 范围

def get_color_finger_pos(img):
    """
    在当前相机画面里寻找“最可能是手指”的颜色块中心。

    返回值：
    - 找到时返回 (x, y)
    - 没找到时返回 None

    实现原理：
    - find_blobs 会返回多个候选色块。
    - 这里简单地取“像素数最多”的那个，认为它最可能是目标手指。
    你可以理解为“用颜色找手指”。
    """
    blobs = img.find_blobs([finger_threshold], pixels_threshold=100, area_threshold=100, merge=True)
    if blobs:
        # 选择像素最多的 blob（假设它最可能是手指）
        largest_blob = max(blobs, key=lambda b: b.pixels())
        return (largest_blob.cx(), largest_blob.cy())
    return None

def check_button_click(pos, buttons):
    """
    判断某个坐标点是否落在某个按钮内，若是则返回按钮文字。
    你可以理解为“判断(x, y)点有没有点到按钮”。
    """
    if not pos:
        return None
    x, y = pos
    for bx, by, bw, bh, label in buttons:
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return label
    return None

def calculate(expr):
    """
    计算字符串表达式。
    这里直接用了 eval，优点是代码短、容易看懂；
    缺点是它会执行字符串里的 Python 表达式，所以更适合教学示例，
    不适合拿去处理不可信输入。
    """
    try:
        return str(eval(expr))  # 使用 eval 解析表达式，例如 "1+2*3"
    except (SyntaxError, ZeroDivisionError, NameError):
        return "Error"


def center_text_x(text, size):
    """
    粗略估算文字宽度，让整行文字大致居中显示。
    """
    # draw_string_advanced font width is approximately size/2 for ASCII chars.
    estimated = len(text) * (size // 2)
    return max(20, (DISPLAY_WIDTH - estimated) // 2)


def center_text_in_box_x(box_x, box_w, text, size):
    """
    让按钮中的文字大致位于按钮中央。
    """
    estimated = len(text) * (size // 2)
    return box_x + max(0, (box_w - estimated) // 2)


def button_theme(label):
    """
    根据按钮类型返回不同的底色和文字颜色。
    """
    if label in OP_SET:
        return CLR_OP, CLR_TEXT_MAIN
    if label == "=":
        return CLR_EQUAL, (25, 20, 12)
    if label == "C":
        return CLR_FUNC, CLR_TEXT_MAIN
    return CLR_DIGIT, CLR_TEXT_MAIN


def draw_translucent_rect(img, x, y, w, h, color, border_color):
    """
    画一个带边框的半透明矩形，常用于面板和按钮。
    """
    img.draw_rectangle(x, y, w, h, color=color, fill=True)
    img.draw_rectangle(x, y, w, h, color=border_color, thickness=2)

# -----------------------------------------------------------------------------
# 第 5 部分：主循环状态变量
# -----------------------------------------------------------------------------
# last_click_time：记录某个按钮开始被长按的时间。
# clicked_button：颜色追踪模式下，当前“正在按住”的按钮。
# hover_button：当前悬停在哪个按钮上，主要给 UI 高亮使用。
last_click_time = 0
clicked_button = None
hover_button = None
pointer_gesture = None
pointer_box = None
frame_counter = 0
total_frame_counter = 0
last_diag_time = time.ticks_ms()
diag_fps = 0
last_step = "init"
last_loop_cost_ms = 0
diag_mem_free = -1
diag_mem_alloc = -1

try:
    while True:
        # =============================
        # 主循环：每一帧都采集输入→识别→绘制UI→显示
        # =============================
        # 你可以把它理解为“每隔几十毫秒刷新一次画面和输入”。
        if hasattr(os, "exitpoint"):
            os.exitpoint()
        loop_begin = time.ticks_ms()
        frame_counter += 1
        total_frame_counter += 1
        last_step = "frame_start"

        # 1. 采集摄像头画面和AI输入帧
        img = None
        pointer_gesture = None
        pointer_box = None
        if use_ai_pointer:
            try:
                # chn0: 给人看的大画面， chn2: 给AI用的小画面
                last_step = "snap_ch0"
                img = sensor.snapshot(chn=CAM_CHN_ID_0)
                last_step = "snap_ch2"
                ai_img = sensor.snapshot(chn=CAM_CHN_ID_2)
                last_step = "ai_infer"
                pointer_state = hand_tracker.get_pointer_state(ai_img.to_numpy_ref())
                del ai_img
                if pointer_state:
                    finger_pos = pointer_state["point"]
                    pointer_gesture = pointer_state["gesture"]
                    pointer_box = pointer_state["box"]
                else:
                    finger_pos = None
            except Exception as e:
                # 如果AI通道抓帧失败，自动降级为色块追踪，保证程序不死。
                use_ai_pointer = False
                pointer_status_text = "INPUT: TOUCH + COLOR"
                pointer_hint_text = "AI chn2 failed, fallback to color"
                ai_error_text = str(e)
                if hand_tracker is not None:
                    try:
                        hand_tracker.deinit()
                    except Exception:
                        pass
                    hand_tracker = None
                try:
                    sensor.stop()
                except Exception:
                    pass
                last_step = "fallback_reinit"
                sensor.reset()
                configure_color_mode(sensor)
                sensor.run()
                img = sensor.snapshot()
                finger_pos = get_color_finger_pos(img)
        else:
            last_step = "snap_color"
            img = sensor.snapshot()
            finger_pos = get_color_finger_pos(img)

        # 2. 清空UI图层，准备绘制新一帧界面
        last_step = "draw_ui"
        ui_img.clear()
        current_time = time.ticks_ms()

        # 3. 读取触摸屏输入（优先级最高）
        touch_pos = None
        touch_just_down = False
        try:
            points = tp.read(1)
            if points:
                pt = points[0]
                touch_pos = (pt.x, pt.y)
                touch_just_down = (pt.event == TOUCH.EVENT_DOWN)
        except Exception:
            pass

        # 4. 选择输入点：优先用触摸，没有触摸就用AI/色块追踪
        display_pos = touch_pos if touch_pos else finger_pos
        current_hover = check_button_click(display_pos, buttons)

        # 5. 计算“长按进度”，只在颜色追踪模式下用来显示进度条
        press_progress = 0
        if not touch_just_down and display_pos and current_hover and clicked_button == current_hover:
            held = time.ticks_diff(current_time, last_click_time)
            press_progress = min(100, (held * 100) // PRESS_HOLD_MS)

        # 6. 绘制UI界面（按钮、表达式、光标等）
        # 顶部信息面板
        draw_translucent_rect(ui_img, 18, 16, DISPLAY_WIDTH - 36, 128, color=CLR_PANEL, border_color=CLR_PANEL_BORDER)
        draw_text(ui_img, 36, 26, "LCKFB TOUCH CALCULATOR", color=CLR_TEXT_SUB, size=22)
        draw_text(ui_img, 36, 54, pointer_status_text, color=CLR_TEXT_SUB, size=18)
        if pointer_gesture:
            draw_text(ui_img, DISPLAY_WIDTH - 240, 50, "gesture: " + pointer_gesture, color=CLR_TEXT_SUB, size=18)
        else:
            draw_text(ui_img, DISPLAY_WIDTH - 320, 50, "pointer follows fingertip", color=CLR_TEXT_SUB, size=18)
        if ai_error_text and not use_ai_pointer:
            draw_text(ui_img, 36, 74, "last ai error: " + ai_error_text[:38], color=(225, 235, 160, 160), size=16)

        # 显示当前表达式或结果
        shown_text = (expression or result)[-20:]
        draw_text(ui_img, center_text_x(shown_text, 54), 72, shown_text, color=CLR_TEXT_MAIN, size=54)

        # 画出计算器主区域边框
        draw_translucent_rect(ui_img, GRID_X - 14, GRID_Y - 14, GRID_WIDTH + 28, GRID_HEIGHT + 28, color=CLR_PANEL, border_color=CLR_PANEL_BORDER)

        # 绘制所有按钮
        for bx, by, bw, bh, label in buttons:
            base_color, text_color = button_theme(label)
            is_hover = (label == current_hover)
            # 先画阴影，再画按钮本体
            ui_img.draw_rectangle(bx + 3, by + 4, bw, bh, color=CLR_SHADOW, fill=True)
            draw_translucent_rect(ui_img, bx, by, bw, bh, color=base_color, border_color=CLR_BTN_BORDER)
            if is_hover:
                ui_img.draw_rectangle(bx - 2, by - 2, bw + 4, bh + 4, color=CLR_HOVER, thickness=2)
            draw_text(ui_img, center_text_in_box_x(bx, bw, label, 40), by + (bh // 2) - 16, label, color=text_color, size=40)
            # 长按进度条（只在颜色追踪模式下有意义）
            if is_hover and press_progress > 0:
                progress_w = (bw * press_progress) // 100
                ui_img.draw_rectangle(bx + 2, by + bh - 10, progress_w - 4 if progress_w > 6 else progress_w, 6, color=CLR_HOVER, fill=True)

        # 在当前输入位置上画一个圆点，方便用户观察系统认为“你点到了哪里”
        if display_pos:
            fx, fy = display_pos
            ui_img.draw_circle(fx, fy, 10, color=CLR_FINGER_DOT, fill=True)
            ui_img.draw_circle(fx, fy, 15, color=CLR_FINGER_RING, thickness=2, fill=False)

        # AI模式下画出手掌框，便于调试
        if pointer_box:
            box_x, box_y, box_w, box_h = pointer_box
            ui_img.draw_rectangle(box_x, box_y, box_w, box_h, color=(200, 0, 255, 0), thickness=2)

        hover_button = current_hover

        # 7. 处理输入逻辑
        # 触摸屏：按下瞬间就触发一次（最优先，最可靠）
        if touch_just_down:
            btn = check_button_click(touch_pos, buttons)
            if btn:
                if btn == "C":
                    expression = ""
                    result = "0"
                elif btn == "=":
                    if expression:
                        result = calculate(expression)
                        expression = result
                else:
                    expression += btn
            clicked_button = None
        # 颜色追踪/AI：必须长按1秒才触发一次，防止误触
        elif finger_pos:
            btn = check_button_click(finger_pos, buttons)
            if btn:
                if clicked_button != btn:
                    clicked_button = btn
                    last_click_time = current_time
                elif time.ticks_diff(current_time, last_click_time) > PRESS_HOLD_MS:
                    if btn == "C":
                        expression = ""
                        result = "0"
                    elif btn == "=":
                        if expression:
                            result = calculate(expression)
                            expression = result
                    else:
                        expression += btn
                    last_click_time = current_time
                    clicked_button = None
            else:
                clicked_button = None
        else:
            clicked_button = None
            hover_button = None

        # 8. 显示输出（先底图再UI叠加层）
        now_ms = time.ticks_ms()
        last_loop_cost_ms = time.ticks_diff(now_ms, loop_begin)
        if hasattr(gc, "mem_free"):
            diag_mem_free = gc.mem_free()
            diag_mem_alloc = gc.mem_alloc()
        if DEBUG_RUNTIME:
            draw_text(ui_img, 36, 94, "step:" + last_step + " loop:" + str(last_loop_cost_ms) + "ms", color=(225, 170, 225, 170), size=16)
            if diag_mem_free >= 0:
                draw_text(ui_img, 36, 112, "mem free:" + str(diag_mem_free) + " alloc:" + str(diag_mem_alloc), color=(225, 170, 225, 170), size=16)

        if img is not None:
            Display.show_image(img)
        Display.show_image(ui_img, layer=Display.LAYER_OSD1)  # 半透明 UI 叠加层

        # 9. 内存保护与调试输出
        if total_frame_counter % FORCE_GC_INTERVAL_FRAMES == 0:
            gc.collect()
        if AI_RUNTIME_AVAILABLE and nn and total_frame_counter % SHRINK_POOL_INTERVAL_FRAMES == 0:
            nn.shrink_memory_pool()
        # 低内存保护：自动切 det-only 模式
        if use_ai_pointer and hand_tracker is not None and diag_mem_free > 0 and diag_mem_free < MEM_GUARD_FREE_MIN and not hand_tracker.det_only_mode:
            hand_tracker.det_only_mode = True
            pointer_hint_text = "mem low, switch to det-only"
            print("[guard] low memory, switch to det-only mode")
        if DEBUG_RUNTIME:
            if time.ticks_diff(now_ms, last_diag_time) >= DEBUG_PRINT_INTERVAL_MS:
                elapsed = max(1, time.ticks_diff(now_ms, last_diag_time))
                diag_fps = (frame_counter * 1000) // elapsed
                print("[diag] mode=", "AI" if use_ai_pointer else "COLOR",
                      "det_only=", hand_tracker.det_only_mode if hand_tracker else False,
                      "fps=", diag_fps,
                        "frames=", total_frame_counter,
                      "loop_ms=", last_loop_cost_ms,
                      "step=", last_step,
                      "mem_free=", diag_mem_free,
                      "mem_alloc=", diag_mem_alloc,
                      "ai_err=", ai_error_text)
                frame_counter = 0
                last_diag_time = now_ms
        if last_loop_cost_ms > LOOP_HANG_WARN_MS:
            print("[warn] long loop:", last_loop_cost_ms, "ms at step", last_step)
        # 10. 控制刷新速度，减少CPU压力
        time.sleep(IDLE_SLEEP_S)
except KeyboardInterrupt:
    pass
except BaseException as e:
    # 在 CanMV IDE 里点击停止时，很多时候会抛出 IDE interrupt。
    # 这不是真正的程序错误，所以这里忽略它。
    if "IDE interrupt" not in str(e):
        raise
finally:
    # 程序结束时释放硬件资源，避免下次运行出错
    if isinstance(sensor, Sensor):
        sensor.stop()
    if hand_tracker is not None:
        hand_tracker.deinit()
    Display.deinit()
    time.sleep_ms(100)