#https://wiki.lckfb.com/zh-hans/lushan-pi-k230/
# 这个项目关于一个立创开发版的K230，有摄像头和LCD,基于micropython

from media.sensor import *
from media.display import *
from machine import TOUCH
import image
import time

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

GRID_COLS = 4
GRID_ROWS = 4
GRID_GAP = 12
GRID_WIDTH = int(DISPLAY_WIDTH * 0.9)
GRID_HEIGHT = int(DISPLAY_HEIGHT * 0.62)
GRID_X = (DISPLAY_WIDTH - GRID_WIDTH) // 2
GRID_Y = DISPLAY_HEIGHT - GRID_HEIGHT - 20
BTN_WIDTH = (GRID_WIDTH - (GRID_COLS - 1) * GRID_GAP) // GRID_COLS
BTN_HEIGHT = (GRID_HEIGHT - (GRID_ROWS - 1) * GRID_GAP) // GRID_ROWS

PRESS_HOLD_MS = 1000

# UI theme colors (ARGB overlay; no stripe artifacts)
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

OP_SET = set(["/", "*", "-", "+"])

# 初始化摄像头
sensor = Sensor()
sensor.reset()
try:
    # 水平镜像，修正左右反向操作问题
    sensor.set_hmirror(True)
except Exception:
    pass
sensor.set_pixformat(Sensor.RGB565)
sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, crop=True)
try:
    sensor.skip_frames(time=2000)
except NotImplementedError:
    time.sleep_ms(2000)

# 初始化显示 (假设使用LCD，需根据硬件调整)
Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True, osd_num=1)  # 或 Display.LT9611 等，根据你的LCD型号
sensor.run()

# 尝试清理可能残留的OSD图层内容（来自之前示例脚本）
try:
    clear_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
    clear_img.clear()
    Display.show_image(clear_img, layer=Display.LAYER_OSD1, alpha=0)
    Display.show_image(clear_img, layer=Display.LAYER_OSD2, alpha=0)
except Exception:
    pass

ui_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
tp = TOUCH(0)


def draw_text(img, x, y, text, color=(255, 255, 255), size=24):
    try:
        img.draw_string_advanced(x, y, size, text, color=color)
    except Exception:
        # Some firmware variants may not provide draw_string fallback.
        if hasattr(img, "draw_string"):
            try:
                img.draw_string(x, y, text, color=color)
            except Exception:
                pass

# 计算器状态变量
expression = ""  # 当前输入表达式
result = "0"     # 显示结果

BUTTON_LABELS = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", "C", "=", "+"],
]


def build_buttons():
    items = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            bx = GRID_X + c * (BTN_WIDTH + GRID_GAP)
            by = GRID_Y + r * (BTN_HEIGHT + GRID_GAP)
            items.append((bx, by, BTN_WIDTH, BTN_HEIGHT, BUTTON_LABELS[r][c]))
    return items


buttons = build_buttons()

# 手指颜色阈值（LAB颜色空间，针对红色手指，需根据环境/灯光调整）
finger_threshold = (30, 100, 15, 127, 15, 127)  # L A B 范围

# 函数：检测手指位置（返回中心坐标）
def get_finger_pos(img):
    blobs = img.find_blobs([finger_threshold], pixels_threshold=100, area_threshold=100, merge=True)
    if blobs:
        # 选择像素最多的blob（假设是手指）
        largest_blob = max(blobs, key=lambda b: b.pixels())
        return (largest_blob.cx(), largest_blob.cy())
    return None

# 函数：检查手指是否在按钮内（返回按钮标签）
def check_button_click(pos, buttons):
    if not pos:
        return None
    x, y = pos
    for bx, by, bw, bh, label in buttons:
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return label
    return None

# 函数：计算表达式（简单eval，支持+ - * /）
def calculate(expr):
    try:
        return str(eval(expr))  # 使用eval解析表达式
    except (SyntaxError, ZeroDivisionError, NameError):
        return "Error"


def center_text_x(text, size):
    # draw_string_advanced font width is approximately size/2 for ASCII chars.
    estimated = len(text) * (size // 2)
    return max(20, (DISPLAY_WIDTH - estimated) // 2)


def center_text_in_box_x(box_x, box_w, text, size):
    estimated = len(text) * (size // 2)
    return box_x + max(0, (box_w - estimated) // 2)


def button_theme(label):
    if label in OP_SET:
        return CLR_OP, CLR_TEXT_MAIN
    if label == "=":
        return CLR_EQUAL, (25, 20, 12)
    if label == "C":
        return CLR_FUNC, CLR_TEXT_MAIN
    return CLR_DIGIT, CLR_TEXT_MAIN


def draw_translucent_rect(img, x, y, w, h, color, border_color):
    img.draw_rectangle(x, y, w, h, color=color, fill=True)
    img.draw_rectangle(x, y, w, h, color=border_color, thickness=2)

# 主循环变量
last_click_time = 0
clicked_button = None
hover_button = None

try:
    while True:
        img = sensor.snapshot()  # 捕捉摄像头帧
        finger_pos = get_finger_pos(img)  # 颜色追踪手指位置
        ui_img.clear()
        current_time = time.ticks_ms()

        # 读取触摸屏输入（触摸优先级高于颜色追踪）
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

        # 触摸位置优先，否则用颜色追踪
        display_pos = touch_pos if touch_pos else finger_pos
        current_hover = check_button_click(display_pos, buttons)

        press_progress = 0
        if not touch_just_down and display_pos and current_hover and clicked_button == current_hover:
            held = time.ticks_diff(current_time, last_click_time)
            press_progress = min(100, (held * 100) // PRESS_HOLD_MS)

        # 绘制UI
        # 顶部信息面板
        draw_translucent_rect(ui_img, 18, 16, DISPLAY_WIDTH - 36, 128, color=CLR_PANEL, border_color=CLR_PANEL_BORDER)
        draw_text(ui_img, 36, 26, "LCKFB TOUCH CALCULATOR", color=CLR_TEXT_SUB, size=22)

        # 显示当前表达式或结果（上方居中）
        shown_text = (expression or result)[-20:]
        draw_text(ui_img, center_text_x(shown_text, 54), 72, shown_text, color=CLR_TEXT_MAIN, size=54)

        # 画出计算器主区域边框，让UI更聚焦在中心
        draw_translucent_rect(ui_img, GRID_X - 14, GRID_Y - 14, GRID_WIDTH + 28, GRID_HEIGHT + 28, color=CLR_PANEL, border_color=CLR_PANEL_BORDER)

        # 绘制按钮
        for bx, by, bw, bh, label in buttons:
            base_color, text_color = button_theme(label)
            is_hover = (label == current_hover)

            # 按钮阴影层
            ui_img.draw_rectangle(bx + 3, by + 4, bw, bh, color=CLR_SHADOW, fill=True)

            # 按钮主体
            draw_translucent_rect(ui_img, bx, by, bw, bh, color=base_color, border_color=CLR_BTN_BORDER)

            # 悬停高亮
            if is_hover:
                ui_img.draw_rectangle(bx - 2, by - 2, bw + 4, bh + 4, color=CLR_HOVER, thickness=2)

            draw_text(ui_img, center_text_in_box_x(bx, bw, label, 40), by + (bh // 2) - 16, label, color=text_color, size=40)

            # 按压进度条
            if is_hover and press_progress > 0:
                progress_w = (bw * press_progress) // 100
                ui_img.draw_rectangle(bx + 2, by + bh - 10, progress_w - 4 if progress_w > 6 else progress_w, 6, color=CLR_HOVER, fill=True)

        # 指尖标记：淡灰色实心圆 + 外环（触摸或颜色追踪位置）
        if display_pos:
            fx, fy = display_pos
            ui_img.draw_circle(fx, fy, 10, color=CLR_FINGER_DOT, fill=True)
            ui_img.draw_circle(fx, fy, 15, color=CLR_FINGER_RING, thickness=2, fill=False)

        # 更新悬停按钮状态
        hover_button = current_hover

        # 触摸屏：单次按下瞬时触发
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

        # 颜色追踪：长按1秒触发（无触摸时生效）
        elif finger_pos:
            btn = check_button_click(finger_pos, buttons)
            if btn:
                if clicked_button != btn:
                    clicked_button = btn
                    last_click_time = current_time
                elif time.ticks_diff(current_time, last_click_time) > PRESS_HOLD_MS:  # 长按触发阈值
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

        Display.show_image(img)  # 相机底图
        Display.show_image(ui_img, layer=Display.LAYER_OSD1)  # 半透明UI叠加层
        time.sleep(0.05)  # 约20 FPS，优化实时性
except KeyboardInterrupt:
    pass
except BaseException as e:
    # CanMV IDE stop usually raises: Exception("IDE interrupt")
    if "IDE interrupt" not in str(e):
        raise
finally:
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    time.sleep_ms(100)
