import cv2
import numpy as np

# print(cv2.getBuildInformation())
# WIDTH=1280
# HEIGHT=720

WIDTH=1920
HEIGHT=1080

WIDTH=1920
HEIGHT=1080 - 100 ## プロジェクターだと縦がカットされてる

WIDTH=int(WIDTH/2)
HEIGHT=int(HEIGHT/2)

## --- ##
CAP_WIDTH = 640
CAP_HEIGHT = 360
## --- ##

ZOOM=1.37
ZOOM=1.00

COLOR_SET=dict(
    green=(0, 255, 0),
    red=(0, 0, 255),
    blue=(255, 0, 0),
    pink=(255, 0, 165),
    puple=(128, 0, 128),
)

def _zoom(frame, rate=1.0):
    """
    frame: np.array
    rate: zoom rate. 1.0より大きい値を期待、1.0より小さい場合うまく動かない

    画面の中央を切り取り拡大する
    """
    h, w, _ = frame.shape
    crop_h = int((h - h/rate)/2)
    crop_h_to = int(h/rate)
    crop_w = int((w - w/rate)/2)
    crop_w_to = int(w/rate)
    return cv2.resize(frame[crop_h:crop_h_to, crop_w:crop_w_to, :], dsize = (WIDTH, HEIGHT))

pts = []
def mouse_callback(event, x, y, flags, param):
    """
    左上、右上、右下、左下の順番
    """
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print([x, y])
        if len(pts) > 4:
            pts.pop(0)
        pts.append([x, y])
        if len(pts) == 4: 
            print(pts)



def convert(frame, w, h):
    PTS = [[84, 35], [502, 49], [507, 280], [73, 279]]
    from_pts = np.array(PTS)
    dst_pts = np.array([
        [0,0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]], dtype="float32" 
    )

    M = cv2.getPerspectiveTransform(from_pts.astype("float32"), dst_pts)
    t_frame = cv2.warpPerspective(frame, M, (w, h))
    return t_frame

#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_hight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("cap width: ", cap_width)
print("cap height: ", cap_hight)

cv2.namedWindow("camera")
cv2.setMouseCallback("camera", mouse_callback)


#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    ret, frame = cap.read()
    if not ret:
        print("ret false...")
        # continue
    
    for v in pts:
        [x, y]  = v
        cv2.circle(frame, (x,y), 5, COLOR_SET["red"], thickness=2)

    # print(frame.shape)
    #f = _zoom(frame, rate=1.0)
    # cv2.circle(frame, (10, 10), 10, COLOR_SET["red"], thickness=5)
    # cv2.circle(frame, (int(cap_width)-10, int(cap_hight)-10), 10, COLOR_SET["red"], thickness=5)

    _resized_frame = frame
    # _resized_frame = cv2.resize(frame, dsize = (WIDTH, HEIGHT))
    # _resized_frame = _zoom(frame, rate=ZOOM)
    # _resized_frame = convert(frame, CAP_WIDTH, CAP_HEIGHT)
    cv2.imshow('camera' , _resized_frame)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(10)
    if key == 27:
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()
