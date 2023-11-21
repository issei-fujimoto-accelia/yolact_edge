import cv2

# WIDTH=1280
# HEIGHT=720
WIDTH=1920
HEIGHT=1080


def _zoom(frame, rate=1.0):
    h, w, _ = frame.shape
    crop_h = int((h - h/rate)/2)    
    crop_h_to = int(h/rate)
    crop_w = int((w - w/rate)/2)
    crop_w_to = int(w/rate)
    # return cv2.resize(frame[crop_h:crop_h+h, crop_w:crop_w+w, :], dsize = (w, h))
    
    return cv2.resize(frame[crop_h:crop_h_to, crop_w:crop_w_to, :], dsize = (w, h))

#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)
print("width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    ret, frame = cap.read()    
    #カメラの画像の出力
    
    f = _zoom(frame, rate=1.0)
    cv2.imshow('camera' , f)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(10)
    if key == 27:
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()
