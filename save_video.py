import cv2
import argparse 

def main():
    parser = argparse.ArgumentParser(description='カメラで撮影した映像をそのまま保存')
    parser.add_argument('--camera',type=int)
    # parser.add_argument('--save_file', required=True, type=str)
    args = parser.parse_args()

    # camera = cv2.VideoCapture(args.camera)
    camera = cv2.VideoCapture(2)
    camera.set(cv2.CAP_PROP_FPS, 20)
    print("fps: ", camera.get(cv2.CAP_PROP_FPS))
    
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(args.save_file, fourcc, fps, (w, h))
    video = cv2.VideoWriter("./small.mp4", fourcc, fps, (w, h))

    while True:
        ret, frame = camera.read()
        cv2.imshow('camera', frame)
        video.write(frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()