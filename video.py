from function.processing import *


def main(filespath: str) -> None:
    yolo, ocr = loadnets()
    for file in os.listdir(filespath):
        fullpath = os.path.join(filespath, file)
        if os.path.isfile(fullpath):
            cap = cv2.VideoCapture(fullpath, )
            if not cap.isOpened():
                print("Cannot open")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                outputs = yolo.predict(source=frame, save=False)
                for output in outputs:
                    for out in output:
                        if out.boxes.cls == plate:
                            xyxy = out.boxes.xyxy[0]
                            x, y, w, h = map(int, xyxy)
                            carplate = frame[y - 1:h, x - 1:w, :]
                            # im = cv2.resize(im, (94, 24), interpolation=cv2.INTER_CUBIC)
                            numplate = ocr.ocr(carplate, det=True, cls=False)
                            numplate = numplate[0]
                            if (w - x) / (h - y) > 2:
                                numplate.sort(key=lambda coord: (coord[0][0][0], coord[0][0][1]))
                            else:
                                numplate.sort(key=lambda coord: (coord[0][0][1], coord[0][0][0]))
                            numplate_conf, numplate_data = [], []
                            for i in range(min(2, len(numplate))):
                                numplate_conf.append(numplate[i][1][1])
                                numplate_data.append(numplate[i][1][0])
                            if len(numplate_data):
                                numplate = datafilter(''.join(numplate_data))
                                frame = cv2.rectangle(img=frame, pt1=(x, y), pt2=(w, h), color=(255, 0, 255),
                                                   thickness=3)
                                frame = cv2.putText(frame, numplate, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                 fontScale=0.8, color=(0, 255, 255), thickness=2, )

                cv2.imshow('video', frame)
                if cv2.waitKey(30) == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--videopath', '-ip', help='Path to folder with image for recognition', default='testcopy')
    args = parser.parse_args()
    main(args.videopath)
