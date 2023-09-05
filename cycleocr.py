import time

from ultralytics import YOLO
from pathlib import Path
from paddleocr import PaddleOCR
import os
from function.processing import *


weight = Path('detect/model.pt')
fullpath = os.getcwd()
yolo = YOLO(model=os.path.join(fullpath, weight))
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=os.path.join(fullpath, Path('detect/det_dir')),
                rec_model_dir=os.path.join(fullpath, Path('detect/rec_dir')),
                cls_model_dir=os.path.join(fullpath, Path('detect/cls_dir')))


def main(filespath: str) -> None:
    filespath = Path(filespath)
    srun = True
    try:
        while srun:
            for path, _, files in os.walk(filespath):
                if len(files) > 0:
                    if 'stop.txt' in files:
                        srun = False
                    for file in files:
                        data = f''
                        if file.split('.')[-1].lower() in ['jpg', 'bmp', 'png', 'jpeg', 'jpe']:
                            filename = Path(file).stem

                            image = cv2.imread(os.path.join(path, file))
                            # os.remove(os.path.join(path, file))
                            outputs = yolo.predict(source=image, save=False)
                            carsdct, platesdct, car_i, plate_i = dict(), dict(), 0, 0

                            for output in outputs:
                                for out in output:
                                    x, y, w, h = map(int, out.boxes.xyxy[0])
                                    if out.boxes.cls == plate:
                                        platesdct[f'plate_{plate_i}'] = {'area': [x, y, w, h], 'conf': float(out.boxes.conf), }
                                        plate_i += 1
                                    elif out.boxes.cls == car or out.boxes.cls == truck:
                                        carsdct[f'car_{car_i}'] = {'area': [x, y, w, h], 'conf': float(out.boxes.conf),
                                                                   'car_class': int(out.boxes.cls), }
                                        car_i += 1

                            for pl in platesdct:
                                xp, yp, wp, hp = platesdct[pl]['area']
                                for ct in carsdct:
                                    xc, yc, wc, hc = carsdct[ct]['area']
                                    if xc <= xp and yc <= yp and wp <= wc and hp <= hc:
                                        carplate = image[yp: hp, xp: wp, :]
                                        carplate = cv2.resize(carplate, (94, 24), interpolation=cv2.INTER_CUBIC)
                                        numplate = ocr.ocr(carplate, det=False, cls=False)
                                        numplate_conf = numplate[0][0][1]
                                        numplate = datafilter(numplate[0][0][0])
                                        if 0 < len(numplate) < 5:
                                            numplate = ocr.ocr(read_pate(carplate[y - 1: h, x - 1: w, :]), det=False,
                                                               cls=False)
                                            numplate_conf = numplate[0][0][1]
                                            numplate = datafilter(numplate[0][0][0])
                                        data += f'{numplate}\t{round(numplate_conf, 5)}\t'\
                                                f'{round(platesdct[pl]["conf"], 5)}\t'\
                                                f'{"C" if carsdct[ct]["car_class"] == car else "T"}\r\n'
                            with open(f"{os.path.join(path, filename)}.txt", "a") as my_file:
                                my_file.write(data)
                        else:
                            continue
                        os.remove(os.path.join(path, file))
                else:
                    time.sleep(1)
    except KeyboardInterrupt:
        print('Exit from cycle')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', '-ip', help='Path to folder with image for recognition', default='testcopy')
    args = parser.parse_args()
    main(args.imagepath)
