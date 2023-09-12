import time
from function.processing import *


def main(filespath: str) -> None:
    yolo, ocr = loadnets()
    now = time.time()
    with open(f"{os.path.join(filespath, 'work')}.txt", "a") as my_file:
        my_file.write('')
    filespath = Path(filespath)
    srun = True
    try:
        while srun:
            if time.time() - now >= 10:
                with open(f"{os.path.join(filespath, 'work')}.txt", "a") as my_file:
                    my_file.write('')
                    now = time.time()
            for file in os.listdir(filespath):
                if os.path.isfile(os.path.join(filespath, file)):
                    if file == 'stop.txt':
                        srun = False
                        os.remove(os.path.join(filespath, 'stop.txt'))
                        continue
                    data = f''
                    if file.split('.')[-1].lower() in ['jpg', 'bmp', 'png', 'jpeg', 'jpe']:
                        filename = Path(file).stem

                        image = cv2.imread(os.path.join(filespath, file))
                        outputs = yolo.predict(source=image, save=False)
                        carsdct, platesdct, car_i, plate_i = dict(), dict(), 0, 0

                        for output in outputs:
                            for out in output:
                                x, y, w, h = map(int, out.boxes.xyxy[0])
                                if out.boxes.cls == plate:
                                    platesdct[f'plate_{plate_i}'] = {'area': [x, y, w, h],
                                                                     'conf': float(out.boxes.conf), }
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
                                    carplate = image[yp - 1: hp, xp - 1: wp, :]
                                    numplate = ocr.ocr(carplate, det=True, cls=False)
                                    numplate = numplate[0]
                                    if (wp - xp) / (hp - yp) > 2:
                                        numplate.sort(key=lambda coord: (coord[0][0][0], coord[0][0][1]))
                                    else:
                                        numplate.sort(key=lambda coord: (coord[0][0][1], coord[0][0][0]))
                                    numplate_conf, numplate_data = [], []
                                    for i in range(min(2, len(numplate))):
                                        numplate_conf.append(numplate[i][1][1])
                                        numplate_data.append(numplate[i][1][0])
                                    if len(numplate):
                                        numplate = datafilter(''.join(numplate_data))
                                        numplate_conf = sum(numplate_conf) / len(numplate_conf)
                                        data += f'{numplate}\t{round(numplate_conf, 5)}\t'\
                                                f'{round(platesdct[pl]["conf"], 5)}\t'\
                                                f'{"C" if carsdct[ct]["car_class"] == car else "T"}\r\n'
                        with open(f"{os.path.join(filespath, filename)}.txt", "a") as my_file:
                            my_file.write(data)
                    else:
                        continue
                    os.remove(os.path.join(filespath, file))
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
