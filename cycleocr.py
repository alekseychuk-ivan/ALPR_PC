import os
import time
from function.processing import *
from datetime import datetime


def main(filespath: str) -> None:
    now = time.time()
    with open(f"{os.path.join(filespath, 'work')}.txt", "a", encoding='ansi') as my_file:
        my_file.write('')
    yolo, ocr = loadnets()
    filespath = Path(filespath)
    srun = True
    try:
        while srun:
            if time.time() - now >= 10:
                with open(f"{os.path.join(filespath, 'work')}.txt", "a") as my_file:
                    my_file.write('')
                    now = time.time()

            if os.path.isfile(os.path.join(filespath, 'stop.txt')):
                srun = False
                os.remove(os.path.join(filespath, 'stop.txt'))
            if os.path.isfile(os.path.join(filespath, 'todo.txt')):
                try:
                    with open(os.path.join(filespath, 'todo.txt'), 'r', encoding="ansi") as my_file:
                        todo = my_file.read().splitlines()
                    os.remove(os.path.join(filespath, 'todo.txt'))
                    filename = todo[0]
                    data = list()
                    for file in todo[1:]:
                        if file.split('.')[-1].lower() in ['jpg', 'bmp', 'png', 'jpeg', 'jpe']:
                            try:
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
                                                data.append([numplate, round(platesdct[pl]["conf"], 5), round(numplate_conf, 5),
                                                             f'{"C" if carsdct[ct]["car_class"] == car else "T"}'])
                                os.remove(os.path.join(filespath, file))
                            except Exception as ex:
                                with open((os.path.join(filespath, f'{file.split(".")[0]}_'
                                                                   f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')),
                                          'w', encoding='ansi') as my_file:
                                    my_file.write(str(ex))
                    data.sort(key=lambda prob: prob[1] * prob[2], reverse=True)
                    with open(f"{os.path.join(filespath, filename)}", "a",  encoding='ansi') as my_file:
                        my_file.write("\t".join(str(item) for item in data[0]))
                except Exception as ex:
                    with open((os.path.join(filespath, f'error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')), 'w',
                              encoding='ansi') as my_file:
                        my_file.write(str(ex))
            else:
                time.sleep(1)
        if os.path.isfile(os.path.join(filespath, 'work.txt')):
            os.remove(os.path.join(filespath, 'work.txt'))
    except KeyboardInterrupt:
        print('Exit from cycle')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', '-ip', help='Path to folder with image for recognition', default='testcopy')
    args = parser.parse_args()
    main(args.imagepath)
