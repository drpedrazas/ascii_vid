import shutil
import sys, os, argparse
import shutil
from threading import Thread
import time


import numpy as np
from PIL import Image
import functools
import os
import cv2
from threading import Thread

chars = '@%#*+=-:. '

arroba = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1]
])
porcentaje = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1]
])
numeral = np.array([
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0]
])

asterisco = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0]
])

suma = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
])

igual = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
menos = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

dospts = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

pt = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0]
])

blanc = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
char_images = [arroba, porcentaje, numeral, asterisco, suma, igual, menos, dospts, pt, blanc]
correspondencia = dict((x, y) for x, y in zip(chars, char_images))


def getAverageL(image):
    im = np.array(image)
    i, j = im.shape
    if np.isnan(np.average(im.reshape(i * j))):
        return 0
    else:
        return np.average(im.reshape(i * j))


def to_ascii(file_name, cols=75):
    image = Image.open(file_name).convert('L')
    W, H = image.size[0], image.size[1]
    w = W / cols
    h = int(w * 1.5)
    rows = int(H / h)
    aimg = list()
    for j in range(rows):
        y1 = int(j * h)
        y2 = int((j + 1) * h)
        if j == rows - 1:
            y2 = H
        aimg.append("")
        for i in range(cols):
            x1 = int(i * w)
            x2 = int((i + 1) * w)
            if i == cols - 1:
                x2 = W
            img = image.crop((x1, y1, x2, y2))
            avg = int(getAverageL(img))
            gsval = chars[int((avg * 9) / 255)]
            aimg[j] += gsval
    return aimg


def ascii_image(imagen, file_name, cols=75, video=False):
    aimg = to_ascii(imagen, cols)
    matrix = np.concatenate(np.array([correspondencia[i] for i in aimg[0]]), axis=1)
    for row in aimg[1:]:
        t = np.concatenate(np.array([correspondencia[i] for i in row]), axis=1)
        matrix = np.concatenate([matrix, t])
    binary = matrix > 0
    k = Image.fromarray(binary)
    if video:
        k.save("{}_ascii_frames".format(file_name) + "/" + imagen.split("/")[-1])
    else:
        k.save("ascii_images/{}.jpeg".format(file_name))


def toFrames(name, file_name):
    vidcap = cv2.VideoCapture(name)
    success, image = vidcap.read()
    count = 0
    directory = file_name + "_frames"
    while success:
        cv2.imwrite(directory + "/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


def toVideo(name, fps):
    img_array = []
    folder = "{}_frames".format(name)
    os.chdir(folder)
    n_frames = len(os.listdir("."))
    os.chdir("..")
    lista = [folder + "/frame{}.jpg".format(i) for i in range(n_frames)]
    for filename in lista:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter('out/{}.mp4'.format(name), apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=int(fps),
                          frameSize=size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def to_ascii1(project_name, n_frames, cols=75):
    frames_folder = "{}_frames".format(project_name)
    frame_list = ["frame{}.jpg".format(i) for i in range(0, n_frames, 3)]
    for image in frame_list:
        ascii_image(frames_folder + "/" + image, project_name, cols, video=True)


def to_ascii2(project_name, n_frames, cols=75):
    frames_folder = "{}_frames".format(project_name)
    frame_list = ["frame{}.jpg".format(i) for i in range(1, n_frames, 3)]
    for image in frame_list:
        ascii_image(frames_folder + "/" + image, project_name, cols, video=True)


def to_ascii3(project_name, n_frames, cols=75):
    frames_folder = "{}_frames".format(project_name)
    frame_list = ["frame{}.jpg".format(i) for i in range(2, n_frames, 3)]
    for image in frame_list:
        ascii_image(frames_folder + "/" + image, project_name, cols, video=True)


def to_ascii_vid(vid, project_name, cols=75, fps=24):
    frames_folder = "{}_frames".format(project_name)
    os.mkdir(frames_folder)
    os.mkdir("{}_ascii_frames".format(project_name))
    toFrames(vid, project_name)
    os.chdir(frames_folder)
    frame_list = ["frame{}.jpg".format(i) for i in range(len(os.listdir(".")))]
    os.chdir("..")
    t1 = Thread(target=to_ascii1, args=(project_name, len(frame_list), cols))
    t2 = Thread(target=to_ascii2, args=(project_name, len(frame_list), cols))
    t3 = Thread(target=to_ascii3, args=(project_name, len(frame_list), cols))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    toVideo("{}_ascii".format(project_name), fps)


class WorkerThread(Thread):
    def __init__(self, value=0):
        super(WorkerThread, self).__init__()

        self.value = value

    def run(self):
        main()


class ProgressThread(Thread):
    def __init__(self, worker):
        super(ProgressThread, self).__init__()

        self.worker = worker

    def run(self):
        curr = 1
        while True:
            if not self.worker.is_alive():
                print("Done.            ")
                return True

            print("Processing" + "." * (curr % 4) + " " * (4 - (curr % 4)), end='\r')
            time.sleep(0.25)
            curr += 1


def main():
    parser = argparse.ArgumentParser(description="Converting images and video into ascii art.")
    parser.add_argument("-cols", help="Number of columns in your ascii image", type=int)
    parser.add_argument("--video", help="Is your project a video?", action='store_true', required=False)
    parser.add_argument("-fps", help="fps for your video", type=str, required=False)
    parser.add_argument("file_name", help="Image or video you want to convert", type=str)
    parser.add_argument("name", help="Name of your project", type=str)
    args = parser.parse_args()
    print("Processing...", end='\r')
    if args.video:
        if not os.path.isdir("out"):
            os.mkdir("out")
        if os.path.isdir(args.name + "_frames"):
            shutil.rmtree(args.name + "_frames")
        if os.path.isdir(args.name + "_ascii_frames"):
            shutil.rmtree(args.name + "_ascii_frames")
        if args.cols is None or args.fps is None:
            if args.cols is None and args.fps is None:
                to_ascii_vid(args.file_name, args.name, )
            elif args.cols is not None:
                to_ascii_vid(args.file_name, args.name, cols=args.cols)
            else:
                to_ascii_vid(args.file_name, args.name, fps=args.fps)
        else:
            to_ascii_vid(args.file_name, args.name, args.cols, args.fps)
    else:
        if not os.path.isdir("ascii_images"):
            os.mkdir("ascii_images")
        if args.cols is not None:
            ascii_image(args.file_name, args.name, args.cols)
        else:
            ascii_image(args.file_name, args.name, args.cols)


if __name__ == '__main__':
    worker = WorkerThread()
    progress = ProgressThread(worker)
    worker.start()
    progress.start()
    progress.join()
