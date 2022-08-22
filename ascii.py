import shutil
import sys, os, argparse
import converter
import shutil
from threading import Thread
import time


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
        if os.path.isdir(args.name + "_frames"):
            shutil.rmtree(args.name + "_frames")
        if os.path.isdir(args.name + "_ascii_frames"):
            shutil.rmtree(args.name + "_ascii_frames")
        if args.cols is None or args.fps is None:
            if args.cols is None and args.fps is None:
                converter.to_ascii_vid(args.file_name, args.name, )
            elif args.cols is not None:
                converter.to_ascii_vid(args.file_name, args.name, cols=args.cols)
            else:
                converter.to_ascii_vid(args.file_name, args.name, fps=args.fps)
        else:
            converter.to_ascii_vid(args.file_name, args.name, args.cols, args.fps)


if __name__ == '__main__':
    worker = WorkerThread()
    progress = ProgressThread(worker)
    worker.start()
    progress.start()
    progress.join()
