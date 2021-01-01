from CropUtil import CropUtil
from CascadeDetect import Detect
from Classify import Classify
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script for anime_face_track')
    parser.add_argument("command",
                        metavar="<command>",
                        help="Please choose 'capture', 'detect', 'train-detector', 'classify', or 'train-classify'")
    parser.add_argument('--output', required=True,
                        metavar="/path/to/anime_dir/",
                        help='Directory to write all output data. I recommend to create a new directory for each anime')
    parser.add_argument('--frames', required=False,
                        default=0.2,
                        metavar='<frames>',
                        help='1 / (desired frame rate). For example, to capture at 5 fps: --frames 0.2.')
    parser.add_argument('--threads', required=False,
                        default=1,
                        metavar='<threads>',
                        help='Number of threads to use for image processing')
    parser.add_argument('--epoch',
                        required=False,
                        default=100,
                        metavar='<epoch>',
                        help='Number of epochs for training')
    parser.add_argument('--model',
                        required=False,
                        default=None,
                        metavar='/path/to/model.hd5',
                        help='Model to use or update')
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output):
        pass
    else:
        os.mkdir(args.output)

    if args.command == 'capture':
        print('capture')
        cropper = CropUtil(args.output, args.frames)
        print('Outputting to ', cropper.image_path)
        # Screen cap each frame
        cropper.capture()

    elif args.command == 'detect':
        print('detect')
        # Import captured pngs to cascade classifier
        detector = Detect(args.output, threads=args.threads)
        print('Reading from ', detector.input_path)
        detector.find_face()

    elif args.command == 'train-detector':
        print('train-detector')

    elif args.command == 'classify':
        print('classify')
        classifier = Classify(args.output, model_path=args.model)
        print('Reading from ', classifier.input_path)
        print('Model: ', classifier.model_path)
        classifier.importModel()
        classifier.predict()

    elif args.command == 'train-classify':
        print('train-classify')
        classifier = Classify(args.output, model_path=args.model)

        print('Reading from ', classifier.train_path)
        print('Writing model to ', os.path.join(classifier.path, 'transfer_model.h5'))

        classifier.preprocess()
        classifier.baseModel()
        # Train model
        classifier.train()

    else:
        print('Please input a correct command')
        quit()
