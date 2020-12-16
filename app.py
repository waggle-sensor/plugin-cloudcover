import time
import argparse

import cv2
import torch
from PIL import Image
import numpy as np

import waggle.plugin as plugin
from waggle.data import open_data_source

TOPIC_INPUT_IMAGE = "sky_image"
TOPIC_CLOUDCOVER = "env.coverage.cloud"

plugin.init()


def run(args):
    print("Loading {} module".format(args.model))
    model_module = __import__(args.model)

    print("Loading model {}...".format(model_module.model_path))
    if torch.cuda.is_available():
        print("CUDA is available")
        model = torch.load(model_module.model_path)
        model.to('cuda')
    else:
        print("CUDA is not avilable; use CPU")
        model = torch.load(model_module.model_path, map_location=torch.device('cpu'))
    model.eval()

    # print("Cut-out confidence level is set to {:.2f}".format(args.confidence_level))
    sampling_countdown = -1
    if args.sampling_interval >= 0:
        print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
        sampling_countdown = args.sampling_interval
    print("Cloud cover estimation starts...")
    while True:
        with open_data_source(id=TOPIC_INPUT_IMAGE) as cap:
            timestamp, image = cap.get()

            tensor = model_module.preprocess(image)
            tensor = torch.unsqueeze(tensor, 0)
            tensor = tensor.float()
            if torch.cuda.is_available():
                tensor = tensor.cuda()

            with torch.no_grad():
                score = model(tensor)

            ratio = model_module.postprocess(score)
            plugin.publish(TOPIC_CLOUDCOVER, ratio, timestamp=timestamp)
            if args.debug:
                print(f"Cloud coverage: {ratio}")
                print(f"Measures published")

            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                cv2.imwrite('/tmp/sample.jpg', image)
                plugin.upload_file('/tmp/sample.jpg')
                if args.debug:
                    print("A sample is published")
                # Reset the count
                sampling_countdown = args.sampling_interval

            if args.interval > 0:
                time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-model', dest='model',
        action='store', default='unet_module',
        help='Path to model')
    parser.add_argument(
        '-image-size', dest='image_size',
        action='store', default=300, type=int,
        help='Input image size')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    run(parser.parse_args())
