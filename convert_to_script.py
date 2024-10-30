import argparse
import torch

from torch.jit import optimize_for_inference
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='yolo11n.pt', type=str, help='Trained state_dict file path to open')
parser.add_argument('--image_size', default=[640, 640], help='Input image size')
parser.add_argument('--cpu', action='store_true', default=True, help='Use cpu inference')
args = parser.parse_args()


if __name__ == '__main__':

    # Load a model
    model = YOLO(args.trained_model)
    device = 'cpu' if args.cpu else 'cuda'
    model.export(format='torchscript', imgsz=args.image_size, device=device)


    


