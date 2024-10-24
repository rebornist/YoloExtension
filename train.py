import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='yolo11n.pt', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./weights/', type=str, help='Dir to save txt results')
parser.add_argument('--data', default='./data/widerface.yaml', help='Config yaml save path')
parser.add_argument('--batch', default=16, help='Batch size for training')
parser.add_argument('--epochs', default=3, help='Number of epochs')
parser.add_argument('--imgsz', default=640, help='Image size for training')
parser.add_argument('--device', default='cpu', help='Device for training')
parser.add_argument('--save', default=True, help='Save model')
parser.add_argument('--pretrained', default=True, help='Use pre-trained model')
parser.add_argument('--val', default=True, help='Validate model during training')
args = parser.parse_args()

# Load a model
model = YOLO(args.trained_model)

results = model.train(
    data=args.data, 
    epochs=args.epochs, 
    batch=args.batch, 
    imgsz=args.imgsz, 
    device=args.device, 
    save=args.save, 
    pretrained=args.pretrained, 
    val=args.val
)