import torch
import onnx
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # add yolov5/ to path

from models.common import DetectMultiBackend  # YOLOv5 모델 가져오기
from utils.torch_utils import select_device

def export_to_onnx(weights='yolov5s.pt', img_size=(640, 640), batch_size=1, device='cuda'):
    # 장치 설정
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fuse=True)  # 모델 로드 및 장치에 할당
    model.eval()

    # 더미 입력 생성
    dummy_input = torch.zeros(batch_size, 3, *img_size).to(device)

    # ONNX로 내보내기
    onnx_file = "yolov5.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX 모델이 '{onnx_file}'로 저장되었습니다.")
    return onnx_file

# 실행
onnx_file = export_to_onnx(weights='yolov5s.pt', img_size=(640, 640))