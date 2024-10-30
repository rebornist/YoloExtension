import numpy as np
import torch

# trtexec --onnx=yolov5.onnx --saveEngine=yolov5.trt --minShapes=input:1x3x544x960 --optShapes=input:4x3x544x960 --maxShapes=input:4x3x544x960 --fp16

def convert_trt_output_to_yolo_format(output_cov, output_bbox, num_classes=80):
    """
    TensorRT 모델의 출력을 YOLOv5의 예측 형식으로 변환합니다.
    output_cov: 클래스 확률에 대한 Sigmoid 처리 출력
    output_bbox: 바운딩 박스 좌표에 대한 BiasAdd 처리 출력
    num_classes: YOLOv5에서 사용하는 클래스 수 (기본값: 80)
    """
    # output_cov는 클래스 확률을 포함하는 Sigmoid 텐서로, [4, 34, 60] 형태입니다.
    # output_bbox는 바운딩 박스 좌표 텐서로, [16, 34, 60] 형태입니다.
    
    # Tensor 크기 맞추기 및 변환
    batch_size = output_cov.shape[0]
    grid_h, grid_w = output_cov.shape[1:3]

    # YOLOv5 형식의 텐서로 결합 (배치 크기, 그리드 높이, 그리드 너비, 좌표 및 클래스 수)
    output_cov_reshaped = output_cov.reshape(batch_size, grid_h, grid_w, num_classes)
    output_bbox_reshaped = output_bbox.reshape(batch_size, grid_h, grid_w, 4)  # x, y, w, h

    # 두 출력을 결합하여 최종 YOLO 텐서를 생성
    yolo_output = np.concatenate((output_bbox_reshaped, output_cov_reshaped), axis=-1)
    
    # 필요 시 Torch 텐서로 변환하여 반환
    return torch.from_numpy(yolo_output)

# 예시 입력 (여기서는 더미 데이터를 사용)
output_cov = np.random.rand(4, 34, 60).astype(np.float32)  # 실제 TensorRT 출력으로 대체
output_bbox = np.random.rand(16, 34, 60).astype(np.float32)  # 실제 TensorRT 출력으로 대체

# 변환
yolo_output = convert_trt_output_to_yolo_format(output_cov, output_bbox)
print(yolo_output.shape)  # (4, 34, 60, num_classes + 4)