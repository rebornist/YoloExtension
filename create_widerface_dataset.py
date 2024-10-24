import argparse
import cv2
import os


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--dataset_path', default='../datasets/widerface/wider_face_split/wider_face_train_bbx_gt.txt', help='Dataset Labels path')
parser.add_argument('--save_config_path', default='./core/ultralytics/cfg/datasets/widerface.yaml', help='yaml save path')
parser.add_argument('--image_folder', default='../datasets/widerface/train/images', help='Image folder path')
parser.add_argument('--save_folder', default='../datasets/widerface/train/labels', type=str, help='Dir to save txt results')
args = parser.parse_args()


def create_config_file():    
    """
    Config 파일 생성
    """
    
    config_txt = '''
    path: ../datasets/widerface  # dataset root dir

    train: train/images # train images (relative to 'path')
    val: valid/images # val images (relative to 'path')
    test:  test/images # test images (optional)

    # Classes
    nc: 1
    names: ['face']
    '''

    with open("./core/ultralytics/cfg/datasets/widerface.yaml", 'w') as f:
        f.write(config_txt)


def create_label_file(line, data):
    """
    YOLO Label Format으로 변환하여 Label 파일 생성
    """
    # 이미지 파일 경로
    image_path = os.path.join(args.image_folder, line)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # YOLO Label로 변환
    label_data = []
    for d in data:
        d_arr = d.split(' ')
        class_id = d_arr[0]
        x_center, y_center, width, height = convert_to_yolo_label(image_path, d_arr)
        label_data.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Label 파일 경로
    label_file = os.path.join(args.save_folder, line.replace('.jpg', '.txt'))
    label_dir = os.path.dirname(label_file)
    
    # 경로가 존재하지 않으면 디렉터리 생성
    os.makedirs(label_dir, exist_ok=True)

    # YOLO Label로 변환
    with open(label_file, 'w') as f:
        f.write('\n'.join(label_data))



def convert_to_yolo_label(image_path, data):
    """
    Convert Widerface Label to YOLO Label
    <object-class> <x_center> <y_center> <width> <height>
    """

    # 이미지를 읽음
    image = cv2.imread(image_path)

    # 이미지 크기 추출
    img_height, img_width = image.shape[:2]

    # 바운딩 박스의 좌상단 좌표 (x1, y1)와 우하단 좌표 (x2, y2)
    x1, y1, w, h = map(int, data[1:])
    x2 = x1 + w
    y2 = y1 + h

    # x_center, y_center, width, height 계산
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = w / img_width
    height = h / img_height

    return x_center, y_center, width, height



def main():
    """
    Create Widerface Dataset
    """
    current_image = None
    current_labels = []

    # Create Config File
    create_config_file()

    # Create Label Folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Read Dataset Label
    with open(args.dataset_path, 'r') as f:
        lines = f.readlines()

    # Create Train Label
    for line in lines:
        line = line.strip()
        if line.endswith('.jpg'):
            if current_image is not None:
                create_label_file(current_image, current_labels)
                current_labels = []

            current_image = line
        else:
            data = line.split(' ')
            if len(data) > 3:
                current_labels.append('0 ' + ' '.join(data[:4]))
            

if __name__ == '__main__':
    main()