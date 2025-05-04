from . import train

def main():
    train.run(
        data='greenball_dataset/green_ball.yaml',
        imgsz=416,
        batch_size=4,
        epochs=10,
        weights='yolov5s.pt',
        device='cpu',
        name='greenball_yolov5_cpu'
    )

if __name__ == '__main__':
    main()
