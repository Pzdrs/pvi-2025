import cv2


def main():
    cap = cv2.VideoCapture('data/pvi_cv10_video_in.mp4')
    
    while True:
        ret, bgr = cap.read()

        if not ret:
            break
        
        

if __name__ == "__main__":
    main()