import cv2
for i in range(3):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    print(f"Index {i} Success:", ret)
    cap.release()