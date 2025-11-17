import cv2, time

cam = cv2.VideoCapture(0)
TARGET_FPS = 10
PERIOD = 1.0 / TARGET_FPS
last = 0

while True:
    now = time.time()
    if now - last < PERIOD:
        continue
    last = now

    ret, frame = cam.read()
    if not ret:
        break

    # frame 10 FPS thật
    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break
