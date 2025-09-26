import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # try 0, 1, 2, 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow('Test DroidCam Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
