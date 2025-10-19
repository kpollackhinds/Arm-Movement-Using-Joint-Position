import cv2

# Try different camera indices (0, 1, 2, etc.) to find your external camera
# 0 is usually the default webcam, 1 or higher might be your external camera
cap = cv2.VideoCapture(2) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the captured frame
    cv2.imshow('External Camera Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()