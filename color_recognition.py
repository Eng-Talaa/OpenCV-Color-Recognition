import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =========================
    # Red Color
    # =========================

    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = red_mask1 + red_mask2

    # =========================
    # Green Color
    # =========================

    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # =========================
    # Blue Color
    # =========================

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove Noise
    kernel = np.ones((5,5), np.uint8)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # =========================
    # Red Detection
    # =========================

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 500:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

            cv2.putText(frame, "Red",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2)

    # =========================
    # Green Detection
    # =========================

    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 100:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(frame, "Green",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

    # =========================
    # Blue Detection
    # =========================

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 500:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

            cv2.putText(frame, "Blue",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,0,0),
                        2)

    # Show Windows
    cv2.imshow("Original", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("Blue Mask", blue_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()