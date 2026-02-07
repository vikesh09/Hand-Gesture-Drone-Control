import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

label = input("Gesture label (like forward, left, stop): ")

with open("hand_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                row = []
                for lm in handLms.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                row.append(label)
                writer.writerow(row)
                print("Saved")

        cv2.imshow("Collecting", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
