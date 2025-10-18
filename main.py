import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Analyze emotions
        result = DeepFace.analyze(
            rgb_frame,
            actions=['emotion'],
            enforce_detection=False
        )

        # Handle both old and new DeepFace response formats
        if isinstance(result, list):
            # Older versions return a list
            dominant_emotion = result[0]['dominant_emotion']
        else:
            # Newer versions return a dict
            dominant_emotion = result['dominant_emotion']

        # Display the result
        cv2.putText(frame,
                    f'Emotion: {dominant_emotion}',
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

    except Exception as e:
        print("Error analyzing frame:", e)

    # Show video feed
    cv2.imshow('Emotion Detector', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
