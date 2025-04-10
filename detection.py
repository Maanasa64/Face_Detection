import cv2

def detect_faces(image_path, output_path=None):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Adjusted parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # More thorough scaling
        minNeighbors=4,    # Fewer neighbors to reduce false negatives
        minSize=(20, 20),  # Smaller faces allowed
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(image, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, image)

    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces('input_image2.jpg', 'output.jpg')