import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Step 1: Collecting Face Data (Capture Faces)
def collect_face_data():
    face_classifier = cv2.CascadeClassifier(
        'C:/Users/Mukul/Desktop/face_detection_recognition/File/haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0
    dataset_path = 'C:/Users/Mukul/Desktop/face_detection_recognition/dataset/'

    # Prompt for user name at the start
    user_name = input("Enter your name: ")

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = dataset_path + user_name + '_' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")

        if cv2.waitKey(1) == 13 or count == 100:  # Stop after 100 images or Enter key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Face data collection completed.")

# Step 2: Train the model using collected data
def train_face_recognizer():
    data_path = 'C:/Users/Mukul/Desktop/face_detection_recognition/dataset/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels, Names = [], [], []
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Extract name from the filename (before the first underscore)
        name = files.split('_')[0]  # Name is the first part of the filename
        if name not in Names:
            Names.append(name)

        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(Names.index(name))  # Use the index of the name as the label

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model.save('face_model.yml')
    print("Dataset Model Training Completed.")

    # Return the names list so we can use it for recognition later
    return Names

# Step 3: Recognize Faces using the trained model
def recognize_faces(names):
    face_classifier = cv2.CascadeClassifier(
        'C:/Users/Mukul/Desktop/face_detection_recognition/File/haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return img, []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('face_model.yml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))

            if confidence > 82:
                # Map the label back to the name
                name = names[result[0]]  # Use the label index to get the corresponding name
                cv2.putText(image, f"Name: {name}", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Face Cropper', image)
            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)

        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1) == 13:  # Enter key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Input: Number of users to collect face data for
    num_users = int(input("Enter the number of users to collect face data for: "))

    # Loop to collect face data for each user
    for i in range(1, num_users + 1):
        print(f"Collecting face data for User {i}...")
        collect_face_data()

    # Train the model (run this after collecting data)
    print("Training the face recognition model...")
    names = train_face_recognizer()

    # Recognize faces using the trained model
    print("Starting face recognition...")
    recognize_faces(names)
