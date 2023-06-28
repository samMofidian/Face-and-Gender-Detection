"""
    Final Project: Face & Gender Detection
    Ali Mofidian (sAm)
    Parsa Farahani
    Last Update: 6/28/2023
"""
import cv2


def run():
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Load the pre-trained gender classification model
    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt", "gender_net.caffemodel"
    )

    # Define the labels for gender classification
    gender_labels = ["Male", "Female"]

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Loop through each frame from the video stream
    while True:
        # Read a frame from the video stream
        _, frame = cap.read()

        # Convert the frame to grayscale
        """
        The reason for converting the frame to grayscale is because it simplifies the image by
        reducing the color information to a single channel. This makes it easier and
        faster to perform face detection, as well as reducing the complexity of the gender classification task.
        In face detection, the Haar cascade classifier requires the input image to be grayscale.
        This is because the algorithm is based on the difference between adjacent pixels,
        which can be calculated efficiently from the intensity values of the grayscale image
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        """
        In this code, the detectMultiScale method takes three arguments:
        - gray: The grayscale image in which to detect faces.
        - 1.3: Scale factor used to resize the image pyramid.
        In each iteration, the image size is reduced by a factor of 1.3.
        This allows the algorithm to detect faces at different scales.
        - 5: Minimum number of neighbor rectangles that should also be
        classified as positive for the current rectangle to be considered a face.
        A higher value increases the quality and reliability of detection but may miss some faces.
        
        The detectMultiScale method returns a list of rectangles representing the detected faces in the input image.
        Each rectangle contains the (x, y) coordinates of the top-left corner,
        as well as the width and height of the rectangle.
        The returned rectangles can then be used to extract the face region of interest (ROI) from the original
        image for further processing, such as gender classification in this code.
        """
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess the face ROI for gender classification
            """
            The cv2.dnn.blobFromImage() function takes an input image (face_roi in this example) and
            pre-processes it by resizing it to the desired size of 227x227 pixels, subtracting the
            mean values (78.4263377603, 87.7689143744, 114.895847746) from each pixel,
            and optionally swapping the red and blue color channels (swapRB=False).
            These operations are typically performed to normalize the input data and
            improve the accuracy of the neural network.
            """
            blob = cv2.dnn.blobFromImage(
                face_roi, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            # Pass the face ROI through the gender classification model
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()

            # Get the predicted gender label
            gender_pred = gender_labels[gender_preds[0].argmax()]

            # Draw a rectangle around the detected face and display the predicted gender label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, gender_pred, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2
            )

        # Display the frame
        cv2.imshow("Gender Recognition", frame)

        # Check for key press to exit the program
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
