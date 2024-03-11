'''
Face Detection and Recognition System (CODSOFT INTERNSHIP by Dhrubojyoti Chakraborty)

'''
import cv2
import face_recognition

# a sample image or video
image_source = "img1.jpg"
cap = cv2.VideoCapture(image_source)

# a pre-trained face recognition model
# This model is based on the dlib library and uses a deep neural network
known_face_encodings = []
known_face_names = []

# Set the screen size for display
screen_width = 800
screen_height = 600

# Main loop for processing frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the desired size
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame using the face_recognition library
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Display the name near the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the processed frame
    cv2.imshow('Face Detection and Recognition System', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cap.release()

# Add a delay (in milliseconds) to keep the window open
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
