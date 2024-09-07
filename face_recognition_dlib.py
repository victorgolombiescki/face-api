import cv2
import face_recognition

def load_known_faces(known_face_encodings, known_face_names):
    # Adicione aqui as imagens de rostos conhecidos e seus nomes correspondentes
    image_path = "path/to/known_face.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erro ao carregar a imagem {image_path}")
        return
    
    # Converte a imagem para RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_encoding = face_recognition.face_encodings(rgb_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append("Nome do Cadastro")

def recognize_face_from_webcam(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # Redimensionar o frame do vídeo para 1/4 do tamanho para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            for (top, right, bottom, left), name in zip(face_locations, known_face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_face_encodings = []
    known_face_names = []
    load_known_faces(known_face_encodings, known_face_names)
    recognize_face_from_webcam(known_face_encodings, known_face_names)
