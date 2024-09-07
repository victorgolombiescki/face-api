import face_recognition
import cv2
import numpy as np
from PIL import Image

# Função para carregar e converter a imagem usando Pillow
def load_image_with_pillow(file_path):
    image = Image.open(file_path).convert('RGB')
    image_np = np.array(image)
    print(f"Imagem carregada e convertida com Pillow ({file_path}): Tipo {image_np.dtype}, Forma {image_np.shape}")
    
    # Salvar a imagem convertida para verificação
    converted_image = Image.fromarray(image_np)
    converted_image.save(file_path.replace(".jpg", "_converted.jpg"))
    
    return image_np

# Caminho da imagem conhecida
known_image_path = "conhecida1.jpg"

# Carrega a imagem conhecida
known_image = load_image_with_pillow(known_image_path)

# Verifica o formato da imagem conhecida e tenta codificar a face
print(f"Verificando formato da imagem conhecida: {known_image.dtype}, {known_image.shape}")

# Verificar se a imagem é de 8 bits
if known_image.dtype != np.uint8:
    raise ValueError("A imagem conhecida não é de 8 bits. Por favor, converta a imagem para um formato compatível.")

try:
    known_image_encodings = face_recognition.face_encodings(known_image)
    if not known_image_encodings:
        raise ValueError("Não foram encontradas faces na imagem conhecida.")
    known_image_encoding = known_image_encodings[0]
except Exception as e:
    print(f"Erro ao codificar faces na imagem conhecida: {e}")
    known_image_encoding = None

if known_image_encoding is None:
    print("Não foi possível codificar a imagem conhecida. Verifique o formato da imagem e tente novamente.")
else:
    # Captura a imagem da câmera
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera")

    print("Pressione 'q' para capturar a imagem da câmera e comparar")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Falha ao capturar a imagem da câmera")
            continue

        # Mostrar o frame da câmera
        cv2.imshow('Video', frame)

        # Pressione 'q' para capturar a imagem e comparar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Converte o frame para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"Frame capturado convertido para RGB: Tipo {rgb_frame.dtype}, Forma {rgb_frame.shape}")

            # Codifica a face na imagem capturada
            try:
                unknown_image_encodings = face_recognition.face_encodings(rgb_frame)
                if not unknown_image_encodings:
                    print("Não foram encontradas faces na imagem capturada.")
                    continue

                unknown_image_encoding = unknown_image_encodings[0]

                # Compara a imagem conhecida com a imagem capturada
                results = face_recognition.compare_faces([known_image_encoding], unknown_image_encoding)

                if results[0]:
                    print("As faces correspondem!")
                else:
                    print("As faces não correspondem.")
            except Exception as e:
                print(f"Erro ao codificar faces na imagem capturada: {e}")
            break

    # Libera a câmera e fecha a janela
    video_capture.release()
    cv2.destroyAllWindows()
