import cv2
import mediapipe as mp

# Inicializar a captura de vídeo
webcam = cv2.VideoCapture(0)

# Inicializar soluções do MediaPipe para detecção de rostos e mãos
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

solucao_maos = mp.solutions.hands
reconhecedor_maos = solucao_maos.Hands()

# Função para verificar o gesto de positivo
def verificar_gesto_polegar_cima(resultados):
    for mao in resultados.multi_hand_landmarks:
        # Posições dos pontos da mão
        polegar_ponta = mao.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        polegar_base = mao.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC]
        indicador_ponta = mao.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Verificar se o polegar está acima do resto da mão
        if (polegar_ponta.y < indicador_ponta.y) and (polegar_ponta.y < polegar_base.y):
            return True
    return False

while True:
    verificador, frame = webcam.read()
    if not verificador:
        break

    # Converter o frame de BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o frame para reconhecimento de rostos
    lista_rostos = reconhecedor_rostos.process(frame_rgb)
    
    # Processar o frame para reconhecimento de mãos
    lista_maos = reconhecedor_maos.process(frame_rgb)

    # Desenhar as detecções de rosto
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    # Desenhar as detecções de mãos e verificar o gesto de positivo
    if lista_maos.multi_hand_landmarks:
        for mao_landmarks in lista_maos.multi_hand_landmarks:
            desenho.draw_landmarks(frame, mao_landmarks, solucao_maos.HAND_CONNECTIONS)
        if verificar_gesto_polegar_cima(lista_maos):
            cv2.putText(frame, "Gesto de Positivo Detectado!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar o frame processado
    cv2.imshow("Reconhecimento facial e de gestos", frame)

    # Sair do loop quando a tecla 'ESC' for pressionada
    if cv2.waitKey(5) == 27:
        break

# Liberar a captura de vídeo e destruir todas as janelas
webcam.release()
cv2.destroyAllWindows()
