import cv2
import mediapipe as mp

# Inicializar a captura de vídeo
webcam = cv2.VideoCapture(0)

# Inicializar soluções do MediaPipe para detecção de rostos, mãos e pose
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

solucao_maos = mp.solutions.hands
reconhecedor_maos = solucao_maos.Hands()

solucao_pose = mp.solutions.pose
reconhecedor_pose = solucao_pose.Pose()

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

# Função para analisar a postura corporal
def analisar_postura(pose_landmarks):
    # Posições dos pontos chave da pose
    ombro_esquerdo = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    ombro_direito = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    quadril_esquerdo = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    quadril_direito = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    
    # Verificar se os ombros estão alinhados com os quadris (postura básica)
    alinhamento = abs(ombro_esquerdo.y - ombro_direito.y) < 0.05 and abs(quadril_esquerdo.y - quadril_direito.y) < 0.05
    
    pontos_corretos = {}
    pontos_incorretos = []
    
    # Verificar ombros
    if abs(ombro_esquerdo.y - ombro_direito.y) < 0.05:
        pontos_corretos["Ombros"] = "Alinhados"
    else:
        pontos_incorretos.append((ombro_esquerdo, "Ombro Esquerdo"))
        pontos_incorretos.append((ombro_direito, "Ombro Direito"))
    
    # Verificar quadris
    if abs(quadril_esquerdo.y - quadril_direito.y) < 0.05:
        pontos_corretos["Quadris"] = "Alinhados"
    else:
        pontos_incorretos.append((quadril_esquerdo, "Quadril Esquerdo"))
        pontos_incorretos.append((quadril_direito, "Quadril Direito"))

    return alinhamento, pontos_corretos, pontos_incorretos

# Função para desenhar os pontos da pose corporal
def desenhar_pontos_postura(image, pose_landmarks, pontos_incorretos):
    # Desenhar todos os pontos da pose
    desenho.draw_landmarks(image, pose_landmarks, solucao_pose.POSE_CONNECTIONS)
    
    # Desenhar pontos incorretos com legendas
    for ponto, legenda in pontos_incorretos:
        cv2.circle(image, (int(ponto.x * image.shape[1]), int(ponto.y * image.shape[0])), 10, (0, 0, 255), -1)
        cv2.putText(image, f"Errado: {legenda}", (int(ponto.x * image.shape[1]) - 50, int(ponto.y * image.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Escrever pontos corretos na imagem
    if "Ombros" in pontos_corretos:
        cv2.putText(image, f"Correto: Ombros {pontos_corretos['Ombros']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if "Quadris" in pontos_corretos:
        cv2.putText(image, f"Correto: Quadris {pontos_corretos['Quadris']}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    # Processar o frame para análise de pose
    resultado_pose = reconhecedor_pose.process(frame_rgb)

    # Desenhar as detecções de rosto
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    # Desenhar as detecções de mãos e verificar o gesto de positivo
    if lista_maos.multi_hand_landmarks:
        for mao_landmarks in lista_maos.multi_hand_landmarks:
            desenho.draw_landmarks(frame, mao_landmarks, solucao_maos.HAND_CONNECTIONS)
        if verificar_gesto_polegar_cima(lista_maos):
            cv2.putText(frame, "Gesto de Positivo Detectado!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Analisar e desenhar a pose corporal
    if resultado_pose.pose_landmarks:
        alinhamento, pontos_corretos, pontos_incorretos = analisar_postura(resultado_pose.pose_landmarks)
        desenhar_pontos_postura(frame, resultado_pose.pose_landmarks, pontos_incorretos)

    # Mostrar o frame processado
    cv2.imshow("Reconhecimento facial, de gestos e de postura", frame)

    # Sair do loop quando a tecla 'ESC' for pressionada
    if cv2.waitKey(5) == 27:
        break

# Liberar a captura de vídeo e destruir todas as janelas
webcam.release()
cv2.destroyAllWindows()
