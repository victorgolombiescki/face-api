from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os

app = Flask(__name__)

# Inicializar a solução MediaPipe Pose
solucao_pose = mp.solutions.pose
reconhecedor_pose = solucao_pose.Pose()
desenho = mp.solutions.drawing_utils

# Pasta para salvar as imagens
IMAGES_FOLDER = 'imagens_salvas'
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Legendas para os pontos
LEGENDAS = {
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER: "Ombro Esquerdo",
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER: "Ombro Direito",
    mp.solutions.pose.PoseLandmark.LEFT_HIP: "Quadril Esquerdo",
    mp.solutions.pose.PoseLandmark.RIGHT_HIP: "Quadril Direito"
}

# Função para analisar a postura
def analisar_postura(pose_landmarks):
    # Posições dos pontos chave (exemplos: ombro, quadril, joelho)
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
        pontos_incorretos.append((ombro_esquerdo, LEGENDAS[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]))
        pontos_incorretos.append((ombro_direito, LEGENDAS[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]))
    
    # Verificar quadris
    if abs(quadril_esquerdo.y - quadril_direito.y) < 0.05:
        pontos_corretos["Quadris"] = "Alinhados"
    else:
        pontos_incorretos.append((quadril_esquerdo, LEGENDAS[mp.solutions.pose.PoseLandmark.LEFT_HIP]))
        pontos_incorretos.append((quadril_direito, LEGENDAS[mp.solutions.pose.PoseLandmark.RIGHT_HIP]))

    return alinhamento, pontos_corretos, pontos_incorretos

# Função para desenhar os pontos incorretos
def desenhar_pontos_incorretos(image, pontos):
    for ponto, legenda in pontos:
        cv2.circle(image, (int(ponto.x * image.shape[1]), int(ponto.y * image.shape[0])), 10, (0, 0, 255), -1)
        cv2.putText(image, legenda, (int(ponto.x * image.shape[1]) - 50, int(ponto.y * image.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Converter o frame de BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o frame para análise de pose
    resultado_pose = reconhecedor_pose.process(frame_rgb)

    # Verificar a postura
    if resultado_pose.pose_landmarks:
        alinhamento, pontos_corretos, pontos_incorretos = analisar_postura(resultado_pose.pose_landmarks)

        # Desenhar a pose no frame
        desenho.draw_landmarks(frame, resultado_pose.pose_landmarks, solucao_pose.POSE_CONNECTIONS)
        # Desenhar pontos incorretos
        desenhar_pontos_incorretos(frame, pontos_incorretos)

        # Salvar a imagem
        nome_arquivo = 'imagem_salva.jpg'
        caminho_arquivo = os.path.join(IMAGES_FOLDER, nome_arquivo)
        cv2.imwrite(caminho_arquivo, frame)

        return jsonify({
            "postura_ergonomicamente_correta": alinhamento,
            "pontos_corretos": pontos_corretos,
            "pontos_incorretos": {legenda: {"coordenadas": (ponto.x, ponto.y)} for ponto, legenda in pontos_incorretos},
            "imagem_salva": caminho_arquivo
        })

    return jsonify({"error": "Unable to detect pose in the provided image"})

if __name__ == '__main__':
    app.run(debug=True)
