from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

solucao_maos = mp.solutions.hands
reconhecedor_maos = solucao_maos.Hands()

def verificar_gesto_polegar_cima(resultados):
    for mao in resultados.multi_hand_landmarks:
        polegar_ponta = mao.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        polegar_base = mao.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC]
        indicador_ponta = mao.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        if (polegar_ponta.y < indicador_ponta.y) and (polegar_ponta.y < polegar_base.y):
            return True
    return False

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lista_rostos = reconhecedor_rostos.process(frame_rgb)

    lista_maos = reconhecedor_maos.process(frame_rgb)

    resultado = {
        "rostos_detectados": False,
        "gesto_polegar_cima": False
    }

    if lista_rostos.detections:
        resultado["rostos_detectados"] = True

    if lista_maos.multi_hand_landmarks:
        if verificar_gesto_polegar_cima(lista_maos):
            resultado["gesto_polegar_cima"] = True

    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
