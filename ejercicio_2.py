import cv2
import numpy as np

# Ruta al archivo de Haar Cascade para detección de rostros
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Cargar el modelo Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Inicializa la captura de vídeo (0 es para la cámara principal)
cap = cv2.VideoCapture(0)

# Comienza la detección
print("Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Convierte el frame a escala de grises para mejorar el rendimiento
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostros
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el vídeo con las detecciones
    cv2.imshow('Detección de Rostros', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
