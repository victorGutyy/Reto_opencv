# ejercicio 1 visualizacion de imagenes

import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen_path = "static/caballeros_1.jpg"

# Cargar la imagen en color
imagen_color = cv2.imread(imagen_path, cv2.IMREAD_COLOR)

# Cargar la imagen en escala de grises
imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

# Mostrar la imagen en color
cv2.imshow("Imagen en Color", imagen_color)

# Mostrar la imagen en escala de grises
cv2.imshow("Imagen en Escala de Grises", imagen_gris)

# Esperar hasta que el usuario presione una tecla para cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

#transformacion geometrica ejercicio 2 

imagen_path = "static/caballeros_2.jpg"

# Cargar la imagen original
imagen = cv2.imread(imagen_path)

# Rotar la imagen 90 grados en sentido horario
# Usamos rotación y traslación con cv2.getRotationMatrix2D
alto, ancho = imagen.shape[:2]
centro = (ancho // 2, alto // 2)
matriz_rotacion = cv2.getRotationMatrix2D(centro, -90, 1)  # Ángulo -90 para sentido horario
imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto))

# Escalar la imagen al doble de su tamaño original
imagen_escalada = cv2.resize(imagen, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)

# Mostrar las imágenes originales, rotadas y escaladas
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen Rotada 90°", imagen_rotada)
cv2.imshow("Imagen Escalada x2", imagen_escalada)

# Esperar hasta que el usuario presione una tecla para cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

#ejercicio 3filtrado

imagen_path = "static/supercampeones.jpg"

# Cargar la imagen en escala de grises
imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

# Aplicar filtro Gaussiano (suavizado)
imagen_gaussiana = cv2.GaussianBlur(imagen_gris, (5, 5), 1.5)

# Aplicar filtro de realce (usando un kernel de enfoque)
kernel_realce = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]], dtype=np.float32)
imagen_realzada = cv2.filter2D(imagen_gaussiana, -1, kernel_realce)

# Mostrar las imágenes originales, suavizadas y realzadas
plt.figure(figsize=(12, 8))

# Imagen original
plt.subplot(1, 3, 1)
plt.title("Imagen Original (Escala de Grises)")
plt.imshow(imagen_gris, cmap='gray')
plt.axis("off")

# Imagen suavizada (filtro Gaussiano)
plt.subplot(1, 3, 2)
plt.title("Imagen Suavizada (Filtro Gaussiano)")
plt.imshow(imagen_gaussiana, cmap='gray')
plt.axis("off")

# Imagen con filtro de realce
plt.subplot(1, 3, 3)
plt.title("Imagen con Filtro de Realce")
plt.imshow(imagen_realzada, cmap='gray')
plt.axis("off")

# Mostrar gráficos
plt.tight_layout()
plt.show()

#ejercicio 4 deteccion de bordes

imagen_path = "static/pokemon.jpg"

# Cargar la imagen en escala de grises
imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

# Aplicar el operador Sobel para detectar bordes en el eje X
sobel_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)

# Aplicar el operador Sobel para detectar bordes en el eje Y
sobel_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud de los bordes combinando sobel_x y sobel_y
bordes = cv2.magnitude(sobel_x, sobel_y)

# Normalizar la magnitud a un rango de 0 a 255 para visualizar mejor
bordes = cv2.convertScaleAbs(bordes)

# Mostrar las imágenes originales y con bordes detectados
plt.figure(figsize=(12, 6))

# Imagen original
plt.subplot(1, 2, 1)
plt.title("Imagen Original (Escala de Grises)")
plt.imshow(imagen_gris, cmap='gray')
plt.axis("off")

# Imagen con bordes detectados
plt.subplot(1, 2, 2)
plt.title("Bordes Detectados (Sobel)")
plt.imshow(bordes, cmap='gray')
plt.axis("off")

# Mostrar los gráficos
plt.tight_layout()
plt.show()