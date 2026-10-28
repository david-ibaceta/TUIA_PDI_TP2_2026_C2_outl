# Librerías necesarias
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def buscar_monedas(imagen_original):
    # Copiamos la imagen para no modificar la original
    imagen_circulos = imagen_original.copy()

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen_circulos, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque Gaussiano más fuerte para eliminar ruido
    gris_suave = cv2.medianBlur(gris, 7)

    # Aplicar la Transformada de Hough para Círculos
    circulos = cv2.HoughCircles(
        gris_suave,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=40,
        minRadius=140,
        maxRadius=176
    )

    # Dibujar los círculos detectados
    if circulos is not None:
        # Convertir las coordenadas y los radios a enteros
        circulos = np.uint16(np.around(circulos))

        for circulo in circulos[0, :]:
            centro_x, centro_y, radio = circulo[0], circulo[1], circulo[2]
            # Dibujar el contorno del círculo en verde
            cv2.circle(imagen_circulos, (centro_x, centro_y), radio, (0, 255, 0), 3)
            # Dibujar el centro del círculo en rojo
            cv2.circle(imagen_circulos, (centro_x, centro_y), 3, (0, 0, 255), 4)

        print(f"Se detectaron {len(circulos[0])} monedas/círculos.")
    else:
        print("No se detectaron círculos con los parámetros actuales.")

    # Retornar la imagen resultante
    return imagen_circulos

def buscar_dados(imagen_original):
    # Copiamos la imagen para no modificar la original
    imagen_lineas = imagen_original.copy()

    # Convertir a escala de grises y suavizar
    gris = cv2.cvtColor(imagen_lineas, cv2.COLOR_BGR2GRAY)
    gris_suave = cv2.GaussianBlur(gris, (3, 3), 0)

    # Crear una máscara binaria de ceros (negra) del mismo tamaño que la imagen
    mascara = np.zeros_like(gris_suave)

    # Definir las coordenadas del rectángulo que encierra los dados (Región de Interés - ROI)
    alto, ancho = gris_suave.shape
    ymin, ymax = int(alto * 0.1), int(alto * 0.5)
    xmin, xmax = int(ancho * 0.2), int(ancho * 0.8)

    # Dibujamos un rectángulo blanco (255) en la máscara
    mascara[ymin:ymax, xmin:xmax] = 255

    # Detectar bordes con Canny de forma global
    bordes = cv2.Canny(gris_suave, 50, 230, apertureSize=3)

    # Aplicar la máscara sobre los bordes
    bordes_enmascarados = cv2.bitwise_and(bordes, bordes, mask=mascara)

    # Aplicar la Transformada de Hough Probabilística
    lineas = cv2.HoughLinesP(
        bordes_enmascarados,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=120,
        maxLineGap=8
    )

    # Dibujar un rectángulo azul para visualizar el área delimitada por la máscara
    cv2.rectangle(imagen_lineas, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)

    # Dibujar las líneas detectadas (en verde) sobre la imagen
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea.ravel()
            cv2.line(imagen_lineas, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print(f"Se detectaron {len(lineas)} segmentos de línea recta dentro del área enmascarada.")
    else:
        print("No se detectaron líneas en la región seleccionada.")

    # Retornar la imagen resultante
    return imagen_lineas

def main():
    ruta_imagen = './TP_2/git/TUIA_PDI_TP2_2026_C2_outl/monedas.jpg'
    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        print("Error: No se pudo cargar la imagen para el proceso principal.")
        return

    print("--- Buscando Monedas (Círculos) ---")
    resultado_monedas = buscar_monedas(imagen)

    print("\n--- Buscando Dados (Líneas) ---")
    resultado_dados = buscar_dados(imagen)

    figura, ejes = plt.subplots(1, 2, figsize=(14, 7))
    ejes[0].imshow(cv2.cvtColor(resultado_monedas, cv2.COLOR_BGR2RGB))
    ejes[0].set_title("Monedas")
    ejes[0].axis("off")

    ejes[1].imshow(cv2.cvtColor(resultado_dados, cv2.COLOR_BGR2RGB))
    ejes[1].set_title("Dados")
    ejes[1].axis("off")

    figura.tight_layout()
    plt.show()

# Ejecutar el flujo principal
if __name__ == '__main__':
    main()
