import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def mostrar_etapas(titulo_ventana, imagenes, titulos):
    fig = plt.figure(figsize=(15, 5))
    fig.canvas.manager.set_window_title(titulo_ventana)
    for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
        plt.subplot(1, len(imagenes), i + 1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(titulo)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def detectar_y_segmentar(img_path):
    print(f"\n--- Procesando {img_path} ---")
    img = cv2.imread(img_path)
    if img is None:
        return

    # ==========================================
    # PARTE A: Detección Robusta (Basada en Proporciones)
    # ==========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavizado para eliminar ruido pero conservar bordes
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Buscar contornos sin jerarquía
    contornos, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:30]
    
    x_p, y_p, w_p, h_p = 0, 0, 0, 0
    patente_encontrada = False
    
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        # Una patente argentina tiene una proporción ancho/alto de entre 2.0 y 4.5
        if 2.0 < aspect_ratio < 4.5 and w * h > 1500:
            # Comprobación de solidez para evitar agarrar parrillas o faros
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            rect_area = w * h
            if hull_area / rect_area > 0.7:  # Debe ser una forma mayormente rectangular
                x_p, y_p, w_p, h_p = x, y, w, h
                patente_encontrada = True
                break

    if not patente_encontrada:
        print("No se detectó la patente.")
        return

    img_con_contorno = img.copy()
    cv2.rectangle(img_con_contorno, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 255, 0), 3)

    patente_recortada = img[y_p:y_p+h_p, x_p:x_p+w_p]
    patente_gray = gray[y_p:y_p+h_p, x_p:x_p+w_p]

    mostrar_etapas(f"Parte A - {img_path}", 
                   [img, edged, img_con_contorno, patente_recortada], 
                   ['Original', 'Bordes', 'Detección', 'Patente Segmentada'])

    # ==========================================
    # PARTE B: Segmentación Inmune a Sombras
    # ==========================================
    
    # 1. Umbralización Adaptativa (Fundamental para contrarrestar sombras)
    # Convierte los caracteres negros en blancos para findContours
    binaria = cv2.adaptiveThreshold(patente_gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 5)
    
    # 2. Pequeño filtro morfológico para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contornos_chars, _ = cv2.findContours(binaria_limpia.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patente_con_caracteres = patente_recortada.copy()
    caracteres_validos = []
    alto_placa = h_p

    # 3. Filtrar los contornos de los caracteres
    for c in contornos_chars:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        # Heurística: El carácter debe ocupar entre el 40% y el 90% del alto de la placa
        # y debe ser más alto que ancho (aspect_ratio < 1)
        if (alto_placa * 0.4) < h < (alto_placa * 0.9) and 0.2 < aspect_ratio < 0.9:
            caracteres_validos.append((x, y, w, h))

    # Ordenar de izquierda a derecha
    caracteres_validos = sorted(caracteres_validos, key=lambda b: b[0])
    
    # Conservar solo los 6 mejores si hay ruido extra
    if len(caracteres_validos) > 6:
        caracteres_validos = caracteres_validos[:6]

    for i, (x, y, w, h) in enumerate(caracteres_validos):
        color = (0, 255, 0) if i < 3 else (255, 0, 0) # Diferenciar letras y números
        cv2.rectangle(patente_con_caracteres, (x, y), (x + w, y + h), color, 2)

    print(f"Se segmentaron {len(caracteres_validos)} caracteres.")

    mostrar_etapas(f"Parte B - {img_path}", 
                   [patente_recortada, binaria, binaria_limpia, patente_con_caracteres], 
                   ['Recorte', 'Adaptative Threshold', 'Filtro Limpieza', 'Caracteres Aislados'])

def main():
    for i in range(1, 13):
        encontrado = False
        for ext in ['.png', '.jpg']:
            nombre_archivo = f"img{i:02d}{ext}"
            if os.path.exists(nombre_archivo):
                detectar_y_segmentar(nombre_archivo)
                encontrado = True
                break
        if not encontrado:
            print(f"No se encontró img{i:02d}")

if __name__ == '__main__':
    main()