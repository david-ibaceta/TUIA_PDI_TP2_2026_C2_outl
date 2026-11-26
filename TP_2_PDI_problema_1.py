import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. CARGA Y NORMALIZACIÓN DE LA IMAGEN
# ---------------------------------------------------------
img_cruda = cv2.imread('C:/TUIA_PDI_2026_C2/TP_2/git/TUIA_PDI_TP2_2026_C2_outl/monedas.jpg')
if img_cruda is None:
    raise FileNotFoundError("No se encontró la imagen 'monedas.jpg'")

# Redimensionar a un ancho fijo de 1200px para normalizar métricas de área
valor_w = 1200
h_orig, w_orig = img_cruda.shape[:2]
escala = valor_w / float(w_orig)
target_h = int(h_orig * escala)
img_original = cv2.resize(img_cruda, (valor_w, target_h))

img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis('off')
plt.pause(3)
plt.close()

# ---------------------------------------------------------
# 2. FLUJO DE PROCESAMIENTO MORFOLÓGICO Y FILTRADO
# ---------------------------------------------------------
# PASO 1: Pasar a escala de grises (Blanco y Negro)
img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# PASO 2: Filtro pasa-bajos (GaussianBlur) para atenuar ruido
img_blur = cv2.GaussianBlur(img_gris, (7, 7), 0)

# PASO 3: Detección de bordes finos con Canny
canny_edges = cv2.Canny(img_blur, threshold1=15, threshold2=125)

plt.imshow(canny_edges, cmap='gray')
plt.title("Bordes Canny")
plt.axis('off')
plt.pause(2)
plt.close()


# PASO 4: Dilatación para cerrar contornos incompletos
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
iteraciones = 3
dilated_edges = cv2.dilate(canny_edges, kernel, iterations=iteraciones)

plt.imshow(dilated_edges, cmap='gray')
plt.title("Bordes dilatados")
plt.axis('off')
plt.pause(2)
plt.close()


# PASO 5: Rellenado de huecos (FloodFill) con marco de seguridad
# 1. Agregamos un borde negro (value=0) de 1 píxel en los 4 lados para evitar 
#    que falle cv2.flodfill()
padded = cv2.copyMakeBorder(dilated_edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# 2. Calculamos las nuevas dimensiones (ahora tienen +2 píxeles respecto a la original)
h, w = padded.shape
floodfilled = padded.copy()
# Tipo de máscara obligatoria de cv2 (1 px más que la imagen)
mask = np.zeros((h + 2, w + 2), np.uint8)

# 3. Inundamos de blanco (255) desde el (0,0) sin entrar en figuras cerradas
# Conclusión: quedan sólo los agujeros negros internos de cada objeto
cv2.floodFill(floodfilled, mask, (0, 0), 255)

# 4. Recortamos el marco de 1 píxel para volver al tamaño original
floodfilled = floodfilled[1:-1, 1:-1]

# Invierte la imagen y nos quedamos sólo con los huecos en blanco
filled_holes = cv2.bitwise_not(floodfilled)

# Sumamos ambas imagenes y obtenemos nuestros objetos
filled_objects = cv2.bitwise_or(dilated_edges, filled_holes)

plt.imshow(filled_objects, cmap='gray')
plt.title("Objetos rellenos")
plt.axis('off')
plt.pause(2)
plt.close()


# PASO 6: Erosión para restaurar tamaño original
# Mismo kernel e iteraciones que el utilizado en dilatación
eroded = cv2.erode(filled_objects, kernel, iterations=iteraciones)

plt.imshow(eroded, cmap='gray')
plt.title("Bordes erosionados")
plt.axis('off')
plt.pause(2)
plt.close()


# PASO 7: Apertura Morfológica para suavizado y apertura de bordes
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
binary_final = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_open, iterations=2)

plt.imshow(binary_final, cmap='gray')
plt.title("Objetos binarios finales")
plt.axis('off')
plt.pause(2)
plt.close()


# ---------------------------------------------------------
# 3. SEGMENTACIÓN Y DISCRIMINACIÓN POR FACTOR DE FORMA
# ---------------------------------------------------------
contours, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

monedas_cnts = []
dados_cnts = []
rois_dados_inv = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 400:  # Filtrar ruido menor
        continue

    perimetro = cv2.arcLength(cnt, True)
    factor_forma = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    x, y, bw, bh = cv2.boundingRect(cnt)
    solidez = area / float(bw * bh)
    aspect_ratio = float(bw) / bh

    # Criterio geométrico: Dado vs Moneda
    es_dado = (solidez > 0.81 or factor_forma < 0.84) and (0.70 < aspect_ratio < 1.35) and (area > 3000)

    if es_dado:
        # --- UMBRALADO INVERSO DE LA ROI DEL DADO (Fondo blanco -> 0, Pips negros -> 255) ---
        roi_gray = img_gris[y:y+bh, x:x+bw]
        _, roi_inv = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Margen de seguridad del 10% para descartar sombras de bordes
        mx, my = int(bw * 0.10), int(bh * 0.10)
        roi_inner = roi_inv[my:bh-my, mx:bw-mx]

        # Buscamos los puntitos de las caras de los dados
        pip_cnts, _ = cv2.findContours(roi_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contamos la cantidad de puntitos
        valor_dado = 0
        area_dado_inner = (bw - 2*mx) * (bh - 2*my)
        min_p_area = area_dado_inner * 0.015
        max_p_area = area_dado_inner * 0.08
        
        for p_cnt in pip_cnts:
            p_area = cv2.contourArea(p_cnt)
            p_perim = cv2.arcLength(p_cnt, True)
            # Factor de forma ideal p_circ=1
            p_circ = (4 * np.pi * p_area) / (p_perim ** 2) if p_perim > 0 else 0
            
            # Identificar contornos circulares compatibles con los puntos del dado
            if min_p_area < p_area < max_p_area and p_circ > 0.8:
                valor_dado += 1
                
        dados_cnts.append({
            'contour': cnt,
            'bbox': (x, y, bw, bh),
            'area': area,
            'valor_dado': valor_dado
        })
        rois_dados_inv.append((roi_inv, valor_dado))
    else:
        monedas_cnts.append({
            'contour': cnt,
            'area': area,
            'factor_forma': factor_forma
        })

# ---------------------------------------------------------
# 4. CLASIFICACIÓN DE MONEDAS MEDIANTE K-MEANS
# ---------------------------------------------------------
if len(monedas_cnts) > 0:
    areas_monedas = np.array([m['area'] for m in monedas_cnts]).reshape(-1, 1)
    n_clusters = min(3, len(monedas_cnts))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(areas_monedas)
    
    # Ordenar los grupos por tamaño creciente de área
    centroides = kmeans.cluster_centers_.flatten()
    indices_ordenados = np.argsort(centroides)
    
    mapa_categorias = {}
    nombres_categorias = ["Moneda Pequeña", "Moneda Mediana", "Moneda Grande"]
    valores_estimados = ["10 Centavos", "$1 Peso", "50 Centavos"]
    
    for idx_nuevo, idx_orig in enumerate(indices_ordenados):
        mapa_categorias[idx_orig] = {
            'subtipo': nombres_categorias[idx_nuevo],
            'valor': valores_estimados[idx_nuevo]
        }
    
    for i, m in enumerate(monedas_cnts):
        cluster_id = labels[i]
        m['subtipo'] = mapa_categorias[cluster_id]['subtipo']
        m['valor_estimado'] = mapa_categorias[cluster_id]['valor']

# ---------------------------------------------------------
# 5. GENERACIÓN DE RESULTADOS PINTADOS Y REPORTE
# ---------------------------------------------------------
img_classified = img_original.copy()
reporte_lista = []

# Pintar Dados en AZUL
for j, d in enumerate(dados_cnts):
    cnt = d['contour']
    x, y, bw, bh = d['bbox']
    val = d['valor_dado']
    
    cv2.drawContours(img_classified, [cnt], -1, (255, 0, 0), -1)
    cv2.rectangle(img_classified, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
    cv2.putText(img_classified, f"Dado: {val}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    reporte_lista.append({
        'Objeto': f'Dado {j+1}',
        'Tipo': 'Dado',
        'Subtipo/Categoría': f'Dado (Cara {val})',
        'Área (px)': round(d['area'], 1),
        'Valor Estimado': f"Cara {val}"
    })

# Pintar Monedas en VERDE
for i, m in enumerate(monedas_cnts):
    cnt = m['contour']
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    
    cv2.drawContours(img_classified, [cnt], -1, (0, 200, 0), -1)
    cv2.circle(img_classified, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
    cv2.putText(img_classified, f"M{i+1}", (int(cx)-10, int(cy)+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    reporte_lista.append({
        'Objeto': f'Moneda {i+1}',
        'Tipo': 'Moneda',
        'Subtipo/Categoría': m['subtipo'],
        'Área (px)': round(m['area'], 1),
        'Valor Estimado': m['valor_estimado']
    })

# Impresión de DataFrame en consola
df = pd.DataFrame(reporte_lista)
print("==========================================")
print("     RESUMEN GENERAL DE LO ENCONTRADO     ")
print("==========================================")
print(f" Monedas detectadas: {len(monedas_cnts)}")
print(f" Dados detectados:   {len(dados_cnts)}")
print("------------------------------------------")
if len(monedas_cnts) > 0:
    print("\nClasificación por K-Means (Monedas):")
    print(df[df['Tipo'] == 'Moneda']['Subtipo/Categoría'].value_counts())
print("\nDetalle completo de objetos:")
print(df[['Objeto', 'Tipo', 'Subtipo/Categoría', 'Área (px)', 'Valor Estimado']])

# ---------------------------------------------------------
# 6. MOSTRAR ETAPAS INTERMEDIAS Y RESULTADOS FINALES
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes[0, 0].imshow(img_gris, cmap='gray')
axes[0, 0].set_title("1. Grises (Blanco y Negro)")

axes[0, 1].imshow(img_blur, cmap='gray')
axes[0, 1].set_title("2. Filtro Pasa-Bajos")

axes[0, 2].imshow(canny_edges, cmap='gray')
axes[0, 2].set_title("3. Bordes Canny")

axes[0, 3].imshow(dilated_edges, cmap='gray')
axes[0, 3].set_title("4. Dilatación")

axes[1, 0].imshow(filled_objects, cmap='gray')
axes[1, 0].set_title("5. Rellenado de Huecos")

axes[1, 1].imshow(eroded, cmap='gray')
axes[1, 1].set_title("6. Erosión")

axes[1, 2].imshow(binary_final, cmap='gray')
axes[1, 2].set_title("7. Apertura Morfológica")

axes[1, 3].imshow(cv2.cvtColor(img_classified, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title(f"8. Clasificación Pintada ({len(monedas_cnts)} M, {len(dados_cnts)} D)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# Subplot con el umbralado inverso exclusivo de las ROI de los dados
fig_dados = plt.figure(figsize=(10, 4))
for idx, (roi_inv, val) in enumerate(rois_dados_inv):
    ax = fig_dados.add_subplot(1, len(rois_dados_inv), idx + 1)
    ax.imshow(roi_inv, cmap='gray')
    ax.set_title(f"ROI Dado {idx+1} Umbral Inverso (Cara superior valor: {val})")
    ax.axis('off')

plt.tight_layout()
plt.show()