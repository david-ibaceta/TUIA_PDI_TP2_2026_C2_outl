import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. CARGA Y NORMALIZACIÓN DE LA IMAGEN
# ---------------------------------------------------------
img_raw = cv2.imread('monedas.jpg')
if img_raw is None:
    raise FileNotFoundError("No se encontró la imagen 'monedas.jpg'")

# Redimensionar a un ancho fijo de 1200px para normalizar métricas de área
target_w = 1200
h_orig, w_orig = img_raw.shape[:2]
scale = target_w / float(w_orig)
target_h = int(h_orig * scale)
img_original = cv2.resize(img_raw, (target_w, target_h))

# ---------------------------------------------------------
# 2. FLUJO DE PROCESAMIENTO MORFOLÓGICO Y FILTRADO
# ---------------------------------------------------------
# PASO 1: Pasar a escala de grises (Blanco y Negro)
gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# PASO 2: Filtro pasa-bajos (GaussianBlur) para atenuar ruido
low_pass = cv2.GaussianBlur(gray, (7, 7), 0)

# PASO 3: Detección de bordes finos con Canny
canny_edges = cv2.Canny(low_pass, threshold1=30, threshold2=100)

# PASO 4: Dilatación para cerrar contornos incompletos
kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_edges = cv2.dilate(canny_edges, kernel_3x3, iterations=2)

# PASO 5: Rellenado de huecos (FloodFill)
h, w = dilated_edges.shape
floodfilled = dilated_edges.copy()
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(floodfilled, mask, (0, 0), 255)
filled_holes = cv2.bitwise_not(floodfilled)
filled_objects = cv2.bitwise_or(dilated_edges, filled_holes)

# PASO 6: Erosión para restaurar tamaño original
eroded = cv2.erode(filled_objects, kernel_3x3, iterations=2)

# PASO 7: Apertura Morfológica para suavizado de bordes
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary_final = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_open, iterations=1)

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
        roi_gray = gray[y:y+bh, x:x+bw]
        _, roi_inv = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Margen de seguridad del 10% para descartar sombras de bordes
        mx, my = int(bw * 0.10), int(bh * 0.10)
        roi_inner = roi_inv[my:bh-my, mx:bw-mx]
        
        pip_cnts, _ = cv2.findContours(roi_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valor_dado = 0
        area_dado_inner = (bw - 2*mx) * (bh - 2*my)
        min_p_area = area_dado_inner * 0.015
        max_p_area = area_dado_inner * 0.08
        
        for p_cnt in pip_cnts:
            p_area = cv2.contourArea(p_cnt)
            p_perim = cv2.arcLength(p_cnt, True)
            p_circ = (4 * np.pi * p_area) / (p_perim ** 2) if p_perim > 0 else 0
            
            # Identificar contornos circulares compatibles con los puntos del dado
            if min_p_area < p_area < max_p_area and p_circ > 0.55:
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
    valores_estimados = ["10 / 25 Centavos", "50 Centavos", "$1 Peso"]
    
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
print("       RESUMEN GENERAL DEL ALGORITMO      ")
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

axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title("1. Grises (Blanco y Negro)")

axes[0, 1].imshow(low_pass, cmap='gray')
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
    ax.set_title(f"ROI Dado {idx+1} Umbral Inverso (Cara: {val})")
    ax.axis('off')

plt.tight_layout()
plt.show()