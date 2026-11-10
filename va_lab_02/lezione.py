import cv2
import numpy as np

# Creazione di un'immagine sintetica di test (se non presente su disco)
# Dimensioni: altezza 300px, larghezza 400px, 3 canali (BGR)
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:, :] = [180, 105, 255]  # Sfondo rosa in formato BGR
cv2.circle(img, (200, 150), 60, (0, 255, 0), -1)  # Cerchio verde
cv2.rectangle(img, (50, 50), (120, 120), (255, 0, 0), -1)  # Quadrato blu
cv2.imwrite("immagine_input.jpg", img)

# 1. Caricamento e Proprietà dell'Immagine
# OpenCV legge le immagini come array NumPy in formato BGR (non RGB)
immagine = cv2.imread("immagine_input.jpg")
if immagine is None:
    raise FileNotFoundError("Impossibile caricare l'immagine specificata.")

# Visualizzazione a Schermo
cv2.imshow("Originale (BGR)", immagine)
cv2.waitKey(0)

altezza, larghezza, canali = immagine.shape
print(
    f"Dimensioni: {larghezza}x{altezza}, Canali: {canali}, Tipo di dato: {immagine.dtype}"
)

# 2. Conversioni di Spazio Colore
# Da BGR a Scala di Grigi (Grayscale)
grigi = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
cv2.imshow("Scala di Grigi", grigi)
cv2.waitKey(0)

# Da BGR a HSV (Hue, Saturation, Value) - fondamentale per la segmentazione per colore
hsv = cv2.cvtColor(immagine, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)

# Da BGR a RGB (necessario se si visualizza con Matplotlib)
rgb = cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)
cv2.imshow("Originale (RGB)", rgb)
cv2.waitKey(0)

# 3. Trasformazioni Geometriche di Base
# Ridimensionamento: attenzione, l'ordine è (larghezza, altezza)
ridimensionata = cv2.resize(immagine, (200, 150), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Ridimensionata", ridimensionata)
cv2.waitKey(0)

# Ritaglio (Cropping): usa lo slicing NumPy [y_inizio:y_fine, x_inizio:x_fine]
ritaglio = immagine[50:250, 100:300]
cv2.imshow("Ritagliata", ritaglio)
cv2.waitKey(0)

# Rotazione attorno al centro dell'immagine
centro = (larghezza // 2, altezza // 2)
matrice_rotazione = cv2.getRotationMatrix2D(centro, angle=45, scale=1.0)
ruotata = cv2.warpAffine(immagine, matrice_rotazione, (larghezza, altezza))
cv2.imshow("Ruotata", ruotata)
cv2.waitKey(0)

# Riflessione speculare (Flip: 1 = orizzontale, 0 = verticale, -1 = entrambi)
specchiata = cv2.flip(immagine, 1)
cv2.imshow("Specchiata", specchiata)
cv2.waitKey(0)

# 4. Modifiche di Colore e Aritmetica dei Pixel
# Regolazione luminosità e contrasto: nuova_img = alpha * img + beta
# alpha > 1.0 aumenta il contrasto; beta > 0 aumenta la luminosità
contrasto_luminosita = cv2.convertScaleAbs(immagine, alpha=1.2, beta=40)
cv2.imshow("Contrasto Luminosità", contrasto_luminosita)
cv2.waitKey(0)

# Separazione e unione dei canali
b, g, r = cv2.split(immagine)
# Aumento del canale Rosso con saturazione a 255 (evita l'overflow)
r_potenziato = cv2.add(r, 50)
canali_uniti = cv2.merge([b, g, r_potenziato])
cv2.imshow("Verde \"Potenziato\"", canali_uniti)
cv2.waitKey(0)

# 5. Filtraggio e Soglia (Thresholding)
# Filtro Gaussiano per ridurre il rumore e sfocare i dettagli
sfocata = cv2.GaussianBlur(immagine, (7, 7), sigmaX=1.5)
cv2.imshow("Sfocata", sfocata)
cv2.waitKey(0)

# Binarizzazione a soglia (Thresholding) su scala di grigi
# Pixel > 127 diventano 255 (bianco), gli altri diventano 0 (nero)
_, soglia = cv2.threshold(grigi, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binaria", soglia)
cv2.waitKey(0)

# 6. Salvataggio su Disco
cv2.imwrite("output_grigi.jpg", grigi)
cv2.imwrite("output_ruotata.jpg", ruotata)
cv2.imwrite("output_soglia.png", soglia)
print("Immagini elaborate salvate correttamente su disco.")

# 7. Visualizzazione a Schermo
cv2.imshow("Originale (BGR)", immagine)
cv2.imshow("Scala di Grigi", grigi)
cv2.imshow("Soglia Binaria", soglia)

print("Premi un tasto qualsiasi su una delle finestre per chiudere...")
cv2.waitKey(0)
cv2.destroyAllWindows()
