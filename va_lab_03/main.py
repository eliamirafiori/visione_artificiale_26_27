import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift, ifft2


# ------------------------------------------------------------------
# 1. Creazione di una semplice immagine 2D e spettro di Fourier 2D
# ------------------------------------------------------------------
# Usiamo una forma asimmetrica (simile a una "F" con un cerchio):
# in questo modo una rotazione di 180° è chiaramente distinguibile.
def crea_immagine_test(dim=256):
    img = np.zeros((dim, dim), dtype=np.float32)
    cv2.rectangle(img, (85, 65), (110, 195), 1.0, -1)  # Asta verticale
    cv2.rectangle(img, (85, 65), (175, 90), 1.0, -1)  # Barra superiore
    cv2.rectangle(img, (85, 115), (150, 138), 0.7, -1)  # Barra centrale
    cv2.circle(img, (165, 165), 18, 0.9, -1)  # Cerchio in basso a destra
    return cv2.GaussianBlur(img, (5, 5), 1.0)


img_orig = crea_immagine_test(256)

# Calcolo della Trasformata di Fourier 2D (equivalente 2D di fft() in 1D)
F = fft2(img_orig)
# fftshift porta le basse frequenze (componente DC) al centro per la visualizzazione
spettro_mag = np.log1p(np.abs(fftshift(F)))
# Ricostruzione inversa (equivalente 2D di ifft() in 1D)
img_ricostruita = ifft2(F).real

rmse = np.sqrt(np.mean((img_orig - img_ricostruita) ** 2))
print(f"Errore di ricostruzione (RMSE): {rmse:.3e}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_orig, cmap="gray")
axes[0].set_title("Immagine Originale")
axes[1].imshow(spettro_mag, cmap="magma")
axes[1].set_title("Spettro di Magnitudine (log)")
axes[2].imshow(img_ricostruita, cmap="gray")
axes[2].set_title("Immagine Ricostruita (IFFT2)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# 2. Stima della sola TRASLAZIONE con Correlazione di Fase
# ------------------------------------------------------------------
# Proprietà di Fourier: traslare un'immagine nello spazio NON cambia la
# magnitudine dello spettro, ma aggiunge solo uno sfasamento nella FASE.
def correlazione_di_fase(img_rif, img_spostata):
    """
    Stima di quanti pixel (dx, dy) `img_spostata` è traslata rispetto a `img_rif`.
    """
    F_rif = fft2(img_rif)
    F_spost = fft2(img_spostata)

    # Manteniamo solo la differenza di fase normalizzando per il modulo
    prodotto = F_spost * np.conj(F_rif)
    spettro_fase = prodotto / (np.abs(prodotto) + 1e-8)

    # Tornando nel dominio spaziale otteniamo un picco netto nella posizione (dy, dx)
    mappa_corr = np.real(ifft2(spettro_fase))
    picco = float(np.max(mappa_corr))
    y_max, x_max = np.unravel_index(np.argmax(mappa_corr), mappa_corr.shape)

    # Gestione degli spostamenti negativi (periodicità della FFT)
    h, w = img_rif.shape
    dy = y_max if y_max <= h // 2 else y_max - h
    dx = x_max if x_max <= w // 2 else x_max - w
    return dx, dy, picco, mappa_corr


dx_vero, dy_vero = 35, -22
M_trasl = np.float32([[1, 0, dx_vero], [0, 1, dy_vero]])
img_traslata = cv2.warpAffine(img_orig, M_trasl, (256, 256))

dx_st, dy_st, picco, mappa = correlazione_di_fase(img_orig, img_traslata)
print(
    f"\n[Solo Traslazione] Reale: dx={dx_vero}, dy={dy_vero} | "
    f"Stimata: dx={dx_st}, dy={dy_st} (picco={picco:.3f})"
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_orig, cmap="gray")
axes[0].set_title("Originale")
axes[1].imshow(img_traslata, cmap="gray")
axes[1].set_title(f"Traslata (dx={dx_vero}, dy={dy_vero})")
axes[2].imshow(fftshift(mappa), cmap="inferno")
axes[2].set_title(f"Picco Correlazione di Fase\nStimato: ({dx_st}, {dy_st})")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# 3. Stima di ROTAZIONE + TRASLAZIONE con Fourier e Coordinate Polari
# ------------------------------------------------------------------
# Idea chiave in 3 passi:
#   a) Lo spettro di magnitudine |F| ignora la traslazione e RUOTA come l'immagine.
#   b) Convertendo |F| in coordinate polari (raggio, angolo), una rotazione
#      diventa una semplice traslazione verticale lungo l'asse degli angoli.
#   c) Trovato l'angolo, riallineiamo l'orientamento e calcoliamo (dx, dy)
#      con la correlazione di fase del punto 2.
def stima_rotazione_e_traslazione(img_rif, img_trasformata):
    h, w = img_rif.shape
    centro = (w / 2.0, h / 2.0)

    # Finestra di Hann per attenuare i bordi (evita linee artificiali nella FFT)
    hann = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    mag1 = np.log1p(np.abs(fftshift(fft2(img_rif * hann))))
    mag2 = np.log1p(np.abs(fftshift(fft2(img_trasformata * hann))))

    # Coordinate polari: 360 righe -> 1 riga corrisponde esattamente a 1 grado
    raggio_max = min(h, w) / 2.0
    polare1 = cv2.warpPolar(
        mag1, (int(raggio_max), 360), centro, raggio_max, cv2.WARP_POLAR_LINEAR
    )
    polare2 = cv2.warpPolar(
        mag2, (int(raggio_max), 360), centro, raggio_max, cv2.WARP_POLAR_LINEAR
    )

    # Lo spostamento verticale tra i due spettri polari è la rotazione in gradi
    _, d_angolo, _, _ = correlazione_di_fase(polare1, polare2)
    # Segno opposto: warpPolar cresce in senso orario, getRotationMatrix2D in senso antiorario
    angolo_candidato = -d_angolo

    # Lo spettro di un'immagine reale è simmetrico di 180°:
    # proviamo sia `angolo` sia `angolo + 180°` e teniamo il picco più alto
    migliore = None
    for angolo in (angolo_candidato, angolo_candidato + 180):
        # Ruotiamo l'immagine di riferimento: ciò che resta rispetto a img_trasformata
        # è esattamente la traslazione pura (dx, dy)
        M_rot = cv2.getRotationMatrix2D(centro, angolo, 1.0)
        img_rif_ruotata = cv2.warpAffine(img_rif, M_rot, (w, h))

        dx, dy, picco_val, _ = correlazione_di_fase(img_rif_ruotata, img_trasformata)

        if migliore is None or picco_val > migliore["picco"]:
            # Riallineiamo img_trasformata sull'originale (prima -traslazione, poi -rotazione)
            M_tr_inv = np.float32([[1, 0, -dx], [0, 1, -dy]])
            senza_trasl = cv2.warpAffine(img_trasformata, M_tr_inv, (w, h))
            M_rot_inv = cv2.getRotationMatrix2D(centro, -angolo, 1.0)
            img_allineata = cv2.warpAffine(senza_trasl, M_rot_inv, (w, h))

            ang_norm = ((angolo + 180) % 360) - 180  # Riporta l'angolo in (-180, 180]
            migliore = {
                "angolo": ang_norm,
                "dx": dx,
                "dy": dy,
                "picco": picco_val,
                "img_allineata": img_allineata,
                "mag1": mag1,
                "mag2": mag2,
                "polare1": polare1,
                "polare2": polare2,
            }
    return migliore


# Applichiamo una rotazione + traslazione nota
angolo_vero, dx_vero, dy_vero = 30, 25, -18
h, w = img_orig.shape
centro = (w / 2.0, h / 2.0)

img_rot = cv2.warpAffine(
    img_orig, cv2.getRotationMatrix2D(centro, angolo_vero, 1.0), (w, h)
)
img_rot_trasl = cv2.warpAffine(
    img_rot, np.float32([[1, 0, dx_vero], [0, 1, dy_vero]]), (w, h)
)

ris = stima_rotazione_e_traslazione(img_orig, img_rot_trasl)
rmse_all = np.sqrt(np.mean((img_orig - ris["img_allineata"]) ** 2))

print(f"\n[Rotazione + Traslazione]")
print(f"  Reale   -> angolo={angolo_vero}°, dx={dx_vero}, dy={dy_vero}")
print(
    f"  Stimato -> angolo={ris['angolo']}°, dx={ris['dx']}, dy={ris['dy']} "
    f"(picco={ris['picco']:.3f}, RMSE={rmse_all:.4f})"
)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes[0, 0].imshow(img_orig, cmap="gray")
axes[0, 0].set_title("1. Originale")
axes[0, 1].imshow(img_rot_trasl, cmap="gray")
axes[0, 1].set_title(f"2. Ruotata ({angolo_vero}°) + Traslata ({dx_vero}, {dy_vero})")
axes[0, 2].imshow(ris["img_allineata"], cmap="gray")
axes[0, 2].set_title(
    f"3. Riallineata\n(ang={ris['angolo']}°, dx={ris['dx']}, dy={ris['dy']})"
)

axes[1, 0].imshow(ris["mag1"], cmap="magma")
axes[1, 0].set_title("Spettro |F| Originale")
axes[1, 1].imshow(ris["mag2"], cmap="magma")
axes[1, 1].set_title("Spettro |F| Ruotato (invariante a dx, dy)")
axes[1, 2].imshow(ris["polare2"], cmap="magma", aspect="auto")
axes[1, 2].set_title("Spettro in Coordinate Polari (y = angolo)")
axes[1, 2].set_xlabel("Raggio (frequenza)")
axes[1, 2].set_ylabel("Angolo (0° - 360°)")

for r in range(2):
    for c in range(3):
        if not (r == 1 and c == 2):
            axes[r, c].axis("off")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# 4. Morfologia Matematica di Base (pulizia maschera ed estrazione bordi)
# ------------------------------------------------------------------
# Binarizziamo la nostra figura (0 = sfondo nero, 255 = oggetto bianco)
maschera_pulita = (img_orig > 0.3).astype(np.uint8) * 255

# Aggiungiamo rumore tipico di una segmentazione reale:
# - puntini bianchi sullo sfondo (rumore "sale")
# - piccoli buchi neri dentro l'oggetto (rumore "pepe")
rng = np.random.default_rng(42)
maschera_rumorosa = maschera_pulita.copy()

for _ in range(80):
    x, y = rng.integers(5, 250, size=2)
    cv2.circle(maschera_rumorosa, (int(x), int(y)), 1, 255, -1)

ys_obj, xs_obj = np.where(maschera_pulita == 255)
for idx in rng.choice(len(xs_obj), size=15, replace=False):
    cv2.circle(maschera_rumorosa, (int(xs_obj[idx]), int(ys_obj[idx])), 2, 0, -1)

# Elemento strutturante (la "pennellata" che scorre sull'immagine)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Operazioni elementari
erosione = cv2.erode(maschera_rumorosa, kernel, iterations=1)
dilatazione = cv2.dilate(maschera_rumorosa, kernel, iterations=1)

# Operazioni composte
apertura = cv2.morphologyEx(
    maschera_rumorosa, cv2.MORPH_OPEN, kernel
)  # Erosione -> Dilatazione
chiusura = cv2.morphologyEx(
    maschera_rumorosa, cv2.MORPH_CLOSE, kernel
)  # Dilatazione -> Erosione

# Pipeline pratica: prima Apertura (elimina puntini esterni), poi Chiusura (riempie buchi interni)
maschera_riparata = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)

# Gradiente morfologico = Dilatazione - Erosione -> isola il contorno dell'oggetto
kernel_bordi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
contorni = cv2.morphologyEx(maschera_riparata, cv2.MORPH_GRADIENT, kernel_bordi)

err_prima = np.mean(maschera_rumorosa != maschera_pulita) * 100
err_dopo = np.mean(maschera_riparata != maschera_pulita) * 100
print(
    f"\n[Morfologia] Pixel errati prima: {err_prima:.2f}% -> dopo Apertura+Chiusura: {err_dopo:.2f}%"
)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
immagini_morf = [
    (maschera_rumorosa, "Maschera Rumorosa (Input)"),
    (erosione, "Erosione (rimpicciolisce)"),
    (dilatazione, "Dilatazione (ingrandisce)"),
    (apertura, "Apertura (toglie puntini esterni)"),
    (maschera_riparata, "Apertura + Chiusura (pulita)"),
    (contorni, "Gradiente Morfologico (contorni)"),
]
for ax, (im, titolo) in zip(axes.flat, immagini_morf):
    ax.imshow(im, cmap="gray")
    ax.set_title(titolo)
    ax.axis("off")
plt.tight_layout()
plt.show()
