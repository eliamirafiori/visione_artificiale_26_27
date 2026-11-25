# Trasformata di Fourier 2D e Morfologia Matematica

Passare dai segnali 1D (come l'audio) all'elaborazione delle immagini 2D, imparando a stimare **traslazioni e rotazioni** con Fourier e a **pulire maschere binarie** con la morfologia matematica.

---

## 1. Dalla Trasformata 1D alla Trasformata di Fourier 2D

Nel caso 1D, la Trasformata di Fourier (`fft`) scompone un suono nella somma di tante onde sinusoidali semplici, ciascuna con una certa **frequenza**, **ampiezza (magnitudine)** e **fase**.

Un'immagine in scala di grigi è un segnale **2D** (ha righe e colonne). La Trasformata di Fourier 2D (`fft2`) fa esattamente la stessa cosa: scompone l'immagine in onde sinusoidali che oscillano in tutte le direzioni (orizzontali, verticali e diagonali).

* **Basse frequenze:** rappresentano le grandi aree uniformi e la forma generale dell'immagine.
* **Alte frequenze:** rappresentano i dettagli fini, i bordi netti e il rumore.
* **Visualizzazione (`fftshift` e logaritmo):** usiamo `fftshift` per spostare le basse frequenze al centro dell'immagine e `np.log1p(np.abs(...))` per comprimere i valori altissimi del centro e riuscire a vedere anche le frequenze più deboli.
* **Ricostruzione perfetta (`ifft2`):** applicando la trasformata inversa torniamo all'immagine originale senza perdere informazione ($\text{RMSE} \approx 0$).

---

## 2. Stimare la Traslazione: Correlazione di Fase

Supponiamo di avere un'immagine di riferimento $I_1$ e una sua copia $I_2$ spostata di $(dx, dy)$ pixel. Come facciamo a scoprire di quanto si è mossa?

### La proprietà magica di Fourier

Spostare un'immagine nello spazio **non cambia la magnitudine** del suo spettro, ma aggiunge soltanto uno sfasamento lineare nella **fase**:

$$
F_2(u, v) = F_1(u, v) \cdot e^{-j 2\pi \left(\frac{u \cdot dx}{W} + \frac{v \cdot dy}{H}\right)}
$$

### Come funziona l'algoritmo in 3 passi

1. Calcoliamo le trasformate $F_1 = \text{FFT2}(I_1)$ e $F_2 = \text{FFT2}(I_2)$.
2. Isoliamo **solo la differenza di fase**, normalizzando il prodotto incrociato per il suo modulo (così ignoriamo variazioni di luminosità):

$$
R(u, v) = \frac{F_2(u, v) \cdot F_1^*(u, v)}{\left|F_2(u, v) \cdot F_1^*(u, v)\right| + \varepsilon}
$$

3. Calcoliamo l'antitrasformata $r(x, y) = \text{IFFT2}(R)$. Il risultato è una mappa tutta nera con **un unico picco bianco molto netto** proprio alle coordinate dello spostamento $(dx, dy)$.

> **Attenzione agli spostamenti negativi:** poiché la FFT tratta l'immagine come se fosse periodica (cioè "arrotolata" sui bordi), uno spostamento di $-22$ pixel su un'immagine alta $256$ appare all'indice $256 - 22 = 234$. Se la coordinata del picco supera metà immagine ($H/2$ o $W/2$), basta sottrarre $H$ o $W$ per riottenere il numero negativo.

---

## 3. Stimare Rotazione + Traslazione (Il Trucco Polare)

Se l'immagine $I_2$ è sia **ruotata** sia **traslata**, la correlazione di fase diretta non funziona più. Possiamo però separare i due problemi sfruttando due proprietà dello spettro di magnitudine $|F(u, v)|$:

1. $|F(u, v)|$ è **invariante alla traslazione** (non gli importa se l'oggetto si è spostato).
2. $|F(u, v)|$ **ruota esattamente dello stesso angolo** dell'immagine originale.

### La pipeline in 4 passi

1. **Finestra di Hann (`np.hanning`):** moltiplichiamo le immagini per una maschera che sfuma dolcemente i bordi verso il nero. Questo evita che i tagli netti dei bordi dell'immagine creino una "croce" artificiale nello spettro di Fourier.
2. **Passaggio a Coordinate Polari (`cv2.warpPolar`):** "srotoliamo" lo spettro $|F|$ dal centro verso l'esterno in un rettangolo dove le colonne sono il raggio e le **righe sono l'angolo** (da $0^\circ$ a $360^\circ$).
   * **Il vantaggio:** una **rotazione** nell'immagine originale diventa una semplice **traslazione verticale** nell'immagine polare!
   * Usando $360$ righe, ogni riga di spostamento equivale esattamente a $1^\circ$. Applichiamo quindi la stessa `correlazione_di_fase` del punto 2 per leggere subito l'angolo!
3. **Risolvere l'ambiguità di $180^\circ$:** lo spettro di magnitudine di un'immagine reale è sempre simmetrico rispetto al centro (ruotarlo di $30^\circ$ o di $210^\circ$ produce lo stesso spettro). Per capire qual è il verso giusto, proviamo entrambi i candidati ($\theta$ e $\theta + 180^\circ$): quello corretto produrrà un picco di correlazione molto più alto.
4. **Calcolo finale di $(dx, dy)$:** una volta compensata la rotazione, resta solo una traslazione pura, che troviamo con un'ultima chiamata alla `correlazione_di_fase`.

---

## 4. Morfologia Matematica di Base

La morfologia matematica lavora tipicamente su **maschere binarie**, dove i pixel valgono solo $0$ (sfondo nero) oppure $255$ (oggetto bianco).

Tutte le operazioni si basano su un piccolo pennello chiamato **Elemento Strutturante** (o *kernel*, ad esempio un cerchietto $5 \times 5$) che viene fatto scorrere su ogni pixel dell'immagine.

### Le operazioni fondamentali

| Operazione | Funzione OpenCV | Come funziona (intuizione) | Effetto visivo |
| :--- | :--- | :--- | :--- |
| **Erosione** | `cv2.erode` | Un pixel resta bianco **solo se tutti** i pixel sotto il kernel sono bianchi (filtro di minimo). | Rimpicciolisce l'oggetto ed elimina i puntini bianchi isolati. |
| **Dilatazione** | `cv2.dilate` | Un pixel diventa bianco se **almeno un** pixel sotto il kernel è bianco (filtro di massimo). | Ingrandisce l'oggetto e chiude piccoli buchi o fessure. |
| **Apertura** | `cv2.MORPH_OPEN` | Prima **Erosione**, poi **Dilatazione**. | Elimina il rumore esterno (puntini bianchi sullo sfondo) senza rimpicciolire l'oggetto. |
| **Chiusura** | `cv2.MORPH_CLOSE` | Prima **Dilatazione**, poi **Erosione**. | Riempie i piccoli buchi neri interni all'oggetto senza ingrandirlo. |
| **Gradiente** | `cv2.MORPH_GRADIENT` | Differenza: $\text{Dilatazione} - \text{Erosione}$. | Estrae esattamente il **contorno** (bordo) dell'oggetto. |

### La guida pratica per pulire una maschera

Quando segmentiamo un oggetto da una foto reale, la maschera ha spesso sia puntini bianchi sparsi sullo sfondo (rumore *"sale"*) sia piccoli fori neri dentro l'oggetto (rumore *"pepe"*).
La sequenza standard per ripulirla preservando la forma originale è:

$$
\text{Maschera Rumorosa} \xrightarrow{\text{Apertura}} \text{Senza puntini esterni} \xrightarrow{\text{Chiusura}} \text{Maschera Pulita}
$$
