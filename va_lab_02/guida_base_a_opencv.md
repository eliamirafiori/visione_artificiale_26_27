# Guida Introduttiva a OpenCV con Python

Una panoramica pratica e intuitiva per muovere i primi passi nell'elaborazione delle immagini digitali (Computer Vision).

## 1. Come vede un'immagine il computer?

Prima di iniziare con il codice, è importante capire una cosa fondamentale: **per il computer, un'immagine non è altro che una tabella di numeri**.

* Ogni immagine è composta da minuscoli quadratini chiamati **pixel**.
* Se l'immagine è in bianco e nero (scala di grigi), ogni pixel è un numero intero compreso tra $0$ (nero assoluto) e $255$ (bianco puro).
* Se l'immagine è a colori, ogni pixel ha **3 valori numerici**, ciascuno compreso tra $0$ e $255$, che indicano l'intensità di tre colori primari.

In Python, la libreria **OpenCV** memorizza le immagini come semplici array **NumPy**.

## 2. Lo Script

### Passo 0: Creare un'immagine di test

```python
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:, :] = [180, 105, 255]
cv2.circle(img, (200, 150), 60, (0, 255, 0), -1)
cv2.rectangle(img, (50, 50), (120, 120), (255, 0, 0), -1)
cv2.imwrite("immagine_input.jpg", img)
```

* **Perché lo facciamo?** Per evitare errori del tipo *"File not found"*. Creiamo un'immagine sintetica direttamente da codice.
* `np.zeros((300, 400, 3), dtype=np.uint8)` crea una tela vuota:
  * Altezza: $300$ pixel (righe)
  * Larghezza: $400$ pixel (colonne)
  * Canali: $3$ (colori)
  * `uint8` significa "numeri interi positivi a 8 bit", ovvero valori da $0$ a $255$.
* Usiamo funzioni di disegno come `cv2.circle` e `cv2.rectangle` per aggiungere forme geometriche colorate. Il parametro `-1` indica di riempire la forma.

### Passo 1: Caricare e ispezionare l'immagine

```python
immagine = cv2.imread("immagine_input.jpg")
altezza, larghezza, canali = immagine.shape
```

* `cv2.imread()` legge il file dal disco e lo converte in matrice.
* `immagine.shape` restituisce una tupla con 3 valori: `(altezza, larghezza, numero_canali)`.
* **Attenzione:** la prima coordinata è sempre l'**altezza** ($y$, righe), la seconda è la **larghezza** ($x$, colonne).

### Passo 2: Cambiare lo "Spazio Colore"

Il colore può essere espresso in modi diversi a seconda del nostro obiettivo:

1. **Da BGR a Scala di Grigi (`cv2.COLOR_BGR2GRAY`)**:
   * Riduce l'immagine da 3 canali a 1 solo canale.
   * **Perché si usa?** Riconoscere forme, contorni o volti non richiede quasi mai i colori: lavorare su 1 solo canale rende gli algoritmi $3$ volte più veloci.
2. **Da BGR a HSV (`cv2.COLOR_BGR2HSV`)**:
   * Rappresenta i colori secondo **Tonalità** (*Hue*), **Saturazione** (*Saturation*) e **Luminosità** (*Value*).
   * **Perché si usa?** È lo spazio colore ideale per isolare un colore specifico (es. trovare tutti gli oggetti rossi o verdi), perché la tonalità del colore è separata dalla quantità di luce presente nella scena.
3. **Da BGR a RGB (`cv2.COLOR_BGR2RGB`)**:
   * OpenCV per motivi storici memorizza i canali nell'ordine **Blu-Verde-Rosso (BGR)** anziché Rosso-Verde-Blu (RGB).
   * Se proviamo a mostrare un'immagine OpenCV con la libreria grafica `matplotlib`, i volti appariranno bluastri! Questa conversione serve a rimettere i canali nell'ordine convenzionale.

### Passo 3: Trasformazioni Geometriche di Base

#### 1. Ridimensionamento (`cv2.resize`)

```python
ridimensionata = cv2.resize(immagine, (200, 150))
```

* Rimpicciolisce o ingrandisce l'immagine.
* **Nota:** qui l'ordine dei parametri si inverte rispetto a NumPy! OpenCV accetta `(larghezza, altezza)`, cioè $(x, y)$.

#### 2. Ritaglio (Cropping)

```python
ritaglio = immagine[50:250, 100:300]
```

* Non serve alcuna funzione speciale di OpenCV: basta usare lo *slicing* di NumPy.
* Sintassi: `array[y_inizio:y_fine, x_inizio:x_fine]`.
* In questo esempio stiamo tenendo solo i pixel verticali tra riga $50$ e $250$, e quelli orizzontali tra colonna $100$ e $300$.

#### 3. Rotazione (`cv2.getRotationMatrix2D` + `cv2.warpAffine`)

* Si calcola prima la matrice matematica di rotazione specificando il punto perno (il centro) e l'angolo in gradi.
* `cv2.warpAffine` applica tale matrice per spostare fisicamente i pixel nella nuova posizione.

#### 4. Riflessione (`cv2.flip`)

* Crea un effetto specchio:
  * `1`: riflessione orizzontale (specchio classico).
  * `0`: riflessione verticale (sottosopra).
  * `-1`: entrambe contemporaneamente.

### Passo 4: Modifiche di Colore e Luminosità

#### Formula del contrasto e luminosità

Ogni pixel $P_{vecchio}$ viene ricalcolato secondo la funzione lineare:

$$P_{nuovo} = \alpha \cdot P_{vecchio} + \beta$$

* $\alpha > 1.0$ aumenta il contrasto (i pixel scuri diventano più scuri, quelli chiari più chiari).
* $\beta > 0$ aggiunge luce uniforme all'intera immagine.
* La funzione `cv2.convertScaleAbs()` esegue questa operazione assicurandosi che i valori restino compresi nell'intervallo $[0, 255]$.

#### Separazione e Aritmetica con Saturazione

```python
b, g, r = cv2.split(immagine)
r_potenziato = cv2.add(r, 50)
canali_uniti = cv2.merge([b, g, r_potenziato])
```

* `cv2.split()` separa i 3 fogli di colore (Blu, Verde, Rosso).
* **Perché usiamo `cv2.add()` anziché `r + 50`?**
  * La normale somma NumPy lavora in modulo $256$: se un pixel vale $220$ e aggiungiamo $50$, otteniamo $270 \rightarrow 14$ (diventa improvvisamente scuro!).
  * `cv2.add()` usa la saturazione: qualsiasi valore maggiore di $255$ viene bloccato a $255$.

### Passo 5: Filtri e Soglia (Thresholding)

#### Filtro Gaussiano (`cv2.GaussianBlur`)

* Sfoca l'immagine calcolando la media ponderata dei pixel vicini.
* Serve per **eliminare il "rumore" di fondo** (granulosità, artefatti della fotocamera) prima di applicare algoritmi di riconoscimento.

#### Binarizzazione / Soglia (`cv2.threshold`)

* Prende un'immagine in scala di grigi e imposta una regola binaria:
  $$\text{Pixel Finale} = \begin{cases} 255 & \text{se } \text{Pixel} > 127 \\ 0 & \text{altrimenti} \end{cases}$$
* Il risultato è un'immagine con soli pixel bianchi e neri netti, ideale per isolare la sagoma di un oggetto dallo sfondo.

### Passo 6: Salvataggio su Disco

```python
cv2.imwrite("output_grigi.jpg", grigi)
```

* Prende la matrice in memoria e la codifica nel formato desiderato in base all'estensione fornita (`.jpg`, `.png`, `.bmp`, ecc.).

### Passo 7: Visualizzazione a Schermo

```python
cv2.imshow("Finestra", immagine)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

* `cv2.imshow()` apre la finestra grafica.
* `cv2.waitKey(0)` è **fondamentale**: dice al sistema operativo di fermarsi e attendere che l'utente prema un tasto qualsiasi prima di proseguire. Senza questa istruzione, la finestra si chiuderebbe in una frazione di millisecondo.
* `cv2.destroyAllWindows()` chiude tutte le finestre create per liberare la memoria.
