# Questo è come scrivo un commento in Python

def la_mia_funzione(argomento_1):
    """
    Queste invece è una funzione!
    """

    variabile_1 = 2 # Qui ho definito una variabile

    # Ora vediamo l'if-else statement
    
    if argomento_1 + variabile_1 >= 18:
        return "Wow, hai passato l\'esame!"
    else:
        return "Riprova. Non sarai più fortunato."

variabile = la_mia_funzione(15)
print(variabile)

# In Python le variabili non sono tipicizzate
# Una variabile può assumere diverse tipologie di dato
variabile = 3
variabile = "Hei ora sono una stringa"
variabile = [] # Ora è una lista

