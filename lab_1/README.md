## Decision Tree Classifier

Această aplicație Python construiește un arbore decizional în baza unui set de date de instruire, și îl folosește pentru a lua decizii pe baza unor date noi.
Setul de Date

Setul de date este format din 16 înregistrări, fiecare dintre ele conținând valorile pentru următoarele atribute:

    Timp (însorit, înnorat, ploaie)
    Temperatură (cald, moderat, rece)
    Umiditate (mare, normală)
    Vânt (da, nu)

și valoarea pentru variabila țintă binară joc (da, nu).

## Cum să Rulați Aplicația


Această aplicație depinde de următoarele biblioteci Python:

    pandas
    numpy
    sklearn
    graphviz

Puteți instala aceste biblioteci cu pip:

    pip install pandas numpy sklearn graphviz

Această aplicație depinde, de asemenea, de Graphviz pentru a genera reprezentarea grafică a arborelui decizional.

Asigurați-vă că aveți Graphviz instalat și că executabilul dot este în calea PATH a sistemului.