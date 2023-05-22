# Machine Learning Analysis

Această aplicație Python efectuează o analiză a datelor folosind tehnici de învățare automată, inclusiv scalarea datelor, eliminarea valorilor aberante și antrenarea unui model de regresie liniară. Rezultatele analizei sunt salvate într-un fișier Markdown.

## Setul de Date

Setul de date este citit din fișierul CSV "data_cars.csv". Acesta conține informații despre mașini, inclusiv caracteristici numerice și categorice.

## Rezultatele Analizei

Analiza cuprinde următoarele etape și rezultate:

- **Statisticile de bază**: Se calculează statistici descriptive pentru setul de date, cum ar fi media, deviația standard și cuartile.
- **Corelații**: Se calculează matricea de corelații pentru variabilele numerice.
- **Covarianțe**: Se calculează matricea de covarianțe pentru variabilele numerice.
- **Valori nule**: Se identifică numărul de valori nule pentru fiecare variabilă din setul de date.
- **Preprocesare**: Se aplică preprocesarea datelor, inclusiv înlocuirea valorilor nule cu valori medii/mode, scalarea datelor și eliminarea valorilor aberante.
- **Model de regresie liniară**: Se antrenează un model de regresie liniară și se calculează coeficienții, interceptul și diverse metrici de evaluare (MAE, MSE, RMSE, R2).
- **Analiza de bias și varianță**: Se utilizează metoda bootstrap pentru a estima bias-ul și varianța modelului de regresie liniară.

Toate aceste rezultate sunt salvate într-un fișier Markdown denumit "Output-[data și oră curentă].md".

## Cum să Rulați Aplicația

1. Asigurați-vă că aveți instalate dependențele necesare (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, mlxtend).
1. Descărcați sau clonați acest repository în sistemul dvs.
1. Asigurați-vă că fișierul "data_cars.csv" este în același director cu scriptul Python.
1. Executați scriptul Python index.py.
1. Fișierul rezultat "Output-[data și oră curentă].md" va fi generat în același director.

## Dependințe

Această aplicație necesită următoarele biblioteci Python:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    scipy
    mlxtend

Puteți instala aceste biblioteci folosind comanda `make install` sau folosind pip:

`pip install pandas numpy matplotlib seaborn scikit-learn scipy mlxtend markdown tabulate`