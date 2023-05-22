import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import graphviz
import pydotplus

# Definim setul de date
data = {
    'Timp': ['însorit', 'însorit', 'însorit', 'însorit', 'însorit', 'însorit', 'însorit', 'însorit', 'înnorat', 'înnorat', 'înnorat', 'ploaie', 'ploaie', 'ploaie', 'ploaie', 'ploaie'],
    'Temperatura': ['cald', 'cald', 'cald', 'cald', 'moderat', 'moderat', 'rece', 'rece', 'rece', 'cald', 'moderat', 'cald', 'moderat', 'moderat', 'rece', 'rece'],
    'Umiditate': ['normală', 'normală', 'normală', 'normală', 'normală', 'normală', 'mare', 'mare', 'normală', 'normală', 'mare', 'normală', 'mare', 'mare', 'mare', 'mare'],
    'Vânt': ['da', 'nu', 'nu', 'nu', 'da', 'nu', 'da', 'nu', 'da', 'nu', 'nu', 'nu', 'nu', 'da', 'da', 'da'],
    'Joc': ['nu', 'da', 'da', 'da', 'nu', 'da', 'nu', 'da', 'da', 'da', 'nu', 'da', 'da', 'nu', 'nu', 'nu']
}
df = pd.DataFrame(data)

# Vom utiliza label encoding pentru a converti valorile string în valori numerice
le = preprocessing.LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Definim X și y
X = df_encoded[['Timp', 'Temperatura', 'Umiditate', 'Vânt']]
y = df_encoded['Joc']

# Construim arborele decizional
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# Exportăm arborele în format .dot
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=['Timp', 'Temperatura', 'Umiditate', 'Vânt'],
                                class_names=['nu', 'da'],
                                filled=True)

# Creăm un grafic cu arborele
graph = graphviz.Source(dot_data, format="png") 
graph.render(filename="tree")

# Predictia pentru noile date
new_data = {
    'Timp': ['însorit', 'ploaie', 'ploaie'],
    'Temperatura': ['cald', 'rece', 'rece'],
    'Umiditate': ['normală', 'normală', 'mare'],
    'Vânt': ['da', 'da', 'nu']
}
new_df = pd.DataFrame(new_data)
new_df_encoded = new_df.apply(le.fit_transform)

print("Deciziile recomandate sunt:")
print(clf.predict(new_df_encoded))
