from projet import *
from sklearn.metrics import accuracy_score
import os

classifieur = load_classifier("classifieur.pkl")
y_test = store_test_labels("Data/Unknown/")
to_predict = process_graycomatrix("Data/Unknown/")
to_predict = to_predict.reshape(len(to_predict), -1)
y_predits = classifieur.predict(to_predict)
score = accuracy_score(y_predits, y_test)
print(score)

# Partie permettant de retourner les images incorrectement classifiées par notre classifieur sur les données test mi-parcours
files = os.listdir("Data/Unknown")
for i in range(len(y_predits)):
    if y_predits[i] != y_test[i]:
        print(files[i])