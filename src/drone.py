from projet import *
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("ERROR : Invalid number of parameters")
        print("Usage : python test_unknown_images.py /path/to/images [-v]")
    else:
        if sys.argv[1][-1] != "/":
            path = sys.argv[1]+"/"
        else:
            path = sys.argv[1]
        classifieur = load_classifier("drone_final.pkl")
        to_predict = process_graycomatrix(path)
        to_predict = to_predict.reshape(len(to_predict), -1)
        y_predits = classifieur.predict(to_predict)

        if len(sys.argv) >2 and sys.argv[2] == "-v":
            # Partie permettant d'afficher la prediction et le nom de l'image prédite
            files = os.listdir(path)
            for i in range(len(y_predits)):
                print(files[i]+" ("+str(y_predits[i])+")")
        else:
            for prediction in y_predits:
                print(prediction)