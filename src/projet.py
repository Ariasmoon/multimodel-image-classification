import os
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import mahotas
from skimage import data, exposure
from skimage.future import graph
from skimage.measure import regionprops
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from pprint import pprint
from sklearn.model_selection import cross_val_score
import pickle
import warnings

# Constantes
IMAGE_SIZE_TUPLE = (639, 399)
RESIZE_ALGORITHM = Image.LANCZOS
CLASSIFIEUR = RandomForestClassifier(n_estimators=353, max_depth=6, criterion='entropy', min_samples_leaf=5, min_samples_split=15)

# Fonction permettant de générer une image depuis un répertoire 
def image_generator_from_path(path):
    image_list = os.listdir(path)
    for file in image_list:
        yield Image.open(path+file).convert('RGB').resize(IMAGE_SIZE_TUPLE, resample=RESIZE_ALGORITHM)
""" Exemple d'utilisation du generateur d'image
for image in image_generator_from_path("Data/Mer/"):
    pprint(list(image.getdata())[:10])
"""
# Fonction permettant de générer une image depuis un répertoire sous forme de nparray
def image_nparray_generator_from_path(path):
    image_list = os.listdir(path)
    for file in image_list:
        yield np.array(list(Image.open(path+file).convert('RGB').resize(IMAGE_SIZE_TUPLE, resample=RESIZE_ALGORITHM).getdata()))
        
""" Exemple d'utilisation du generateur d'image sous forme de nparray
for image in image_nparray_generator_from_path("Data/Mer/"):
    pprint(list(image.getdata())[:10])
"""
# Fonction de représentation des données sous forme de tableau de pixels
def process_pixels(path = None):
    if (path != None):
        return np.array([pixels for pixels in image_nparray_generator_from_path(path)])
    else:
        x_mer = [pixels for pixels in image_nparray_generator_from_path("Data/Mer/")]
        x_ailleurs = [pixels for pixels in image_nparray_generator_from_path("Data/Ailleurs/")]
        return x_mer, x_ailleurs

# Fonction retournant les nuances/niveaux de bleu d'une image (uniquement canal bleu)
def process_blue_nuances(path = None):
    if (path != None):
        return np.array([pixels[2] for pixels in image_nparray_generator_from_path(path)])
    else:
        x_mer = [pixels[2] for pixels in image_nparray_generator_from_path("Data/Mer/")]
        x_ailleurs = [pixels[2] for pixels in image_nparray_generator_from_path("Data/Ailleurs/")]
        return x_mer, x_ailleurs

# Fonction renvoyant l'histogramme des couleurs d'une image donnée 
def process_histogram(path = None, nb_bins = 768):
    if (path != None):
        return np.array([np.histogram(image, bins = nb_bins)[0] for image in image_generator_from_path(path)])
    else:
        x_mer = [np.histogram(image, bins = nb_bins)[0] for image in image_generator_from_path("Data/Mer/")]
        x_ailleurs = [np.histogram(image, bins = nb_bins)[0] for image in image_generator_from_path("Data/Ailleurs/")]
        return x_mer, x_ailleurs

# Fonction renvoyant l'histogramme des gradients d'une image donnée 
def process_hog(path = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (path != None):
            return np.array([hog(image.resize((256,512), resample=RESIZE_ALGORITHM),
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        multichannel=True)[1] for image in image_generator_from_path(path)])
        else:
            x_mer = [hog(image.resize((256,512), resample=RESIZE_ALGORITHM),
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        multichannel=True)[1] for image in image_generator_from_path("Data/Mer/")]
            x_ailleurs = [hog(image.resize((256,512), resample=RESIZE_ALGORITHM),
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        multichannel=True)[1] for image in image_generator_from_path("Data/Ailleurs/")]
            return x_mer, x_ailleurs

# Fonction renvoyant les motifs binaires locaux d'une image donnée 
def process_lbp(n_points=24, radius=3, path=None):
    if (path != None):
        return np.array([local_binary_pattern(image, P=n_points, R=radius) for image in image_nparray_generator_from_path(path)])
    else:
        x_mer = [local_binary_pattern(image, P=n_points, R=radius) for image in image_nparray_generator_from_path("Data/Mer/")]
        x_ailleurs = [local_binary_pattern(image, P=n_points, R=radius) for image in image_nparray_generator_from_path("Data/Ailleurs/")]
        return x_mer, x_ailleurs

# Fonction renvoyant la texture haralick d'une image donnée
def process_haralick(path = None):
    if (path != None):
        return np.array([mahotas.features.haralick(pixels) for pixels in image_nparray_generator_from_path(path)])
    else:
        x_mer = [mahotas.features.haralick(pixels) for pixels in image_nparray_generator_from_path("Data/Mer/")]
        x_ailleurs = [mahotas.features.haralick(pixels) for pixels in image_nparray_generator_from_path("Data/Ailleurs/")]
        return x_mer, x_ailleurs

# Fonction renvoyant la matrice de co-occurence des niveaux de gris d'une image donnée 
def process_graycomatrix(path = None, distances_list=[4], angles_list=[30]):
    if (path != None):
        return np.array([graycomatrix(pixels, distances=distances_list, angles=angles_list, levels = 256) for pixels in image_nparray_generator_from_path(path)])
    else:
        x_mer = [graycomatrix(pixels, distances=distances_list, angles=angles_list, levels = 256) for pixels in image_nparray_generator_from_path("Data/Mer/")]
        x_ailleurs = [graycomatrix(pixels, distances=distances_list, angles=angles_list, levels = 256) for pixels in image_nparray_generator_from_path("Data/Ailleurs/")]
        return x_mer, x_ailleurs

# Fonction de préparation des données étiquetées 
def prepare_dataset(x_mer,x_ailleurs):
    X = x_mer
    Y = [1] * (len(x_mer))
    for i in x_ailleurs:
        X.append(i)
        Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(Y.shape[0], -1)
    return (X, Y)

# Fonction de préparation pour concaténation de données 
def prepare_dataset2(x_mer2,x_ailleurs2):
    X1 = x_mer2
    Y1 = [1] * (len(x_mer2))
    for i in x_ailleurs2:
        X1.append(i)
        Y1.append(-1)
    X1 = np.array(X1)
    Y1 = np.array(Y1)
    X1 = X1.reshape(Y1.shape[0], -1)
    print("X1 :", X1.shape)
    return (X1, Y1)

# Fonction de sauvegarde des étiquettes des données mi-parcours
def store_test_labels(path):
    labels = []
    for filename in os.listdir(path)[0:]:
        y = 1
        if (int(filename[0]) == 0):
            y = -1
        labels.append(y)
    return(np.array(labels))

# Fonction de chargement du classifieur 
def load_classifier(path):
    with open(path, 'rb') as open_file:
        return pickle.load(open_file)

# Fonction de sauvegarde du classifieur 
def store_classifier(classifier, path):
    with open(path, "wb") as open_file:
        pickle.dump(classifier, open_file)

# Fonction renvoyant la représentation de données optimale en termes de score
def find_best_image_processing_protocol(classifieur = CLASSIFIEUR):
    for process in [process_pixels,
                    process_blue_nuances,
                    process_histogram,
                    process_lbp,
                    process_hog,
                    process_haralick,
                    process_graycomatrix]:
        x_mer, x_ailleurs = process()
        X, Y = prepare_dataset(x_mer, x_ailleurs)
        scores = cross_val_score(classifieur, X, Y, cv=5)
        print("Traitement : %s, score de précision : %0.3f, déviation standard : %0.3f" % (str(process.__name__), scores.mean(), scores.std()))

# Fonction permettant de trouver la meilleure combinaison représentation des données/classifieur
def find_best_combo():
    for classifier in [RandomForestClassifier(),DecisionTreeClassifier(),GaussianNB(),KNeighborsClassifier()]:
        print(str(classifier.__class__())[:-2])
        find_best_image_processing_protocol(classifieur=classifier)
        print("---\n\n")
        
# Fonction renvoyant les paramètres optimaux de l'histogramme des couleurs
def hist_fine_tuning():
    results = []
    for bins in range(10,780,10):
        x_mer, x_ailleurs = process_histogram(nb_bins=bins)
        X, Y = prepare_dataset(x_mer, x_ailleurs)
        scores = cross_val_score(CLASSIFIEUR, X, Y, cv=5)
        print("Score de précision : %0.3f, déviation standard : %0.3f" % (scores.mean(), scores.std()))
        results.append((bins,scores.mean(),scores.std()))
    
    with open("histogram_bin_number_tuning_results.csv","w") as file:
        for line in results:
            file.write(str(line)[1:-1]+"\n")

# Fonction permettant de renvoyer les paramètres optimaux de la matrice de co-occurence des niveaux de gris
def graycomatrix_fine_tuning():
    results = []
    for angles_list in [[0], [30], [45], [60], [90]]:
        for distances_list in [[2], [3], [4], [5], [6]]:
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Processing started"))
            x_mer, x_ailleurs = process_graycomatrix(angles_list=angles_list, distances_list=distances_list)
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Processing done"))
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Dataset preparation started"))
            X, Y = prepare_dataset(x_mer, x_ailleurs)
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Dataset preparation done"))
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Cross validation started"))
            scores = cross_val_score(CLASSIFIEUR, X, Y, cv=5)
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Cross validation done"))
            print("Score de précision : %0.3f, déviation standard : %0.3f" % (scores.mean(), scores.std()))
            print(("Angle : "+str(angles_list)+", Distance : "+str(distances_list)+" - Results : "+str(scores.mean())+"+-"+str(scores.std())))
            results.append((angles_list, distances_list, scores.mean(),scores.std()))
            print("Iteration results saved, moving on...")
    
    with open("graycomatrix_tuning_results.csv","w") as file:
        for line in results:
            file.write(str(line)[1:-1]+"\n")

if __name__ == "__main__":
    x_mer, x_ailleurs = process_graycomatrix()
    X, Y = prepare_dataset(x_mer, x_ailleurs)
    
    # Partie Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    CLASSIFIEUR.fit(X_train, y_train)
    y_predicts = CLASSIFIEUR.predict(X_test)
    print("Les vraies classes :")
    print(y_test)
    print("Les classes prédites :")
    print(y_predicts)
    CLASSIFIEUR.score(X_test, y_test)
    print(accuracy_score(y_test,y_predicts))
    
    # Partie validation croisée
    scores = cross_val_score(CLASSIFIEUR, X, Y, cv=5)
    print("Score de précision : %0.3f, écart-type : %0.3f" % (scores.mean(), scores.std()))

    # Sauvegarde du classifieur
    store_classifier(CLASSIFIEUR, "classifieur.pkl")
