# Le code de ce fichier provient en grande partie de cette source : https://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from projet import *

def objective(trial):
    x_mer, x_ailleurs = process_graycomatrix()
    X, Y = prepare_dataset(x_mer, x_ailleurs)
    
    n_estimators = trial.suggest_int('n_estimators', 2, 800)
    max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))
    
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth)
    
    return sklearn.model_selection.cross_val_score(clf, X, Y, n_jobs=-1, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
optuna.visualization.plot_optimization_history(study).write_image("optuna_results.png")