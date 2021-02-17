import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, auc, roc_curve
import copy

# Comun a todos los modelos
def get_data():
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq")
    return df

def get_prediction():
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0")
    return df['volveria']

def get_holdout():
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A")
    return df

# Divide los datos para el test y el holdout
def split_data(data, result):        
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        data, result, random_state=20, test_size=0.1, stratify=result
    )
    return X_train, X_holdout, y_train, y_holdout

# Plotea las metricas pasadas en un grafico de barras
def plot_metrics(y, proba, predict):
    serie = pd.Series(
        [
            roc_auc_score(y, proba), 
            accuracy_score(y, predict), 
            precision_score(y, predict), 
            recall_score(y, predict),
            f1_score(y, predict)
        ],
        index=['Roc_auc', 'Accuracy', 'Precision', 'Recall', 'F1']
    )
    
    plt.figure(dpi=150)
    serie.plot(kind = 'bar', rot = 1)
    plt.title("Metricas")
    plt.xlabel("Score")
    plt.ylabel("Porcentaje")
    plt.ylim([0,1])

    plt.show()
    display(serie)

# Plotea la matriz de confusion
def plot_confusion(model, x, y):
    fig, ax = plt.subplots(figsize=(15, 7))

    plt.grid(False)
    plot_confusion_matrix(
        model, x, y, cmap=plt.cm.Blues, display_labels=['1', '0'], ax=ax
    )
    
    plt.show()
    
# Plotea la curva auc roc
def plot_roc(y, pred_proba):
    _fpr, _tpr, thresholds = roc_curve(y, pred_proba)
    roc_auc = auc(_fpr, _tpr)

    plt.figure(figsize=(15, 10))
    plt.plot(
        _fpr, _tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})'
    )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    display(roc_auc_score(y, pred_proba))
    
# Plotea varias curvas auc roc
# Devuelve el numero del mejor modelo
sns.set()
def plot_rocs(test_pred, y_test, columns, rows):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    max_auc_index = 0
    max_auc = 0
    for i in range(columns * rows):
        fpr[i], tpr[i], _ = roc_curve(y_test, test_pred[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if roc_auc[i] > max_auc:
            max_auc = roc_auc[i]
            max_auc_index = i
        
    plt.figure()
    lw = 2
    cl = 0
    fig, axs = plt.subplots(rows, columns, figsize=(15, 5 * rows), sharex=True, sharey=True)
    if rows == 1:
        axs = [axs]
    
    i = 0
    for ax in axs:
        j = 0
        for col in ax:
            col.plot(
                fpr[cl],
                tpr[cl],
                color='darkorange',
                lw=lw,
                label='(AUC= %0.4f)' % (roc_auc[cl]),
            )
            cl += 1
            col.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            col.set_xlim([0.0, 1.0])
            col.set_ylim([0.0, 1.05])
            col.set_title('ROC model %s' % (cl-1))
            col.legend(loc="lower right")
    
    for i in range(columns):
        if rows > 1:
            axs[rows-1, i].set_xlabel('False Positive Rate', weight="bold")
        else:
            axs[0][i].set_xlabel('False Positive Rate', weight="bold")
    
    for i in range(rows):
        if rows > 1:
            axs[i, 0].set_ylabel('True Positive Rate', weight="bold")
        else:
            axs[0][i].set_ylabel('True Positive Rate', weight="bold")
    
    plt.subplots_adjust()
    return max_auc_index

# Entrena para un tipo de modelo, distintos tipos de preprocesado
# Devuelve una lista con el mejor modelo y su respectivo preprocesado.
def training(model, X, y, list_preprocessing):
    result = []
    for p in list_preprocessing:
        X_ = p.transform(X)
        m = copy.deepcopy(model)
        
        m.fit(X_,y)
        result.append((m,p))
    
    return result

# Entrena para un tipo de modelo, distintos tipos de preprocesado con una division de kfolds=5
# buscando los mejores hiperparametros pasados por argumento.
# Devuelve una lista con el mejor modelo y su respectivo preprocesado.
def super_training(X, y, list_preprocessing, searchCV):
    result = []
    for p in list_preprocessing:
        X_ = p.transform(X)
        scv = searchCV.fit(X_, y)
        result.append((scv.best_estimator_, p))
        
    return result

# Preprocesa el dataset pasado por parametro y devuelve las probabilidades de predicciones positivas de 
# cada modelo en la lista pasada por argumento
def get_proba_predicts(model_preprocessing, X):
    proba_predictions = []
    for m,p in model_preprocessing:
        X_ = p.transform(X)
        proba_predictions.append(m.predict_proba(X_)[:,1])
    
    return proba_predictions