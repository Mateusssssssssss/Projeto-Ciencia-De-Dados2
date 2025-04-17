from sklearn.svm import SVC
from notebooks.preprocess import *
from sklearn.model_selection import cross_val_score
# 2. Treinar o modelo SVM
svc = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
#Validação cruzada
results = cross_val_score(svc, x_train, y_train, cv=3)
print(f'Cross Validation: {results}')

svc.fit(x_train, y_train)