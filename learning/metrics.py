from sklearn.metrics import accuracy_score

#Acurracy
y_pred = [0, 2, 1, 3, 1]
y_true = [0, 1, 2, 3, 0]

accuracy = accuracy_score(y_true, y_pred)

print(accuracy)