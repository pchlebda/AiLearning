from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"DecisionTreeClassifier accuracy: {acc:.2f}%")

plt.figure()
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()