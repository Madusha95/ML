from sklearn import svm

# Training data: [weight, sweetness] of apples (0) and oranges (1)
X = [[100, 5], [120, 7], [140, 8], [160, 9]]  # Features
y = [0, 0, 1, 1]                               # Labels (0=apple, 1=orange)

# Train SVM
clf = svm.SVC(kernel='linear')  # Linear divider
clf.fit(X, y)

# Predict a new fruit
new_fruit = [[130, 6]]  # Weight=130g, Sweetness=6
print(clf.predict(new_fruit))  # Output: 0 (apple) or 1 (orange)?