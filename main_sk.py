# Author: Nick Sebasco
# Assignment 4 Machine Learning - generate nonlinear data, use kernel trick + SVM to separate & classify data. 
# Date: 5/2/2021
# 0) Imports
from gen_data_v2 import generate, traintest, plot_data 
from sklearn import svm
import numpy as np

# 1) Generate & partition data into test/ train sets.  
data, labels = generate(5,5,3,var1x =0.9, var1y = 0.6, var2 = 1, bias = 5, examples = 150)
train, test = traintest(data, labels)
trainX, trainY = train
testX, testY = test

# 2) build a linear classifier
# The kernel trick allows data to be mapped into a higher dimension so an optimal
# separating hyperplane can be found.
kernel_trick = lambda X, Y: X**2 + Y**2
useRBF_kernel = False # Cheat by using the builtin rbf kernel option.
use_kernel_trick = True # Use the kernel trick to map into 3D space.
supportVectorClassifier = svm.SVC(kernel="linear" if not useRBF_kernel else "rbf")
trainR = [[i[0],i[1],i[0]**2 + i[1]**2] for i in trainX] if use_kernel_trick else trainX
supportVectorClassifier.fit(trainR, trainY)

# 3) test accuracy
correct = 0
for tX, tY in zip(testX, testY):
    tX = list(tX) + [tX[0]**2+tX[1]**2]
    pred = supportVectorClassifier.predict(np.array(tX).reshape(1,-1))
    print(f"predicted: {pred[0]}, actual: {tY}")
    correct += 1 if pred[0] == tY else 0
print("Test accuracy: ")
print(f"correct: {correct}/ {len(testX)}")

# 4) Generate plot
plot_data(data, labels, clf=supportVectorClassifier, kernel_trick=kernel_trick, plot3D=True)

