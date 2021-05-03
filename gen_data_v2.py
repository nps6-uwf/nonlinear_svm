# Author: Nick Sebasco
# version: 1
# generate nonlinearly separable data in an effort to use kernel trick + svm binary classification.
# 2 dimensions, 2 classes
import numpy as np
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt

np.random.seed(int(time())) # seed prng

def generate(xc, yc, r, var1x=0.1, var1y=0.2, var2=10, classSplit = 0.5, examples = 10, bias = 5,
    a_label = "classA", b_label = "classB"):
    """
    imagine a circle with center (xc, yc) of radius (r):
    let the interior of the square consist of members of class A
    and the exterior consist of the members of class B.

    var1/ var2: variance of classA (inside circle) and classB respectively.
    classSplit: fraction of examples of classA, (1-classSplit) fraction of classB

    (x - xc)^2 + (y - yc)^2 = r^2 -> (r^2 - (x - xc)^2)^1/2 + yc
    """
    # find y coordinate of circle given x coordinate & radius
    circleY = lambda x, r: (yc + (r**2-(x - xc)**2)**(0.5), yc - (r**2-(x - xc)**2)**(0.5))
    data, labels = [], []
    N_a = int(classSplit*examples)
    N_b = examples - N_a
    for i in range(N_a): # inside circle
        xlb, xub = xc - (r*var1x), xc + (r * var1x)
        rand_x = xlb + np.random.random() * (xub - xlb)
        yub, ylb = circleY(rand_x, r)
        yub = yub - (r * (1-var1y))
        ylb = ylb - (r * (1-var1y))
        rand_y = ylb + np.random.random() * (yub - ylb)
        data.append((rand_x, rand_y, a_label))

    for i in range(N_b): # outside circle
        # choose a random radius with var2 * r
        rand_r = (r+bias) + np.random.random() * (var2 * r)
        xlb, xub = xc - rand_r, xc + rand_r
        rand_x = xlb + np.random.random() * (xub - xlb)
        yub, ylb = circleY(rand_x, rand_r)
        rand_y = np.random.choice((yub, ylb))
        data.append((rand_x, rand_y, b_label))
    np.random.shuffle(data)
    return ([(i,j) for i, j, _ in data], [k for _, _, k in data])

def traintest(X, Y, frac = 0.9):
    """Split data into testing/ training dataset.
    """
    return ((X[:int(len(X)*frac)], Y[:int(len(X)*frac)]), (X[int(len(X)*frac):], Y[int(len(X)*frac):]))

def plot_data(data=None, labels=None, clf=None, plot3D=False, kernel_trick=lambda X,Y: X + Y):
    # If no data was passed to the function, generate some.
    if data == None:
        data, labels = generate(5,5,3,var1x =0.9, var1y = 0.6, var2 = 1, bias = 5, examples = 300)
    X, Y = np.array([i[0] for i in data]), np.array([i[1] for i in data])
    labels = np.array(labels)
    

    if not plot3D:
        # standard 2D scatter plot of data.
        fig, ax = plt.subplots()
        for g in np.unique(labels):
            i = np.where(labels == g)
            ax.scatter(X[i], Y[i], label=g)
        ax.legend()

    if clf and plot3D:
        # if a svm classifier is provided in the clf argument.
        # plot the decision function
        print("start plot 3d/clf")
        z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x-clf.coef_[0][1]*y) / clf.coef_[0][2]

        tmp = np.linspace(-5,12,51)
        x,y = np.meshgrid(tmp,tmp)

        # Plot stuff.
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        
        for g in np.unique(labels):
            i = np.where(labels == g)
            ax.scatter3D(X[i], Y[i], X[i]**2 + Y[i]**2,  label=g)
        ax.legend()
        ax.plot_surface(x, y, z(x,y), 
            cmap=cm.summer, antialiased=False, cstride=10, rstride=10, alpha=0.5)
        #ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
        #ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
    elif clf and not plot3D:
        # if a svm classifier is provided in the clf argument.
        # plot the decision function
        ax = plt.gca() # what is gca?  get current axis, if one doesn't exist one is created.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['black','r','black'], levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', 'solid', '--']) # default: ['--', '-', '--']
        # plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    if plot3D and not clf:
        # apply the kernel trick to the data and visualize the separable data
        # in 3D space.
        r = kernel_trick(X, Y)
        elev=30
        azim=30
        y=Y
        c = np.array(["r" if i == "classA" else "b" for i in labels])
        ax = plt.subplot(projection='3d')
        ax.scatter3D(X, Y, r, c=c, s=50)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')
    plt.show()



