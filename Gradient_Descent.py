#Written by Elliot Epstein in July 2020.
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import cupy as cp
from matplotlib import rc
from PIL import Image
import sys, getopt


def f_vec(x, W, a):
    return (1/math.sqrt(W.shape[0]))*cp.dot(a, cp.maximum(W.dot(cp.transpose(x)), 0))


def initialize(m, n, d):
    W = cp.random.normal(size=(m, d))
    y = cp.random.normal(size=n)
    a = 2*cp.around(cp.random.uniform(size=m))-cp.ones(m)
    x_tmp = cp.random.uniform(size=(n, d))
    x = cp.divide(x_tmp, cp.linalg.norm(x_tmp))
    return x, W, a, y


def gradient_step_vec(x, W, a, y, ny):
    diff = f_vec(x, W, a)-y
    H = 0.5*cp.sign(W.dot(cp.transpose(x)))+0.5
    return cp.transpose(cp.multiply(cp.transpose(- ny*(1/math.sqrt(W.shape[0]))*(cp.multiply(H, diff)).dot(x)), a))


def loss_vec(x, W, a, y):
    return 0.5*cp.sum(cp.square((f_vec(x, W, a)-y)))

def load_weights(model):
    npz = cp.load(model)

    return (npz["loss_k"], npz["epoch_vec"], npz["m"], npz["pattern_change"], npz["max_dist"], npz["least_eig"],
      npz["ny"], npz["n"], npz["d"])


def plot_fig(loss_k, epochs, m, pattern_change, max_dist, least_eig, n, d,save_fig):
    loss_k, pattern_change, max_dist, least_eig, epochs, m = cp.asnumpy(loss_k),\
    cp.asnumpy(pattern_change), cp.asnumpy(max_dist), cp.asnumpy(least_eig), cp.asnumpy(epochs), cp.asnumpy(m)
    dpi = 300
    for i in range(1, len(m)):
        norm_const = 1+(-loss_k[0, 0] + loss_k[0, i])/loss_k[0, 0]
        loss_k[:, i] = loss_k[:, i]/norm_const
    loss_k = np.log(loss_k)

    plt.clf(),plt.close()
    rc('font', **{'family': 'serif', 'serif': ['Times New Roman']}), rc('text', usetex=True)
    plt.rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{type1ec}', ])
    fig = plt.figure()
    fig.suptitle("Results for synthetic data when training \n the neural network with gradient descent:"
                 " n={}, d={}".format(n, d), ha="center", y=0.98)
    fig.subplots_adjust(wspace=0.5, top=0.8)
    vars = np.array([loss_k, pattern_change, max_dist, least_eig])
    title =np.array(["A", "B", "C", "D"])
    ylab = np.array(["log(Training Error)", "Fraction Activation changes ",
     "$\operatorname{max}_{r \in [m]} \left\| \mathbf{w}_r(k)-\mathbf{w}_r(0)\\right\|_2$",
                    "Least eigenvalue of $\mathbf{H}(k)$"])
    for j in range(4):
        plt.subplot(int("22"+str(j+1)))
        for i in range(len(loss_k[0])):
            plt.plot(epochs, vars[j, :, i], label="m="+str(m[i]))
            plt.title(title[j])
            plt.ylabel(ylab[j])
        if j == 0:
            plt.legend(ncol=len(m), frameon=False, loc=(0, 1.12))
        if j == 0 or j == 1:
            plt.xticks([0, 50, 100, 150, 200], [])
        else:
            plt.xticks([0, 50, 100, 150, 200], ["0", "50", "100", "150", "200"])
    plt.show()
    if save_fig == "yes":
        name1 = "Thesis" + str(np.datetime64("now"))+".png"
        fig.savefig(name1, format="png", dpi=dpi)

        temp =Image.open(name1)
        name = "Thesis_{}_dpi1".format(dpi) + str(np.datetime64("now"))+".tiff"
        temp.save(name)


def pattern_changes_vec(W, W_0, x):
    return cp.divide(cp.sum(cp.sign(cp.dot(W, cp.transpose(x))) != cp.sign(cp.dot(W_0, cp.transpose(x)))), cp.size(W))


def max_distance_vec(W, W_0):
    return cp.max(cp.linalg.norm(W-W_0, axis=1))


def timer_fct(which=np.zeros(1)):
    cp.random.seed(2)
    n, d, ny, tries, m = 10, 10, 100, 10,[50, 100, 200, 400,800]
    options=np.array(["gradient_step", "pattern_changes", "Loss", "max_distance", "initialize",
     "create_H", "H_ev"])
    functions = [gradient_step_vec, pattern_changes_vec, loss_vec, max_distance_vec,
                 initialize, create_H_vec_cp, H_ev_vec]
    for j in range(len(options)):
        if options[j] in which:
            for i in range(len(m)):
                x, W, a, y = initialize(m[i], n, d)
                start_time = time.time()
                for k in range(tries):
                    if j == 0:
                        functions[j](x, W, a, y, ny)
                    elif j == 1:
                        functions[j](W, W, x)
                    elif j == 2:
                        functions[j](x, W, a, y)
                    elif j == 3:
                        functions[j](W, W)
                    elif j == 4:
                        x, W, a, y = initialize(m[i], n, d)
                    else:
                        functions[j](x, W)
                cp.cuda.Stream.null.synchronize()
                print("--- %s seconds per call to {}".format(options[j]) % ((time.time() - start_time)/tries))


def time_all():
    timer_fct(np.array(["gradient_step", "pattern_changes", "Loss", "max_distance", "initialize", "create_H", "H_ev"]))


def H_ev_vec(x, W):
    H = create_H_vec_cp(x, W)
    w = cp.asarray(np.linalg.eigvalsh(cp.asnumpy(H)))
    return cp.amin(w)


def create_H_vec_cp(x, W):
    X = cp.dot(x, cp.transpose(x))
    phi = 0.5 * cp.sign(cp.dot(W, cp.transpose(x))) + 0.5
    H = cp.multiply(X, cp.dot(cp.transpose(phi), phi))
    return H /W.shape[0]


def iterate(epochs, m, n, d, ny,verbose=0):
    loss_k, pattern_change, max_dist, least_eig = cp.zeros((4, epochs + 1, len(m)))
    for i in range(len(m)):
        x, W, a, y = initialize(m[i], n, d)
        for k in range(epochs+1):
            if verbose==1:
               print("epoch "+str(k)+"/"+str(epochs)+" in round"+str(i))
            if k == 0:
                W_0 = W
            else:
                diff = gradient_step_vec(x, W, a, y, ny)
                W = W + diff
            pattern_change[k, i],loss_k[k, i] = pattern_changes_vec(W, W_0, x), loss_vec(x, W, a, y)
            max_dist[k, i], least_eig[k, i] = max_distance_vec(W, W_0), H_ev_vec(x, W)
    return loss_k, pattern_change, max_dist, least_eig


def main():

    n, d, epochs, ny, seed_nr, m = 100, 1000, 200, 3, 14, [200, 400, 800, 1600, 3200]
    save_weights, plt_fig, save_fig, timer,verbose = "no", "no", "no", "no", 0
    cp.random.seed(seed_nr)
    epoch_vec = cp.arange(epochs + 1)

    if timer =="yes":
        time_all()
    loss_k, pattern_change, max_dist, least_eig = iterate(epochs, m, n, d, ny,verbose)

    if save_weights == "yes":
        cp.savez("Model"+str(np.datetime64("now")), loss_k=loss_k, epoch_vec=epoch_vec, m=m,
                 pattern_change=pattern_change, max_dist=max_dist,
                 least_eig=least_eig, ny=ny, n=n, d=d, seed_nr=seed_nr)
    if plt_fig == "yes":
        plot_fig(loss_k, epoch_vec, m, pattern_change, max_dist, least_eig, n, d,save_fig)


if __name__ == "__main__":
    main()












