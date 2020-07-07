#Written by Elliot Epstein in July 2020.
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import cupy as cp
from matplotlib import rc
from PIL import Image

def f_vec(x,W,a):
    return (1/math.sqrt((W.shape)[0]))*cp.dot(a,cp.maximum(W.dot(cp.transpose(x)),0))

def initialize(m,n,d):
    cp.random.seed(2)
    W = cp.random.normal(size=(m,d))
    y = cp.random.normal(size=n)
    a = 2*cp.around(cp.random.uniform(size=m))-cp.ones(m)
    x_tmp = cp.random.uniform(size=(n,d))
    x = cp.divide(x_tmp,cp.linalg.norm(x_tmp))
    return x,W,a,y

def gradient_step_vec(x,W,a,y,ny):
    diff =f_vec(x,W,a)-y
    H = 0.5*cp.sign(W.dot(cp.transpose(x)))+0.5
    return cp.transpose(cp.multiply(cp.transpose(- ny*(1/math.sqrt((W.shape)[0]))*(cp.multiply(H,diff)).dot(x)),a))

def Loss_vec(x,W,a,y):
    return 0.5*cp.sum(cp.square((f_vec(x,W,a)-y)))

def plot_fig(Loss_k, epochs,m,Pattern_change,Max_dist,Least_eig,lr,n,d):
    Loss_k,Pattern_change,Max_dist,Least_eig,epochs,m = cp.asnumpy(Loss_k),\
    cp.asnumpy(Pattern_change),cp.asnumpy(Max_dist),cp.asnumpy(Least_eig),cp.asnumpy(epochs),cp.asnumpy(m)
    dpi, save_figure = 300, "no"
    for i in range(1, len(m)):
        norm_const = 1+(-Loss_k[0, 0] +Loss_k[0, i])/Loss_k[0,0]
        Loss_k[:, i] = Loss_k[:, i]/norm_const
    Loss_k = np.log(Loss_k)

    plt.clf(),plt.close()
    rc('font', **{'family': 'serif', 'serif': ['Times New Roman']}), rc('text', usetex=True)
    plt.rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{type1ec}', ])
    fig = plt.figure()
    fig.suptitle("Results for synthetic data when training \n the neural network with gradient descent:"
                 " n={}, d={}".format(n,d),ha="center",y =0.98)
    fig.subplots_adjust(wspace=0.5,top=0.8)
    Vars = np.array([Loss_k, Pattern_change, Max_dist, Least_eig])
    title =np.array(["A","B","C","D"])
    ylab = np.array(["log(Training Error)","Fraction Activation changes ",
    "$\operatorname{max}_{r \in [m]} \left\| \mathbf{w}_r(k)-\mathbf{w}_r(0)\\right\|_2$",
                    "Least eigenvalue of $\mathbf{H}(k)$"])
    for j in range(4):
        plt.subplot(int("22"+str(j+1)))
        for i in range(len(Loss_k[0])):
            plt.plot(epochs,Vars[j,:,i],label="m="+str(m[i]))
            plt.title(title[j])
            plt.ylabel(ylab[j])
        if j==0:
            plt.legend(ncol=len(m), frameon=False, loc=(0, 1.12))
        if j==0 or j==1:
            plt.xticks([0,50,100,150,200], [])
        else:
            plt.xticks([0, 50, 100, 150, 200], ["0", "50", "100", "150", "200"])
    plt.show()
    if save_figure =="yes":
        name1="Thesis" + str(np.datetime64("now"))
        fig.savefig(name1, format="png", dpi=dpi)
        (Image.open(name1+".png")).save("Thesis_{}_dpi1.tiff".format(dpi)+str(np.datetime64("now")))

def pattern_changes_vec(W,W_0,x):
    return cp.divide(cp.sum(cp.sign(cp.dot(W, cp.transpose(x))) != cp.sign(cp.dot(W_0, cp.transpose(x)))),cp.size(W))

def max_distance_vec(W,W_0):
    return cp.max(cp.linalg.norm(W-W_0,axis =1))

def timer_fct(which=[]):
    cp.random.seed(2)
    n, d, ny, tries, m = 10, 10, 100, 10,[50, 100, 200, 400,800]
    options=np.array(["gradient_step", "pattern_changes", "Loss", "max_distance", "initialize",
     "create_H", "H_ev"])
    functions = [gradient_step_vec,pattern_changes_vec,Loss_vec,max_distance_vec,initialize,create_H_vec_cp,H_ev_vec]
    for j in range(len(options)):
        if options[j] in which:
            for i in range(len(m)):
                x, W, a, y = initialize(m[i], n, d)
                start_time = time.time()
                for k in range(tries):
                    if j==0:
                        functions[j](x, W, a, y, ny)
                    elif j==1:
                        functions[j](W,W,x)
                    elif j==2:
                        functions[j](x,W,a,y)
                    elif j==3:
                        functions[j](W,W)
                    elif j ==4:
                        x, W, a, y = initialize(m[i], n, d)
                    else:
                        functions[j](x, W)
                cp.cuda.Stream.null.synchronize()
                print("--- %s seconds per call to {}".format(options[j]) % ((time.time() - start_time)/tries))

def time_all():
    timer_fct(np.array(["gradient_step","pattern_changes","Loss","max_distance","initialize", "create_H","H_ev"]))

def H_ev_vec(x,W):
    H = create_H_vec_cp(x,W)
    w = cp.asarray(np.linalg.eigvalsh(cp.asnumpy(H)))
    return cp.amin(w)

def create_H_vec_cp(x, W):
    X = cp.dot(x, cp.transpose(x))
    PHI = 0.5 * cp.sign(cp.dot(W, cp.transpose(x))) + 0.5
    H = cp.multiply(X, cp.dot(cp.transpose(PHI), PHI))
    return (1 / (W.shape)[0]) *H

def main():
    n, d, epochs, ny, seed_nr, save_weights, plt_fig, m=3, 10, 200, 3, 13, "no", "yes",[20,40,80,160,320]
    cp.random.seed(seed_nr)
    Loss_k, Pattern_change, Max_dist, Least_eig= cp.zeros((4,epochs+1, len(m)))
    Epoch_vec = cp.arange(epochs+1)
    time_all()
    for i in range(len(m)):
        x, W, a, y = initialize(m[i], n, d)
        for k in range(epochs+1):
            print("epoch "+str(k)+"/"+str(epochs)+" in round"+str(i))
            if k == 0:
                W_0 = W
            else:
                diff = gradient_step_vec(x,W,a,y,ny)
                W = W + diff

            Pattern_change[k,i],Loss_k[k,i] = pattern_changes_vec(W,W_0,x), Loss_vec(x,W,a,y)
            Max_dist[k,i],Least_eig[k,i] = max_distance_vec(W,W_0), H_ev_vec(x,W)
    if save_weights == "yes":
        cp.savez("Model"+str(np.datetime64("now")),Loss_k=Loss_k,Epoch_vec=Epoch_vec,m=m,Pattern_change=Pattern_change,
                 Max_dist=Max_dist,Least_eig=Least_eig,ny=ny,n=n,d=d,seed_nr=seed_nr)
    if plt_fig =="yes":
        plot_fig(Loss_k,Epoch_vec,m,Pattern_change,Max_dist,Least_eig,ny,n,d)

if __name__ == "__main__":
    main()












