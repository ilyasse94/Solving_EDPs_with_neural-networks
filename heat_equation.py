import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.animation import FuncAnimation


#Discretization on both axis
nx = 15
ny = 15
nt = 7

#Distance between each two adyascent points
dx = 1. / nx
dy = 1. / ny
dt = 1. / nt

#Construction of the grid
x_space = np.linspace(0, 0.5, nx)
y_space = np.linspace(0, 0.5, ny)
t_space = np.linspace(0, 3, nt)


#f(x)
def f(x, r = 1):
    # return np.exp(-x[0]) * (x[0] - 2. + x[1]**3 + 6*x[1])
    return 0.

#Activation function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    a2 = sigmoid(np.dot(a1, W[1]))
    return np.dot(a2, W[2])

"""
def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])
"""

def A(x):
   
    return  np.exp(x[0] + x[1])

def B(x):
   
    return 0.


def psy_trial(x, net_out):
    return  (x[0] * (1. - x[0]) * x[1] * (1. - x[1]) * A(x) +  x[2] * B(x) + x[0] * (1. - x[0]) * x[1] * (1. - x[1]) * x[2] * net_out)


def loss_function(W, x, y, t):
    loss_sum = 0.
    
    for xi in x:
        for yi in y: 
            for ti in t:
                input_point = np.array([xi, yi, ti])        
                net_out = neural_network(W, input_point)[0]
                #psy_t = psy_trial(input_point, net_out)
                psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
                psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)
    
                gradient_of_trial_d2x = psy_t_hessian[0][0]
                gradient_of_trial_d2y = psy_t_hessian[1][1]
                gradient_of_trial_dt = psy_t_jacobian[2]
    
                func = f(input_point) # right part function
    
                err_sqr = ((gradient_of_trial_dt - 3 * (gradient_of_trial_d2x + gradient_of_trial_d2y)) - func)**2
                loss_sum += err_sqr
    return loss_sum

W = [npr.randn(3, nx), npr.randn(ny, nt), npr.randn(nt, 1)]

lmb = 0.000001



for i in range(10):
    loss_grad =  grad(loss_function)(W, x_space, y_space, t_space)
    
    print (i+1, loss_function(W, x_space, y_space, t_space)/(nx+ny+nt))
    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    W[2] = W[2] - lmb * loss_grad[2]
    if i % 10 == 0 and i != 0:
    	lmb *= 0.95
    	print ('learning rate decrease')


surface = np.zeros((ny, nx, nt))

        
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        for k, t in enumerate(t_space):
            net_outt = neural_network(W, [x, y, t])[0]
            surface[i][j][k] = psy_trial([x, y, t], net_outt)

        
for z, t in enumerate(t_space):        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x_space, y_space)
    surf = ax.plot_surface(X, Y, surface[:,:,z], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(0, 0.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    plt.show()


