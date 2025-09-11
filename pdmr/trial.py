import casadi as ca
import numpy as np
#states
x = ca.SX.sym('x')
y = ca.SX.sym('y')
yaw = ca.SX.sym('yaw')
v = ca.SX.sym('v')
states = ca.vertcat(x, y, yaw, v)
#print(states)

#control
a = ca.SX.sym('a')
delta = ca.SX.sym('delta')
controls = ca.vertcat(a, delta)
#print(controls)

#bicycle model
L = 2.5
rhs = ca.vertcat(
    v*ca.cos(yaw),# x_dot
    v*ca.sin(yaw),#y_dot
    v/L*ca.tan(delta),# yaw_dot
    a# v_dot
    )
f = ca.Function('f', [states, controls], [rhs]) #--> for state forward propogation

#Optimisation

N = 12 #horizon
dt = 0.1

optim = ca.Opti() #casadi optimistaion object
X = optim.variable(4, N+1) #State trajectory
U = optim.variable(2, N) #control trajetory
P = optim.parameter(8)

#dynamic constraints
for k in range(N):
    ste = X[:, k]
    con = U[:, k]
    ste_next = X[:, k+1]
    f_val = f(ste, con)
    ste_next_euler = ste + f_val*dt
    optim.subject_to(ste_next == ste_next_euler)

#cost function
Q = ca.diag(ca.DM([5, 5, 1, 1]))
R = ca.diag(ca.DM([1, 1]))

obj = 0
for k in range(N):
    ste = X[:, k]
    con = U[:, k]
    error = ste - P[4 : 8]
    obj += ca.mtimes([error.T, Q, error]) + ca.mtimes([con.T, R, con])
optim.minimize(obj)

# Control bounds
optim.subject_to(optim.bounded(-2, U[0,:], 2))    # acceleration
optim.subject_to(optim.bounded(-0.5, U[1,:], 0.5)) # steering

# Velocity bounds
v_max = 5.0
optim.subject_to(optim.bounded(0, X[3,:], v_max))

optim.solver('ipopt', {"ipopt.print_level":0, "print_time":0})

# set parameters
x0 = [0,0,0,0]
xs = [10,10,0,0]
optim.set_value(P, x0 + xs)

# initial guesses
optim.set_initial(X, np.tile(np.array(x0).reshape(-1,1), (1,N+1)))
optim.set_initial(U, 0)

sol = optim.solve()
X_opt = sol.value(X)
U_opt = sol.value(U)

print(X_opt)
print(U_opt)
