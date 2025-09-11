# control/mpc.py
import casadi as ca
import numpy as np

def solve_bicycle_mpc(x0, xs, v_max=5.0, N=12, dt=0.1, L=2.5):
    """
    Build fresh opti problem and solve it. Return X_opt (4 x (N+1)) and U_opt (2 x N).
    x0, xs: numpy arrays length 4 [x,y,yaw,v]
    """
    # states: x,y,yaw,v
    x = ca.SX.sym('x'); y = ca.SX.sym('y'); yaw = ca.SX.sym('yaw'); v = ca.SX.sym('v')
    states = ca.vertcat(x,y,yaw,v)
    a = ca.SX.sym('a'); delta = ca.SX.sym('delta')
    controls = ca.vertcat(a, delta)

    rhs = ca.vertcat(v*ca.cos(yaw),
                     v*ca.sin(yaw),
                     v/L*ca.tan(delta),
                     a)
    f = ca.Function('f', [states, controls], [rhs])

    opti = ca.Opti()
    X = opti.variable(4, N+1)
    U = opti.variable(2, N)
    P = opti.parameter(8)  # x0 + xs

    Q = ca.diag(ca.DM([5,5,1,1]))
    R = ca.diag(ca.DM([1,1]))

    obj = 0
    g = []
    # initial cond
    opti.subject_to(X[:,0] == P[0:4])
    for k in range(N):
        st = X[:,k]; con = U[:,k]
        st_next = X[:,k+1]
        f_val = f(st, con)
        st_next_euler = st + dt * f_val
        opti.subject_to(st_next == st_next_euler)
        err = st - P[4:8]
        obj += ca.mtimes([err.T, Q, err]) + ca.mtimes([con.T, R, con])

    opti.minimize(obj)
    # control bounds
    opti.subject_to(opti.bounded(-2, U[0,:], 2))
    opti.subject_to(opti.bounded(-0.5, U[1,:], 0.5))
    # velocity bounds (state index 3)
    opti.subject_to(opti.bounded(0, X[3,:], v_max))

    # solver
    p_opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt', p_opts)

    # set parameter values
    opti.set_value(P, np.concatenate([x0, xs]))
    # initial guesses
    opti.set_initial(X, np.tile(x0.reshape(-1,1), (1, N+1)))
    opti.set_initial(U, 0)

    try:
        sol = opti.solve()
    except Exception as e:
        print("MPC solve failed:", e)
        return None, None

    X_opt = sol.value(X)
    U_opt = sol.value(U)
    return X_opt, U_opt
