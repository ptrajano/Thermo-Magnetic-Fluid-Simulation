from numba import njit
import numpy as np

@njit
def velocidade_x_star(u, v, u_star, theta, Pr, Ram, N_x, N_y, dx, dy, dt, dH_x2, in_obs, obstacles):
    for i in range(1, N_x):
        for j in range(0, N_y):
            if not in_obs[i, j]:    
                c1 = (v[i, j+1] + v[i-1, j+1] + v[i, j] + v[i-1, j])/4
                r = -dt*(u[i, j]*((u[i+1, j]-u[i-1, j])/(2*dx)) + c1*((u[i, j+1]-u[i, j-1])/(2*dy))) + Pr*dt*((u[i+1, j]-(2*u[i, j]) + u[i-1, j])/(dx**2) +(u[i, j+1]-(2*u[i, j]) + u[i, j-1])/(dy**2)) - 0.5*dt*Pr*Ram*theta[i,j]*dH_x2[i,j]
                u_star[i, j] = u[i, j] + r
    
    u_star[:, N_y] = -u_star[:, N_y-1]
    u_star[:, -1] = -u_star[:, 0]
    u_star[0, :] = 0
    u_star[N_x, :] = 0
    
    for a_x, a_y, b_x, b_y in obstacles:
        if (a_x != b_x) and (a_y != b_y):
            u_star[:, int(b_y//dy) - 1] = -u_star[:, int(b_y//dy)]
            u_star[:, int(a_y//dy)] = -u_star[:, int(a_y//dy)-1]

@njit
def velocidade_y_star(u, v, v_star, theta, Pr, Ra, Ram, N_x, N_y, dx, dy, dt, dH_y2, in_obs, obstacles):
    for i in range(0, N_x):
        for j in range(1, N_y):
            if not in_obs[i, j]: 
                c2 = (u[i, j+1] + u[i-1, j+1] + u[i, j] + u[i-1, j])/4
                r = -dt*(c2*((v[i+1, j]-v[i-1, j])/(2*dx)) + v[i, j]*((v[i, j+1]-v[i, j-1])/(2*dy))) + Pr*dt*((v[i+1, j]-(2*v[i, j]) + v[i-1, j])/(dx**2) + (v[i, j+1]-(2*v[i, j]) + v[i, j-1])/(dy**2))+dt*Pr*Ra*(theta[i, j + 1] + theta[i, j])/2 - 0.5*dt*Pr*Ram*theta[i,j]*dH_y2[i,j]
                v_star[i, j] = v[i, j] + r 

    for j in range(0, N_y+1):
        v_star[-1, j] = -v_star[0, j]
        v_star[N_x, j] = -v_star[N_x-1, j]

    for i in range(0, N_x):
        v_star[i, 0] = 0
        v_star[i, N_y] = 0
    
    for a_x, a_y, b_x, b_y in obstacles:
        if (a_x != b_x) and (a_y != b_y):
            for j in range(a_y + 1, b_y):
                v_star[int(b_x//dx) - 1, j] = -v_star[int(b_x//dx), j] 

            for j in range(a_y+1, b_y):
                v_star[int(a_x//dx), j] = -v_star[int(a_x//dx)-1, j]

@njit
def pressao(u_star, v_star, p, N_x, N_y, dx, dy, dt, in_obs, obstacles, tol = 1e-3):
    erro = 10
    while erro > tol:
        r_max = 0
        for i in range(N_x):
            for j in range(N_y):
                if not in_obs[i, j]:       
                    if (i == 0 and j == 0):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (p[i+1, j] - p[i, j])/dx**2 - 
                            (p[i, j+1] - p[i, j])/dy**2)
                    
                    elif (i == 0 and j == N_y-1):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (p[i+1, j] - p[i, j])/dx**2 - 
                            (-p[i, j] + p[i, j-1])/dy**2)
                    
                    elif (i == N_x-1 and j == 0):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (-p[i, j] + p[i-1, j])/dx**2 - 
                            (p[i, j+1] - p[i, j])/dy**2)
                    
                    elif (i == N_x-1 and j == N_y-1):
                        valor_lambda = -(1/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (-p[i, j] + p[i-1, j])/dx**2 - 
                            (-p[i, j] + p[i, j-1])/dy**2)
                    
                    elif (i == 0 and j != 0 and j != N_y-1):
                        valor_lambda = -(1/dx**2 + 2/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (p[i+1, j] - p[i, j])/dx**2 - 
                            (p[i, j+1] - 2*p[i, j] + p[i, j-1])/dy**2)
                    
                    elif (i == N_x-1 and j != 0 and j != N_y-1):
                        valor_lambda = -(1/dx**2 + 2/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (-p[i, j] + p[i-1, j])/dx**2 - 
                            (p[i, j+1] - 2*p[i, j] + p[i, j-1])/dy**2)
                    
                    elif (i != 0 and i != N_x-1 and j == 0):
                        valor_lambda = -(2/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (p[i+1, j] - 2*p[i, j] + p[i-1, j])/dx**2 - 
                            (p[i, j+1] - p[i, j])/dy**2)
                    
                    elif (i != 0 and i != N_x-1 and j == N_y-1):
                        valor_lambda = -(2/dx**2 + 1/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (p[i+1, j] - 2*p[i, j] + p[i-1, j])/dx**2 - 
                            (-p[i, j] + p[i, j-1])/dy**2)
                    
                    else:
                        valor_lambda = -(2/dx**2 + 2/dy**2)
                        R = ((u_star[i+1, j] - u_star[i, j])/(dt*dx) +
                            (v_star[i, j+1] - v_star[i, j])/(dt*dy) - 
                            (p[i+1, j] - 2*p[i, j] + p[i-1, j])/dx**2 - 
                            (p[i, j+1] - 2*p[i, j] + p[i, j-1])/dy**2)
                    
                    R = R/valor_lambda
                    p[i, j] = p[i, j] + R

                    if np.abs(R) > r_max:
                        r_max = np.abs(R)
        erro = r_max
    for i in range(0, N_x):
        p[i, -1] = p[i, 0]
        p[i, N_y] = p[i, N_y - 1]
    
    for j in range(0, N_y):
        p[-1, j] = p[0, j]
        p[N_x, j] = p[N_x - 1, j]

    for a_x, a_y, b_x, b_y in obstacles:
        if (a_x != b_x) and (a_y != b_y):
            for i in range(a_x, b_x):
                p[i, int(a_y//dy)] = p[i, int(a_y//dy) - 1]
                p[i, int(b_y//dy) - 1] = p[i, int(b_y//dy)]

            for j in range(a_y, b_y):
                p[int(a_x//dx), j] = p[int(a_x//dx) - 1, j]
                p[int(b_x//dx) - 1, j] = p[int(b_x//dx), j]

    p[-1,-1] = p[0,0]
    p[-1, N_y] = p[0, N_y - 1]
    p[N_x, -1] = p[N_x - 1, 0]
    p[N_x, N_y] = p[N_x - 1, N_y - 1]


@njit
def velocidade_x(u, u_star, p, N_x, N_y, dx, dt, in_obs):
    for i in range(1, N_x):
        for j in range(-1, N_y+1):
            if not in_obs[i, j]: 
                u[i, j] = u_star[i, j]-dt*((p[i, j]-p[i-1, j])/dx)

@njit
def velocidade_y(v, v_star, p, N_x, N_y, dy, dt, in_obs):
    for i in range(-1 ,N_x + 1):
        for j in range(1,N_y):
            if not in_obs[i, j]: 
                v[i, j] = v_star[i, j] - dt*(p[i,j] - p[i, j - 1])/dy

@njit
def temperatura(u, v, theta, N_x, N_y, dx, dy, dt, theta_new, in_obs, obstacles):
    for i in range(0, N_x):
        for j in range(0, N_y):
            if not in_obs[i, j]: 
                theta_new[i, j] = theta[i, j]+dt*((theta[i+1, j]-2*theta[i, j]+theta[i-1, j])/(dx**2) +(theta[i, j+1]-2*theta[i, j]+theta[i, j-1])/(dy**2) -(0.25/dx)*(u[i+1, j]+u[i, j])*(theta[i+1, j]-theta[i-1, j]) -(0.25/dy)*(v[i, j+1]+v[i, j])*(theta[i, j+1]-theta[i, j-1]))
        
    #theta = np.copy(theta_new)
    for i in range(0, N_x):
        for j in range(0, N_y):
            theta[i, j] = theta_new[i, j]    
    
    for i in range(-1, N_x+1):
        theta[i, -1] = theta[i, 0]
        theta[i, N_y] = theta[i, N_y-1]
    
    for j in range(-1, N_y+1):
        theta[-1, j] = 2-theta[0, j]
        theta[N_x, j] = -theta[N_x-1, j]
        
    for a_x, a_y, b_x, b_y in obstacles:        
        if (a_x != b_x) and (a_y != b_y):
            for j in range(a_y, b_y):
                theta[int(b_x//dx) - 1, j] = theta[int(b_x//dx), j]
                theta[int(a_x//dx), j] = theta[int(a_x//dx) - 1, j]

            for i in range(a_x, b_x):
                theta[i, int(b_y//dy) - 1] = theta[i, int(b_y//dy)]
                theta[i, int(a_y//dy)] = theta[i, int(a_y//dy) - 1]
        #return theta
    
"""
u_star -> u, dH_x2
v_star -> v, dH_y2
p -> u_star, v_star
u -> u_star, p
v -> v_star, p
temperatura -> theta, u, v

dH_x2
|  u
|  |
v  v
u_star <-----------u
     ⊢---> p ----->⊢----> theta
v_star <---------- v
^ ^
| |
| v
dH_y2


dt < dx

dx < 1/sqrt(Re)

dt < (Re * dx**2)/4



Pr*dt*((u[i+1, j]-(2*u[i, j]) + u[i-1, j])/(dx**2) 

0.5*dt*Pr*Ram*theta[i,j]*dH_x2[i,j]

dt*Pr*Ra*(theta[i, j + 1] + theta[i, j])/2

Pr * dt * 4 /dx**2 < 1
dt < dx**2/(4*Pr)

0.5 * dt * Pr * Ram < 1
dt < 2/(Pr * Ram)

dt * Pr * Ra * 2 / 2 < 1
dt < 1/(Pr * Ra)



"""
