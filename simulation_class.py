from multiprocessing import Process 
from datetime import datetime
from functools import reduce
import eletric_field_wire
from typing import List
from numba import njit
from time import time
import fluid_class
import numpy as np
import argparse
import copy
import json
import os

path = os.getcwd()

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_all_data(data, Ra_0, Ram_0, Pr_0, N_x, N_y, L_x, L_y, wires, phi, chi, obstacles):
    os.chdir(path)
    
    if not os.path.exists('data'):
        raise AttributeError("Need the data directory.")

    dict_data = {
        float(vect[13]):
            {
                'N_x': vect[3],
                'N_y': vect[4],
                'L_x': L_x,
                'L_y': L_y,
                'obstacles': vect[16].tolist(),
                'wires': wires,
                'Ra': vect[0],
                'Ram': vect[1],
                'Pr': vect[2],
                'phi': phi,
                'Nu': nusselt(vect[8], vect[3], vect[4], vect[14], vect[15]),
                'u': vect[5].tolist(),
                'v': vect[6].tolist(),
                'p': vect[7].tolist(),
                'theta': vect[8].tolist(),
                'H_x': vect[9].tolist(),
                'H_y': vect[10].tolist(),
                'dH_x2': vect[11].tolist(),
                'dH_y2': vect[12].tolist(),
            }
            for vect in data
    }
    
    page_name = f"L_x-{L_x:1f}-L_y-{L_y:1f}"
    
    folder_path = os.path.join('data', page_name)
    
    create_path(folder_path)

    folder_name = f"N_x-{int(N_x)}-N_y-{int(N_y)}"
    
    folder_path = os.path.join(folder_path, folder_name)
    
    create_path(folder_path)
    
    folder_name = f"phi-{phi}"
    
    folder_path = os.path.join(folder_path, folder_name)
    
    create_path(folder_path)
    
    folder_name = f"chi-{chi}"
    
    folder_path = os.path.join(folder_path, folder_name)
    
    create_path(folder_path)
    
    folder_name = f"obstacles-{len(obstacles)}"
    
    folder_path = os.path.join(folder_path, folder_name)
    
    create_path(folder_path)
    
    folder_name = f"wires-{len(wires)}"
    
    folder_path = os.path.join(folder_path, folder_name)
    
    create_path(folder_path)
    
    os.chdir(folder_path)

    files_in_dir = os.listdir()

    idx = sum(1 for file in files_in_dir if file.startswith(f"Pr0_{int(Pr_0//1)}-Ra0_{int(Ra_0//1)}-Ram0_{int(Ram_0//1)}"))

    file_name = f"Pr0_{int(Pr_0//1)}-Ra0_{int(Ra_0//1)}-Ram0_{int(Ram_0//1)}_idx_{idx}.json"

    with open(file_name, 'w') as arquivo_json:
        json.dump(dict_data, arquivo_json)
    
@njit
def fluxo(u, v, N_x, N_y, dx, dy, psi, tol = 1e-1):
    lambida = -(2/(dx**2))-(2/(dy**2))
    erro = 10.0
    while erro > tol:
        r_max = 0.0
        for i in range(1, N_x):
            for j in range(1, N_y):
                r = -(v[i, j]-v[i-1, j])/dx + (u[i, j]-u[i, j-1])/dy-(psi[i+1, j]-(2*psi[i, j]) + psi[i-1, j])/(dx**2)-(psi[i, j+1]-(2*psi[i, j]) +psi[i, j-1])/(dy**2)
                r = r/lambida
                psi[i, j] = psi[i, j] + r
                if(np.abs(r) > r_max):
                    r_max = np.abs(r)
        erro = r_max

def nusselt(theta, N_x, N_y, dx, dy):
    nu_r = np.sum((theta[N_x, :-1] - theta[N_x-1, :-1]) * dy / dx)
    
    nu_l = np.sum((theta[0, : -2] - theta[1, : - 2]) * dy / dx)
    
    nu_up = np.sum((theta[:, N_y] - theta[:, N_y-1]) * dx / dy)

    nu_d = np.sum((theta[:, -1] - theta[:, 0]) * dx / dy)

    dict_nu = {
        'right': nu_r,
        'left': nu_l,
        'up': nu_up,
        'down': nu_d,
    }

    return dict_nu

def _langevin(gamma):
    return 1/np.tanh(gamma) - 1/gamma

def _G(gamma):
    langevin = _langevin(gamma)
    return gamma * langevin/ (4 + 2*langevin)

def expansao_termica():
    """
    beta
    """
    return 1

def difusividade_termica():
    """
    alpha
    """
    return 1

def condutividade_termica(phi, k_hat):
    """
    K
    """
    return  (1 + 3*phi*(k_hat - 1)/(k_hat - 2))

def massa_especifica(phi, rho_hat):
    return  (1 + phi * (rho_hat - 1))

def viscosidade_campo(phi, H, T):
    """
    nu
    """
    K_B = 1.38e-23
    m = 5.73e-8
    razao_energetica = m * H / K_B * T
    langevin = _langevin(razao_energetica)
    return  (1 + phi * 1.5 * (razao_energetica * langevin**2) / (razao_energetica - langevin))

def viscosidade(phi):
    return 1 + 2.5*phi

def susceptiblidade_magnetica(phi, chi):
    """
    Chi
    """
    return phi * chi

def rayleigh(phi, rho_hat, k_hat, H, T):
    return (massa_especifica(phi, rho_hat) * expansao_termica() / (difusividade_termica() * viscosidade(phi) * condutividade_termica(phi, k_hat)))

def rayleigh_mag(phi, rho_hat,  chi, H, T):
    return (susceptiblidade_magnetica(phi, chi) / (difusividade_termica() * viscosidade(phi)))

def prandtl(phi, rho_hat,H, T):
    return (viscosidade(phi)/(massa_especifica(phi, rho_hat) * difusividade_termica()))

def simulacao_laco(u, v, u_star, v_star, p, H_x, H_y,dH_x2, dH_y2, theta_new, theta, dx, dy, in_obs, Ra, Ram, Pr, obstacles, wires, phi, T_H, T_C, rho_hat, k_hat, chi, tf, dt, N_x, N_y, L_x, L_y, SAVE_DATA, SPT, POOL):
    data = []
    obstacles = np.array(obstacles)
    Ra_0 = Ra
    Ram_0 = Ram
    Pr_0 = Pr
    if phi != 0:
        """
        Se viscosidade depender do campo de temperatura, esse if tem que estar dentro do for para iterar, mas somente as partes que precisam dele
        """
        H = np.sqrt(np.mean(H_x)**2 + np.mean(H_y)**2)
        T = np.mean(theta)
        T = T_C + (T_H - T_C)*T + 273.15
        
        Ra = Ra * rayleigh(phi, rho_hat, k_hat, H, T)
        Ram = Ram * rayleigh_mag(phi, rho_hat, chi, H, T)
        Pr = Pr * prandtl(phi, rho_hat, H, T)
      
    for t in range(int(tf/dt)):

        fluid_class.velocidade_x_star(u, v, u_star, theta, Pr, Ram, N_x, N_y, dx, dy, dt, dH_x2, in_obs, obstacles)
        fluid_class.velocidade_y_star(u, v, v_star, theta, Pr, Ra, Ram, N_x, N_y, dx, dy, dt, dH_y2, in_obs, obstacles)
        fluid_class.temperatura(u, v, theta, N_x, N_y, dx, dy, dt, theta_new, in_obs, obstacles)
        fluid_class.pressao(u_star, v_star, p, N_x, N_y, dx, dy, dt, in_obs, obstacles)
        fluid_class.velocidade_x(u, u_star, p, N_x, N_y, dx, dt, in_obs)
        fluid_class.velocidade_y(v, v_star, p, N_x, N_y, dx, dt, in_obs)

        if SAVE_DATA and not t % SPT:
            data.append([
                Ra, Ram, Pr, N_x, N_y,
                copy.deepcopy(u), copy.deepcopy(v), copy.deepcopy(p),
                copy.deepcopy(theta), copy.deepcopy(H_x), copy.deepcopy(H_y),
                copy.deepcopy(dH_x2), copy.deepcopy(dH_y2), t*dt, dx, dy,
                copy.deepcopy(obstacles)
            ])
        if not t % 1e4:
            print(datetime.now())
            print(100*dt*t/tf)
    
    if SAVE_DATA:
        data.append([
        Ra, Ram, Pr, N_x, N_y,
        copy.deepcopy(u), copy.deepcopy(v), copy.deepcopy(p),
        copy.deepcopy(theta), copy.deepcopy(H_x), copy.deepcopy(H_y),
        copy.deepcopy(dH_x2), copy.deepcopy(dH_y2), t*dt, dx, dy,
        copy.deepcopy(obstacles)
        ])
        #print("Saving Data")
        save_all_data(data, Ra_0, Ram_0, Pr_0, N_x, N_y, L_x, L_y, wires, phi, chi, obstacles)

def executar_simulacao(N_x, N_y, L_x, L_y, dt, tf, Ra, Ram, Pr, phi, chi, k_hat, rho_hat, SAVE_DATA, SPT, obstacles, wires, MAGNETIC_FIELD, T_H, T_C, POOL):
    dx = L_x / N_x
    dy = L_y / N_y

    x = np.linspace(0.0, L_x, N_x+1)
    y = np.linspace(0.0, L_y, N_y+1)

    H_x = np.zeros((N_x + 1, N_y + 1))
    H_y = np.zeros((N_x + 1, N_y + 1))
    dH_x2 = np.zeros((N_x + 1, N_y + 1))
    dH_y2 = np.zeros((N_x + 1, N_y + 1))

    if MAGNETIC_FIELD:
        magnetic_field = []

        for ponto_w in wires:
            magnetic_field.append([eletric_field_wire.calculate_H_x(ponto_w, x, y), 
                                eletric_field_wire.calculate_H_y(ponto_w, x, y),
                                eletric_field_wire.calculate_grad_H2_x(ponto_w, x, y),
                                eletric_field_wire.calculate_grad_H2_x(ponto_w, x, y)])


        magnetic_field = [list(i) for i in zip(*magnetic_field)]

        create_magnetic_list = lambda idx: [*[H for H in magnetic_field[idx]], np.zeros((N_x + 1, N_y + 1))] 

        add_matrix = lambda vect: (reduce(lambda a, b: np.add(a, b), vect))

        H_x = add_matrix(create_magnetic_list(0))
        H_y = add_matrix(create_magnetic_list(1))
        dH_x2 = add_matrix(create_magnetic_list(2))
        dH_y2 = add_matrix(create_magnetic_list(3))
        
    u = np.zeros((N_x+1, N_y+2), float)
    v = np.zeros((N_x+2, N_y+1), float)

    p = np.zeros((N_x+2, N_y+2), float)

    theta = np.zeros((N_x+2, N_y+2), float)

    for j in range(0, N_x+2):
        theta[j, -1] = 2.0

    u_star = np.copy(u)
    v_star = np.copy(v) 

    theta_new = np.zeros((N_x+2, N_y+2), float)

    in_obs = np.zeros((N_x, N_y), float)

    for a_x, a_y, b_x, b_y in obstacles:
            for i in range(0, N_x):
                for j in range(0, N_y):
                        in_obs[i, j] = (((a_x != b_x) & (a_y != b_y)) & ((i*dx >= a_x) & (i*dx <= b_x) & (j*dy >= a_y) & (j*dy <= b_y))) or in_obs[i, j]

    simulacao_laco(u, v, u_star, v_star, p, H_x, H_y,dH_x2, dH_y2, theta_new, theta, dx, dy, in_obs, Ra, Ram, Pr, obstacles, wires, phi, T_H, T_C, rho_hat, k_hat, chi, tf, dt, N_x, N_y, L_x, L_y, SAVE_DATA, SPT, POOL)

def main():
    parser_dict = {
        '-ra': {
            'dest': 'Ra',
            'type': float,
            'help': 'Gravitation Rayleigh'
        },
        '-ram': {
            'dest': 'Ram',
            'type': float,
            'help': 'Magnetic Rayleigh'
        },
        '-pr': {
            'dest': 'Pr',
            'type': float,
            'help': 'Prandtl'
        },
        '-phi': {
            'dest': 'phi',
            'type': float,
            'help': 'Particle volume fraction'
        },
        '-chi': {
            'dest': 'chi',
            'type': float,
            'help': 'magnetic susceptibility'
        },
        '--k_hat': {
            'dest': 'k_hat',
            'type': float,
            'default': 553,
            'help': 'thermal conductivity ratio between the particle and the carrier ﬂuid'
        },
        '--rho_hat': {
            'dest': 'rho_hat',
            'type': float,
            'default': 1.13,
            'help': 'density ratio between the particle and the carrier ﬂuid'
        },
        '--th': {
            'dest': 'T_H',
            'type': float,
            'default': 35,
            'help': 'Maximum temperature in cavity'
        },
        '--tc': {
            'dest': 'T_C',
            'type': float,
            'default': 25,
            'help': 'Minimum temperature in cavity'
        },
        '--obstacles': {
            'dest': 'obstacles',
            'type': list,
            'default': '[[0, 0, 0, 0]]',
            'help': 'Obstacles in the fluid a_x, a_y, b_x, b_y in format of list of lists [[0.4, 0.4, 0.6, 0.6], [0.2, 0.2, 0.4, 0.4]]. Default: No obstacle'
        },
        '--wires': {
            'dest': 'wires',
            'type': list,
            'default': '[]',
            'help': 'Coils in the proximity of the fluid w_x, w_y in format of list of lists [[-0.1, -0.1], [1.1, 1.1]]. Default: No coils'
        },
        '-tf': {
            'dest': 'tf',
            'type': float,
            'help': 'Final time of the simulation'
        },
        '-dt': {
            'dest': 'dt',
            'type': float,
            'help': 'Incremention time of the simulation'
        },
        '-N': {
            'dest': 'N',
            'type': tuple,
            'help': 'Number of cells'
        },
        '-l': {
            'dest': 'l',
            'type': tuple,
            'help': 'cavity size'
        },
        '-save': {
            'dest': 'SAVE_DATA',
            'type': bool,
            'help': 'Save output in a json file'
        },
        '--percent_save': {
            'dest': 'PERCENT_SAVE',
            'type': float,
            'help': 'The percentage of data that will be saved',
            'default': 0
        },
        '--spt': {
            'dest': 'SPT',
            'type': int,
            'help': 'The number of timesteps before saving',
            'default': 1
        },
        '--mpool': {
            'dest': 'POOL',
            'type': bool,
            'help': 'A boolean value that indicates if there will be a multiprocess aproach',
            'default': False
        }
    }
    
    parser = argparse.ArgumentParser(description='Simulate the flow of a thermomagnetic fluid with particles in a cavity with internal obstacles and interference of the magnetic field generated by coils')
    
    for key, value in parser_dict.items():
        parser.add_argument(key, **value)
    
    args = parser.parse_args()
    
    args_dict = {value['dest']: getattr(args, value['dest']) for value in parser_dict.values()}
    
    args_dict['N'] = [int(i) for i in ''.join(list(args_dict['N'])).split(',')]
    args_dict['l'] = [int(i) for i in ''.join(list(args_dict['l'])).split(',')]
    
    args_dict['N_x'], args_dict['N_y'] = args_dict['N']
    args_dict['L_x'], args_dict['L_y'] = args_dict['l']     

    del args_dict['N']
    del args_dict['l']

    
    args_dict['wires'] = eval(''.join(args_dict['wires']))
    args_dict['obstacles'] = eval(''.join(args_dict['obstacles']))
    
    args_dict['MAGNETIC_FIELD'] = bool(args_dict['wires']) 
    
    if args_dict['SAVE_DATA'] and args_dict['PERCENT_SAVE'] != 0:
        args_dict['SPT'] = int((args_dict['PERCENT_SAVE'] * args_dict['tf'] / args_dict['dt']) // 1)
    
    del args_dict['PERCENT_SAVE']
    
    ini = time()
    executar_simulacao(**args_dict)    
    print(time() - ini)


if __name__ == '__main__':
    main()