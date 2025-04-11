# Thermo-Magnetic Fluid Simulation

This project simulates the behavior of a ferrofluid under temperature gradients and magnetic fields using a finite difference approach. For proper visualization of all mathematical equations, please refer to:

[Equations Presentation](https://docs.google.com/presentation/d/1Rpxdb2kU_O7R7PYwr2p4ZkY5GR3cGFNFKqTIuPzGUzM/edit?usp=sharing)

## Key Physical Models

### Governing Equations
The simulation solves three fundamental conservation laws:
1. Mass conservation (continuity equation)
2. Momentum conservation (Navier-Stokes with Boussinesq approximation)
3. Energy conservation (heat transfer equation)

### Dimensionless Parameters
The flow is characterized by three dimensionless numbers:
1. Rayleigh Number (Ra) - buoyancy vs viscous forces
2. Magnetic Rayleigh Number (Ram) - magnetic vs viscous forces  
3. Prandtl Number (Pr) - momentum vs thermal diffusivity

### Magnetic Fluid Properties
The ferrofluid is modeled as a colloidal suspension with:
- Temperature-dependent viscosity
- Magnetic-field dependent properties
- Particle-volume-fraction dependent thermal conductivity
- Modified density and susceptibility

## Numerical Implementation

### Solution Method
1. Finite difference spatial discretization
2. Projection method for pressure-velocity coupling
3. Explicit time integration for temperature
4. Magnetic field calculation from current sources

### Key Features
- Supports arbitrary cavity geometries
- Handles internal obstacles
- Models multiple magnetic field sources
- Tracks heat transfer coefficients

## Usage

Basic command format:
```bash
python main.py -ra <Rayleigh> -ram <MagneticRayleigh> -pr <Prandtl> \
              -phi <volume_fraction> -chi <susceptibility> \
              -tf <final_time> -dt <time_step> \
              -N <grid_x>,<grid_y> -l <length_x>,<length_y>
```

## Output Data
The simulation generates:

- Velocity fields (u, v components)

- Pressure distribution (p)

- Temperature field (θ)

- Magnetic field components (Hx, Hy)

- Nusselt number calculations at all boundaries

## Visualization
For complete mathematical formulation and sample results, see:
Full Equations and Results Presentation

## Author
Pedro Trajano Ferreira

pedro.trajano.ferreira@gmail.com

11/04/25
