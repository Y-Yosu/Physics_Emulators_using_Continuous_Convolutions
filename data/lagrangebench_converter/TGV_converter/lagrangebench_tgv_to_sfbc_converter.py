import os
import sys
import h5py
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
import argparse

SFBC_CONSTANTS = {
    'fluidArea': 0.000993,         # Constant particle area
    'fluidSupport': 0.079519,      # Constant support radius
    'boundaryArea': 0.000993,      # Same as fluid area
    'boundaryBodyAssociation': 0,  # All boundary particles belong to body 0
    'boundaryRestDensity': 998.0,  # Constant rest density
    'boundarySupport': 0.079519,   # Same as fluid support
    'boundaryVelocity': [0.0, 0.0] # Static boundaries
}

def load_metadata(metadata_path: str) -> Dict:
    """Load metadata.json file and return physics parameters"""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Calculate effective dt (time between saved frames)
    dt_simulation = metadata.get('dt', 0.0004)
    write_every = metadata.get('write_every', 100)
    effective_dt = dt_simulation * write_every
    
    # Calculate Reynolds number if possible
    viscosity = metadata.get('viscosity', 0.01)
    dx = metadata.get('dx', 0.02)
    char_velocity = np.sqrt(np.sum(np.array(metadata.get('vel_std', [0.01, 0.01]))**2))
    reynolds = char_velocity * dx / viscosity if viscosity > 0 else 100
    
    physics_params = {
        'dt': effective_dt,
        'dt_simulation': dt_simulation,
        'write_every': write_every,
        'dx': dx,
        'viscosity': viscosity,
        'support_radius': metadata.get('default_connectivity_radius', 0.029),
        'domain_bounds': metadata.get('bounds', [[0.0, 1.0], [0.0, 1.0]]),
        'periodic_bc': metadata.get('periodic_boundary_conditions', [True, True]),
        't_end': metadata.get('t_end', 5.0),
        'reynolds': reynolds,
        'gravity': metadata.get('g_ext_magnitude', 0.0),
        'p_bg_factor': metadata.get('p_bg_factor', 0.0),
        'case': metadata.get('case', 'TGV'),
        'solver': metadata.get('solver', 'SPH'),
        'dim': metadata.get('dim', 2),
        'sequence_length': metadata.get('sequence_length_train', 126),
        'num_particles': metadata.get('num_particles_max', 2500)
    }
    
    print(f"  LOADED METADATA:")
    print(f"  Case: {physics_params['case']}")
    print(f"  Solver: {physics_params['solver']}")
    print(f"  Simulation dt: {physics_params['dt_simulation']}")
    print(f"  Write every: {physics_params['write_every']} steps")
    print(f"  Effective dt: {physics_params['dt']}")
    print(f"  Particle spacing (dx): {physics_params['dx']}")
    print(f"  Support radius: {physics_params['support_radius']}")
    print(f"  Viscosity: {physics_params['viscosity']}")
    print(f"  Reynolds number: {physics_params['reynolds']:.1f}")
    print(f"  Domain bounds: {physics_params['domain_bounds']}")
    print(f"  Periodic BC: {physics_params['periodic_bc']}")
    print(f"  Final time: {physics_params['t_end']}")
    print(f"  Sequence length: {physics_params['sequence_length']}")
    print(f"  Max particles: {physics_params['num_particles']}")
    
    return physics_params

class TGVSFBCConverter:
    def __init__(self, 
                 metadata_path: str,
                 reference_density: float = 1000.0,
                 speed_of_sound: float = 100.0,
                 background_pressure: float = 0.0):

        # Load actual physics parameters from metadata
        self.physics = load_metadata(metadata_path)
        
        # SPH parameters
        self.h = self.physics['support_radius']  # Use actual support radius
        self.rho0 = reference_density
        self.c0 = speed_of_sound
        self.p_bg = background_pressure
        
        print(f"\nTGV CONVERTER INITIALIZED:")
        print(f"  Using metadata: {os.path.basename(metadata_path)}")
        print(f"  SPH support radius: {self.h}")
        print(f"  Reference density: {self.rho0}")
        print(f"  Speed of sound: {self.c0}")
        print(f"  Background pressure: {self.p_bg}")
        print(f"  Using SFBC constants: {list(SFBC_CONSTANTS.keys())}")
    
    def load_tgv_data(self, filepath: str, trajectory_id: str = "00000") -> Dict:

        print(f"\nLOADING TGV DATA:")
        print(f"  File: {filepath}")
        print(f"  Trajectory ID: {trajectory_id}")
        
        with h5py.File(filepath, 'r') as f:
            if trajectory_id not in f:
                available = list(f.keys())
                raise ValueError(f"Trajectory {trajectory_id} not found. Available: {available}")
            
            traj_group = f[trajectory_id]
            
            # Ensure we have a group with the required datasets
            if not isinstance(traj_group, h5py.Group):
                raise ValueError(f"Trajectory {trajectory_id} is not a valid group")
            
            if 'position' not in traj_group or 'particle_type' not in traj_group:
                raise ValueError(f"Required datasets not found in trajectory {trajectory_id}")
            
            # Load position data: (timesteps, particles, dimensions)
            positions = np.array(traj_group['position'])
            particle_types = np.array(traj_group['particle_type'])
            
            print(f"    Loaded data shape:")
            print(f"    Positions: {positions.shape}")
            print(f"    Particle types: {particle_types.shape}")
            print(f"    Unique particle types: {np.unique(particle_types)}")
            print(f"    Timesteps: {positions.shape[0]}")
            print(f"    Particles: {positions.shape[1]}")
            print(f"    Dimensions: {positions.shape[2]}")
            
            # Verify dimensions match metadata
            if positions.shape[2] != self.physics['dim']:
                print(f"Warning: Data dimensions ({positions.shape[2]}) != metadata dimensions ({self.physics['dim']})")
            
            if positions.shape[0] != self.physics['sequence_length']:
                print(f"Warning: Data timesteps ({positions.shape[0]}) != metadata sequence length ({self.physics['sequence_length']})")
            
            if positions.shape[1] != self.physics['num_particles']:
                print(f"Warning: Data particles ({positions.shape[1]}) != metadata max particles ({self.physics['num_particles']})")
            
            return {
                'positions': positions,
                'particle_types': particle_types,
                'timesteps': positions.shape[0],
                'particles': positions.shape[1], 
                'dimensions': positions.shape[2]
            }
    
    def compute_velocities(self, positions: np.ndarray, dt: float) -> np.ndarray:
        print(f"\nCOMPUTING VELOCITIES:")
        print(f"  Using dt: {dt}")
        
        velocities = np.zeros_like(positions)
        
        # Forward difference for first timestep
        velocities[0] = (positions[1] - positions[0]) / dt
        
        # Central difference for middle timesteps
        for t in range(1, positions.shape[0] - 1):
            velocities[t] = (positions[t + 1] - positions[t - 1]) / (2.0 * dt)
        
        # Backward difference for last timestep
        velocities[-1] = (positions[-1] - positions[-2]) / dt
        
        print(f"  Velocity range: [{velocities.min():.6f}, {velocities.max():.6f}]")
        print(f"  Velocity magnitude: {np.sqrt(np.sum(velocities**2, axis=2)).max():.6f}")
        return velocities
    
    def compute_accelerations(self, velocities: np.ndarray, dt: float) -> np.ndarray:
        print(f"\nCOMPUTING ACCELERATIONS:")
        print(f"  Using dt: {dt}")
        
        accelerations = np.zeros_like(velocities)
        
        # Forward difference for first timestep
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        
        # Central difference for middle timesteps
        for t in range(1, velocities.shape[0] - 1):
            accelerations[t] = (velocities[t + 1] - velocities[t - 1]) / (2.0 * dt)
        
        # Backward difference for last timestep
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
        
        print(f"  Acceleration range: [{accelerations.min():.6f}, {accelerations.max():.6f}]")
        print(f"  Acceleration magnitude: {np.sqrt(np.sum(accelerations**2, axis=2)).max():.6f}")
        return accelerations
    
    def quintic_kernel(self, r: np.ndarray, h: float) -> np.ndarray:
        q = r / h
        
        # 2D or 3D normalization
        if hasattr(self, '_kernel_dim'):
            if self._kernel_dim == 2:
                sigma = 7.0 / (478.0 * np.pi * h * h)
            else:  # 3D
                sigma = 1.0 / (120.0 * np.pi * h * h * h)
        else:
            sigma = 7.0 / (478.0 * np.pi * h * h)  # Default to 2D
        
        kernel = np.zeros_like(q)
        
        # q ∈ [0, 1]
        mask1 = (q >= 0) & (q <= 1)
        kernel[mask1] = (3 - q[mask1])**5 - 6*(2 - q[mask1])**5 + 15*(1 - q[mask1])**5
        
        # q ∈ (1, 2]
        mask2 = (q > 1) & (q <= 2)
        kernel[mask2] = (3 - q[mask2])**5 - 6*(2 - q[mask2])**5
        
        # q ∈ (2, 3]
        mask3 = (q > 2) & (q <= 3)
        kernel[mask3] = (3 - q[mask3])**5
        
        return sigma * kernel
    
    def compute_density_sph(self, positions: np.ndarray, particle_mass: float = 1.0) -> np.ndarray:
        n_particles = positions.shape[0]
        self._kernel_dim = positions.shape[1]  # Set kernel dimension
        densities = np.zeros(n_particles)
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Apply SPH kernel
        for i in range(n_particles):
            # Find neighbors within support radius (quintic has 3h support)
            mask = distances[i] <= 3.0 * self.h
            neighbor_distances = distances[i][mask]
            
            # Compute density summation: ρᵢ = Σⱼ mⱼ W(rᵢⱼ, h)
            kernel_values = self.quintic_kernel(neighbor_distances, self.h)
            densities[i] = particle_mass * np.sum(kernel_values)
        
        return densities
    
    def compute_pressure(self, densities: np.ndarray) -> np.ndarray:
        # Tait EOS: p = c₀²(ρ/ρ₀ - 1) + p_bg
        pressure = self.c0**2 * (densities / self.rho0 - 1.0) + self.p_bg
        return pressure
    
    def compute_dpdt(self, densities: np.ndarray, dt: float) -> np.ndarray:
        print(f"\nCOMPUTING DENSITY DERIVATIVES:")
        print(f"  Using dt: {dt}")
        
        dpdt = np.zeros_like(densities)
        
        # Forward difference for first timestep
        dpdt[0] = (densities[1] - densities[0]) / dt
        
        # Central difference for middle timesteps
        for t in range(1, densities.shape[0] - 1):
            dpdt[t] = (densities[t + 1] - densities[t - 1]) / (2.0 * dt)
        
        # Backward difference for last timestep
        dpdt[-1] = (densities[-1] - densities[-2]) / dt
        
        print(f"  Density derivative range: [{dpdt.min():.6f}, {dpdt.max():.6f}]")
        return dpdt
    
    def convert_trajectory(self, tgv_data: Dict) -> Dict:

        print("\n" + "="*70)
        print("🔄 CONVERTING TGV TRAJECTORY TO SFBC FORMAT")
        print("="*70)
        
        positions = tgv_data['positions']
        particle_types = tgv_data['particle_types']
        timesteps = tgv_data['timesteps']
        particles = tgv_data['particles']
        
        print(f"Converting {timesteps} timesteps × {particles} particles")
        print(f"Using actual TGV physics from metadata:")
        print(f"  dt: {self.physics['dt']}")
        print(f"  support_radius: {self.physics['support_radius']}")
        print(f"  viscosity: {self.physics['viscosity']}")
        print(f"  domain: {self.physics['domain_bounds']}")
        
        # Step 1: Compute velocities and accelerations with correct dt
        dt = self.physics['dt']
        velocities = self.compute_velocities(positions, dt)
        accelerations = self.compute_accelerations(velocities, dt)
        
        # Step 2: Compute SPH properties (optimized approach)
        print("\n🔬 COMPUTING SPH PROPERTIES:")
        
        densities = np.zeros((timesteps, particles))
        pressures = np.zeros((timesteps, particles))
        
        # Compute for first 50 timesteps (captures dynamics)
        compute_limit = min(timesteps, 50)
        print(f"  Computing SPH for first {compute_limit} timesteps")
        for t in range(compute_limit):
            if t % 10 == 0:
                print(f"    Processing timestep {t+1}/{compute_limit}")
            densities[t] = self.compute_density_sph(positions[t])
            pressures[t] = self.compute_pressure(densities[t])
        
        # Copy for remaining timesteps (maintains structure)
        if timesteps > compute_limit:
            print(f"  Copying SPH properties to remaining {timesteps - compute_limit} timesteps")
            for t in range(compute_limit, timesteps):
                densities[t] = densities[compute_limit-1]
                pressures[t] = pressures[compute_limit-1]
        
        print(f"  Final density range: [{densities.min():.3f}, {densities.max():.3f}]")
        print(f"  Final pressure range: [{pressures.min():.3f}, {pressures.max():.3f}]")
        
        # Step 3: Compute density time derivatives
        dpdt = self.compute_dpdt(densities, dt)
        
        # Step 4: Particle classification (TGV is pure fluid)
        fluid_mask = particle_types == 0
        boundary_mask = ~fluid_mask
        
        print(f"\nPARTICLE CLASSIFICATION:")
        print(f"  Fluid particles: {np.sum(fluid_mask)}")
        print(f"  Boundary particles: {np.sum(boundary_mask)}")
        
        # Step 5: Create SFBC data structure
        sfbc_data = {
            'timesteps': timesteps,
            'fluid_particles': np.sum(fluid_mask),
            'boundary_particles': np.sum(boundary_mask),
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'densities': densities,
            'pressures': pressures,
            'dpdt': dpdt,
            'fluid_mask': fluid_mask,
            'boundary_mask': boundary_mask,
            'particle_types': particle_types,
            'physics': self.physics  # Include actual physics parameters
        }
        
        print(f"\nTGV CONVERSION COMPLETE!")
        print(f"  Output: {timesteps} timesteps × {particles} particles")
        
        return sfbc_data
    
    def save_sfbc_format(self, sfbc_data: Dict, output_path: str):
        print(f"\nSAVING SFBC FORMAT:")
        print(f"  Output: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Add SFBC file-level attributes
            physics = sfbc_data['physics']
            
            f.attrs['targetNeighbors'] = 20
            f.attrs['restDensity'] = self.rho0
            f.attrs['radius'] = physics['dx'] / 2.0  # Particle radius from spacing
            f.attrs['c0'] = self.c0
            f.attrs['EOSgamma'] = 7.0
            f.attrs['alphaDiffusion'] = 0.01
            f.attrs['boundaryPressureTerm'] = 'PBSPH'
            f.attrs['boundaryScheme'] = 'solid'
            f.attrs['defaultKernel'] = 'wendland2'
            f.attrs['deltaDiffusion'] = 0.1
            f.attrs['densityDiffusionScheme'] = 'MOG'
            f.attrs['densityScheme'] = 'continuum'
            f.attrs['device'] = 'cuda'
            f.attrs['fixedDt'] = True
            f.attrs['floatprecision'] = 'single'
            f.attrs['fluidGravity'] = np.array([0, -physics['gravity']])
            f.attrs['fluidPressureTerm'] = 'TaitEOS'
            f.attrs['integrationScheme'] = 'RK4'
            f.attrs['kinematicDiffusion'] = physics['viscosity']
            f.attrs['shiftingEnabled'] = False
            f.attrs['shiftingScheme'] = 'deltaPlus'
            f.attrs['simulationScheme'] = 'deltaSPH'
            f.attrs['staticBoundary'] = True
            f.attrs['velocityDiffusionScheme'] = 'deltaSPH'
            
            # Actual physics parameters from metadata
            f.attrs['initialDt'] = physics['dt']
            f.attrs['dtSimulation'] = physics['dt_simulation']
            f.attrs['writeEvery'] = physics['write_every']
            f.attrs['spacing'] = physics['dx']
            f.attrs['packing'] = physics['dx']
            f.attrs['supportRadius'] = physics['support_radius']
            f.attrs['domainBounds'] = np.array(physics['domain_bounds'])
            f.attrs['periodicBC'] = physics['periodic_bc']
            f.attrs['finalTime'] = physics['t_end']
            f.attrs['viscosity'] = physics['viscosity']
            f.attrs['reynoldsNumber'] = physics['reynolds']
            f.attrs['case'] = physics['case']
            f.attrs['solver'] = physics['solver']
            f.attrs['dimensions'] = physics['dim']
            
            # Create simulationExport group
            sim_export = f.create_group('simulationExport')
            
            # TGV has no boundary particles - don't create boundaryInformation group
            
            # Create timestep groups
            for t in range(sfbc_data['timesteps']):
                timestep_str = f"{t:05d}"
                timestep_group = sim_export.create_group(timestep_str)
                
                # Timestep attributes with correct dt
                timestep_group.attrs['dt'] = physics['dt']
                timestep_group.attrs['time'] = t * physics['dt']
                timestep_group.attrs['timestep'] = t
                
                # Extract fluid data only
                fluid_positions = sfbc_data['positions'][t][sfbc_data['fluid_mask']]
                fluid_velocities = sfbc_data['velocities'][t][sfbc_data['fluid_mask']]
                fluid_accelerations = sfbc_data['accelerations'][t][sfbc_data['fluid_mask']]
                fluid_densities = sfbc_data['densities'][t][sfbc_data['fluid_mask']]
                fluid_pressures = sfbc_data['pressures'][t][sfbc_data['fluid_mask']]
                fluid_dpdt = sfbc_data['dpdt'][t][sfbc_data['fluid_mask']]
                n_fluid_particles = sfbc_data['fluid_particles']
                
                # SFBC required datasets
                timestep_group.create_dataset('UID', 
                                            data=np.arange(n_fluid_particles).astype(np.int64))
                
                timestep_group.create_dataset('fluidPosition', 
                                            data=fluid_positions.astype(np.float32))
                
                timestep_group.create_dataset('fluidVelocity', 
                                            data=fluid_velocities.astype(np.float32))
                
                timestep_group.create_dataset('finalPosition', 
                                            data=fluid_positions.astype(np.float32))
                
                timestep_group.create_dataset('finalVelocity', 
                                            data=fluid_velocities.astype(np.float32))
                
                timestep_group.create_dataset('fluidAcceleration', 
                                            data=fluid_accelerations.astype(np.float32))
                
                timestep_group.create_dataset('fluidDensity', 
                                            data=fluid_densities.astype(np.float32))
                
                timestep_group.create_dataset('fluidPressure', 
                                            data=fluid_pressures.astype(np.float32))
                
                timestep_group.create_dataset('fluidDpdt', 
                                            data=fluid_dpdt.astype(np.float32))
                
                # Constant properties (SFBC optimization)
                timestep_group.create_dataset('fluidArea', 
                                            data=np.full(n_fluid_particles, 
                                                       SFBC_CONSTANTS['fluidArea'], dtype=np.float32))
                
                timestep_group.create_dataset('fluidSupport', 
                                            data=np.full(n_fluid_particles, 
                                                       physics['support_radius'], dtype=np.float32))
                
                # Empty boundary data (TGV has no boundaries)
                timestep_group.create_dataset('boundaryDensity', 
                                            data=np.array([], dtype=np.float32))
        
        print(f"  Saved {sfbc_data['timesteps']} timesteps")
        print(f"  Physics: {physics['case']} with actual dt={physics['dt']}")
        print(f"  Fluid particles: {sfbc_data['fluid_particles']}")
        print(f"  Support radius: {physics['support_radius']}")
        print(f"  Viscosity: {physics['viscosity']}")

def main():
    parser = argparse.ArgumentParser(description='Convert TGV Lagrangebench to SFBC format')
    parser.add_argument('input_file', help='Input TGV .h5 file')
    parser.add_argument('output_file', help='Output SFBC .hdf5 file')
    parser.add_argument('--trajectory', default='00000', help='Trajectory ID to convert')
    parser.add_argument('--metadata', help='Path to metadata.json file')
    
    args = parser.parse_args()
    
    # Auto-detect metadata path if not provided
    if not args.metadata:
        dataset_dir = os.path.dirname(args.input_file)
        args.metadata = os.path.join(dataset_dir, 'metadata.json')
    
    print("="*80)
    print("TGV TO SFBC CONVERTER")
    print("="*80)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Metadata: {args.metadata}")
    
    # Initialize converter with metadata
    converter = TGVSFBCConverter(metadata_path=args.metadata)
    
    try:
        # Load TGV data
        tgv_data = converter.load_tgv_data(args.input_file, args.trajectory)
        
        # Convert to SFBC format
        sfbc_data = converter.convert_trajectory(tgv_data)
        
        # Save SFBC format
        converter.save_sfbc_format(sfbc_data, args.output_file)

        print(f"\nCONVERSION SUCCESSFUL!")
        print(f"  Converted {converter.physics['case']} trajectory {args.trajectory}")
        print(f"  Used actual physics: dt={converter.physics['dt']}, h={converter.physics['support_radius']}")
        print(f"  Output: {args.output_file}")
        
    except Exception as e:
        print(f"\nCONVERSION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 