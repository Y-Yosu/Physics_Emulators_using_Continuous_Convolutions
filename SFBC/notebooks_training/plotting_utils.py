import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Optional, Tuple
from BasisConvolution.util.augment import loadAugmentedFrame
from BasisConvolution.util.network import runInference

import sphMath.plotting as sp
import sphMath.util as su

# Following the reference implementation pattern from dataLoader-main
# Instead of custom wrapper classes, use proper dataclass states with kinds attribute
from dataclasses import dataclass

@dataclass(slots=True)
class SimpleParticleState:
    """Minimal SPH state for plotting - matches the reference dataLoader implementation pattern"""
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor
    velocities: torch.Tensor  # Added for completeness
    kinds: torch.Tensor # 0 for fluid, 1 for boundary, 2 for ghost

def convertSFBCStateToParticleState(currentState, gt_density=None):
    """Convert SFBC-style separated fluid/boundary state to combined particle state"""
    fluid_positions = currentState['fluid']['positions']
    fluid_supports = currentState['fluid']['supports']
    fluid_masses = currentState['fluid']['masses']
    fluid_densities = currentState['fluid']['densities']
    fluid_velocities = currentState['fluid']['velocities']
    fluid_kinds = torch.zeros(fluid_positions.shape[0], dtype=torch.long, device=fluid_positions.device)

    # Check if boundary particles exist
    if 'boundary' in currentState and currentState['boundary'] is not None and currentState['boundary']['positions'].shape[0] > 0:
        boundary_positions = currentState['boundary']['positions']
        boundary_supports = currentState['boundary']['supports']
        boundary_masses = currentState['boundary']['masses'] 
        boundary_densities = currentState['boundary']['densities']
        boundary_velocities = currentState['boundary']['velocities']
        boundary_kinds = torch.ones(boundary_positions.shape[0], dtype=torch.long, device=boundary_positions.device)

        # Combine fluid and boundary
        all_positions = torch.cat([fluid_positions, boundary_positions], dim=0)
        all_supports = torch.cat([fluid_supports, boundary_supports], dim=0)
        all_masses = torch.cat([fluid_masses, boundary_masses], dim=0)
        all_densities = torch.cat([fluid_densities, boundary_densities], dim=0)
        all_velocities = torch.cat([fluid_velocities, boundary_velocities], dim=0)
        all_kinds = torch.cat([fluid_kinds, boundary_kinds], dim=0)
    else:
        # Only fluid particles
        all_positions = fluid_positions
        all_supports = fluid_supports
        all_masses = fluid_masses
        all_densities = fluid_densities
        all_velocities = fluid_velocities
        all_kinds = fluid_kinds

    return SimpleParticleState(
        positions=all_positions,
        supports=all_supports,
        masses=all_masses,
        densities=all_densities,
        velocities=all_velocities,
        kinds=all_kinds
    )

def compute_density_gradient_velocity(positions: np.ndarray, densities: np.ndarray, 
                                     scale_factor: float = 0.2) -> np.ndarray:
    """
    Compute velocity field from density gradients

    Args:
        positions: Particle positions (N, 2)
        densities: Density values (N,)
        scale_factor: Scaling factor for velocity magnitude

    Returns:
        Velocity vectors (N, 2)
    """
    n_particles = len(positions)
    velocities = np.zeros((n_particles, 2))

    for i in range(n_particles):
        distances = np.linalg.norm(positions - positions[i], axis=1)
        neighbor_mask = distances < 0.08

        if np.sum(neighbor_mask) > 4:
            neighbor_pos = positions[neighbor_mask]
            neighbor_densities = densities[neighbor_mask]

            try:
                relative_pos = neighbor_pos - positions[i]
                density_diff = neighbor_densities - densities[i]

                if len(relative_pos) >= 3:
                    gradient, _, _, _ = np.linalg.lstsq(relative_pos, density_diff, rcond=None)
                    velocities[i] = -gradient * scale_factor
            except:
                pass

    return velocities

def visualize_density_comparison(models_dict: Dict, ds, hyperParameterDict: Dict,
                                trajectory_index: int,
                                frames_per_trajectory: int = 128,
                                frames: List[int] = [0, 30, 50, 100, 126],
                                figsize: Tuple[int, int] = (16, 15)) -> None:
    """
    Visualize density comparison for a specific trajectory

    Args:
        models_dict: Dictionary of trained models
        ds: Dataset
        hyperParameterDict: Hyperparameter dictionary
        trajectory_index: Which trajectory to plot (0-based index)
        frames: Frame numbers within the trajectory to plot (0-126 for SFBC_TGV)
        figsize: Figure size
    """    
    # Calculate actual dataset indices
    trajectory_start_idx = trajectory_index * frames_per_trajectory
    actual_frames = [trajectory_start_idx + f for f in frames]

    # Validate indices
    max_valid_idx = len(ds) - 1
    invalid_frames = [f for f in actual_frames if f > max_valid_idx]
    if invalid_frames:
        print(f"Warning: Invalid frame indices {invalid_frames}, max valid: {max_valid_idx}")
        actual_frames = [f for f in actual_frames if f <= max_valid_idx]
        frames = frames[:len(actual_frames)]

    n_frames = len(frames)
    fig, axes = plt.subplots(n_frames, 4, figsize=figsize)

    if n_frames == 1:
        axes = axes.reshape(1, -1)

    for model in models_dict.values():
        model.eval()

    for row, (frame_offset, frame_idx) in enumerate(zip(frames, actual_frames)):
        try:
            config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
                frame_idx, ds, hyperParameterDict)

            gt = trajectoryStates[0]['fluid']['target']
            positions = currentState['fluid']['positions'].cpu().numpy()

            if isinstance(gt, torch.Tensor):
                gt = gt.cpu().numpy()

            gt_density = gt[:, 0] if gt.shape[1] > 0 else gt.flatten()

            # Ground truth
            sc_gt = axes[row, 0].scatter(positions[:, 0], positions[:, 1], 
                                         c=gt_density, cmap='viridis', s=0.5, alpha=0.8)
            axes[row, 0].set_title(f'Ground Truth (traj={trajectory_index}, f={frame_offset})')
            axes[row, 0].set_aspect('equal')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            # Add colorbar for ground truth
            divider_gt = make_axes_locatable(axes[row, 0])
            cax_gt = divider_gt.append_axes("right", size="5%", pad=0.1)
            cbar_gt = fig.colorbar(sc_gt, cax=cax_gt)
            cbar_gt.set_label("Density")

            # Model predictions
            model_names = ['linear', 'ffourier', 'chebyshev']
            for col, model_name in enumerate(model_names, 1):
                if model_name in models_dict:
                    with torch.no_grad():
                        prediction = runInference(currentState, config, models_dict[model_name], verbose=False)

                    if isinstance(prediction, torch.Tensor):
                        prediction = prediction.cpu().numpy()

                    pred_density = prediction[:, 0] if prediction.shape[1] > 0 else prediction.flatten()

                    sc_pred = axes[row, col].scatter(positions[:, 0], positions[:, 1], 
                                                     c=pred_density, cmap='viridis', s=0.5, alpha=0.8)
                    axes[row, col].set_title(f'{model_name} (traj={trajectory_index}, f={frame_offset})')
                    axes[row, col].set_aspect('equal')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])

                    # Add colorbar for each model
                    divider_pred = make_axes_locatable(axes[row, col])
                    cax_pred = divider_pred.append_axes("right", size="5%", pad=0.1)
                    cbar_pred = fig.colorbar(sc_pred, cax=cax_pred)
                    cbar_pred.set_label("Density")
                else:
                    axes[row, col].text(0.5, 0.5, f'{model_name}\nNot Available', 
                                        ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])

        except Exception as e:
            print(f"Error processing trajectory {trajectory_index}, frame {frame_offset}: {e}")
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f'Error\nTraj {trajectory_index} Frame {frame_offset}', 
                                    ha='center', va='center', transform=axes[row, col].transAxes)

    plt.tight_layout()
    plt.show()

def visualize_network_derived_velocities(models_dict: Dict, ds, hyperParameterDict: Dict,
                                        trajectory_index: int,
                                        frames_per_trajectory: int = 128,
                                        frames: List[int] = [0, 30, 50, 100, 126],
                                        figsize: Tuple[int, int] = (20, 12)) -> None:
    """
    Visualize velocity fields derived from network density predictions for a specific trajectory

    Args:
        models_dict: Dictionary of trained models
        ds: Dataset
        hyperParameterDict: Hyperparameter dictionary
        trajectory_index: Which trajectory to plot (0-based index)
        frames: Frame numbers within the trajectory to plot (0-126 for SFBC_TGV)
        figsize: Figure size
    """
    # Calculate actual dataset indices
    trajectory_start_idx = trajectory_index * frames_per_trajectory
    actual_frames = [trajectory_start_idx + f for f in frames]

    # Validate indices
    max_valid_idx = len(ds) - 1
    invalid_frames = [f for f in actual_frames if f > max_valid_idx]
    if invalid_frames:
        print(f"Warning: Invalid frame indices {invalid_frames}, max valid: {max_valid_idx}")
        actual_frames = [f for f in actual_frames if f <= max_valid_idx]
        frames = frames[:len(actual_frames)]

    n_frames = len(frames)
    fig, axes = plt.subplots(n_frames, 4, figsize=figsize)

    if n_frames == 1:
        axes = axes.reshape(1, -1)

    for model in models_dict.values():
        model.eval()

    def plot_velocity_field(ax, positions, velocities, title, alpha=0.8, subsample=6):
        pos_sub = positions[::subsample]
        vel_sub = velocities[::subsample]

        ax.scatter(pos_sub[:, 0], pos_sub[:, 1], c='lightgray', s=8, alpha=0.4)

        vel_magnitude = np.sqrt(vel_sub[:, 0]**2 + vel_sub[:, 1]**2)

        quiver = None
        if vel_magnitude.max() > 0:
            max_vel = np.percentile(vel_magnitude, 90)
            scale_factor = 0.08 / max_vel if max_vel > 0 else 0.08
            vel_scaled = vel_sub * scale_factor

            quiver = ax.quiver(pos_sub[:, 0], pos_sub[:, 1], 
                               vel_scaled[:, 0], vel_scaled[:, 1],
                               vel_magnitude, 
                               scale=1, scale_units='xy', angles='xy',
                               cmap='plasma', alpha=alpha, width=0.005,
                               headwidth=3, headlength=4)

        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.6, 0.6)  # Updated for SFBC_TGV domain
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')

        return quiver

    for row, (frame_offset, frame_idx) in enumerate(zip(frames, actual_frames)):
        config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
            frame_idx, ds, hyperParameterDict)

        positions = currentState['fluid']['positions'].cpu().numpy()
        gt_velocities = currentState['fluid']['velocities'].cpu().numpy()

        quiver_gt = plot_velocity_field(axes[row, 0], positions, gt_velocities, 
                                        f'Ground Truth (traj={trajectory_index}, f={frame_offset})')

        # Add colorbar for ground truth velocities
        if quiver_gt is not None:
            divider_gt = make_axes_locatable(axes[row, 0])
            cax_gt = divider_gt.append_axes("right", size="5%", pad=0.1)
            cbar_gt = fig.colorbar(quiver_gt, cax=cax_gt)
            cbar_gt.set_label("Velocity Magnitude")

        for col, (model_name, model) in enumerate(models_dict.items(), 1):
            try:
                with torch.no_grad():
                    predicted_features = runInference(currentState, config, model, verbose=False)

                if predicted_features.dim() == 2 and predicted_features.shape[1] == 1:
                    predicted_density = predicted_features.squeeze().cpu().numpy()
                else:
                    predicted_density = predicted_features.cpu().numpy()

                derived_velocity = compute_density_gradient_velocity(positions, predicted_density)
                quiver_pred = plot_velocity_field(axes[row, col], positions, derived_velocity,
                                                  f'{model_name} (traj={trajectory_index}, f={frame_offset})')

                # Add colorbar for predicted velocities
                if quiver_pred is not None:
                    divider_pred = make_axes_locatable(axes[row, col])
                    cax_pred = divider_pred.append_axes("right", size="5%", pad=0.1)
                    cbar_pred = fig.colorbar(quiver_pred, cax=cax_pred)
                    cbar_pred.set_label("Velocity Magnitude")

            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'{model_name}\nError', 
                                    ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_xlim(-0.6, 0.6)
                axes[row, col].set_ylim(-0.6, 0.6)

    plt.suptitle(f'Velocity Fields: Trajectory {trajectory_index} - Ground Truth vs Network-Derived', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_density_comparison_sph(models_dict: Dict, ds, hyperParameterDict: Dict,
                                     trajectory_index: int,
                                     frames_per_trajectory: int = 126,
                                     frames: List[int] = [0, 150, 500, 1200, 2200, 3000],
                                     figsize: Tuple[int, int] = (16, 15)) -> None:
    """
    Simple density comparison using sphMath plotting (based on DataLoader notebook pattern)
    """
    trajectory_start_idx = trajectory_index * frames_per_trajectory

    # Calculate actual dataset indices with proper bounds checking
    actual_frames = []
    valid_frames = []
    max_valid_idx = len(ds) - 1

    for f in frames:
        dataset_idx = trajectory_start_idx + f

        # Check if this frame index is valid in the dataset
        if dataset_idx <= max_valid_idx:
            actual_frames.append(dataset_idx)
            valid_frames.append(f)
        else:
            # If frame is beyond dataset bounds, use the last valid frame from this trajectory
            last_valid_frame = min(frames_per_trajectory - 1, max_valid_idx - trajectory_start_idx)
            if last_valid_frame >= 0:
                dataset_idx = trajectory_start_idx + last_valid_frame
                actual_frames.append(dataset_idx)
                valid_frames.append(last_valid_frame)
                print(f"Warning: Frame {f} exceeds dataset bounds, using frame {last_valid_frame} instead")

    n_frames = len(frames)
    fig, axes = plt.subplots(n_frames, 4, figsize=figsize)
    if n_frames == 1:
        axes = axes.reshape(1, -1)

    for model in models_dict.values():
        model.eval()

    model_names = ['linear', 'ffourier', 'chebyshev']

    for row, (frame_offset, frame_idx) in enumerate(zip(valid_frames, actual_frames)):
        try:
            config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
                frame_idx, ds, hyperParameterDict)

            gt = trajectoryStates[0]['fluid']['target']
            gt_density = gt[:, 0] if gt.shape[1] > 0 else gt.flatten()
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            # Create empty plots for this row
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f'Error loading\nFrame {frame_offset}', 
                                    ha='center', va='center', transform=axes[row, col].transAxes,
                                    color='red', fontsize=10)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        # Convert SFBC state to proper particle state (like reference implementation)
        particles = convertSFBCStateToParticleState(currentState)

        # Use domain bounds from config
        domain = su.DomainDescription(
            min=config['domain']['minExtent'],
            max=config['domain']['maxExtent'], 
            periodic=config['domain']['periodicity'],
            dim=config['domain']['dim']
        )

        # Prepare ground truth density for all particles (fluid + boundary)
        if particles.kinds.shape[0] > gt_density.shape[0]:
            # Pad gt_density for boundary particles (usually zeros)
            boundary_count = particles.kinds.shape[0] - gt_density.shape[0]
            boundary_gt = torch.zeros(boundary_count, dtype=gt_density.dtype, device=gt_density.device)
            combined_gt_density = torch.cat([gt_density, boundary_gt], dim=0)
        else:
            combined_gt_density = gt_density

        # Ground truth with sphMath visualizeParticles (fixed domain bounds)
        sp.visualizeParticles(fig, axes[row, 0],
                              particles=particles,
                              domain=domain,
                              quantity=combined_gt_density,
                              which='fluid',
                              cmap='viridis',
                              markerSize=2.0,
                              domainEpsilon=0.05,  # Proper gap around domain boundary
                              plotDomain=True)
        axes[row, 0].set_title(f'Ground Truth (traj={trajectory_index}, f={frame_offset})')

        # Model predictions
        for col, model_name in enumerate(model_names, 1):
            if model_name in models_dict:
                with torch.no_grad():
                    prediction = runInference(currentState, config, models_dict[model_name], verbose=False)

                pred_density = prediction[:, 0] if prediction.shape[1] > 0 else prediction.flatten()

                # Prepare prediction density for all particles (same structure as ground truth)
                if particles.kinds.shape[0] > pred_density.shape[0]:
                    # Pad pred_density for boundary particles (usually zeros)
                    boundary_count = particles.kinds.shape[0] - pred_density.shape[0]
                    boundary_pred = torch.zeros(boundary_count, dtype=pred_density.dtype, device=pred_density.device)
                    combined_pred_density = torch.cat([pred_density, boundary_pred], dim=0)
                else:
                    combined_pred_density = pred_density

                sp.visualizeParticles(fig, axes[row, col],
                                     particles=particles,  # Use same particles object
                                     domain=domain,
                                     quantity=combined_pred_density,
                                     which='fluid', 
                                     cmap='viridis',
                                     markerSize=2.0,
                                     domainEpsilon=0.05,  # Proper gap around domain boundary
                                     plotDomain=True)
                axes[row, col].set_title(f'{model_name} (traj={trajectory_index}, f={frame_offset})')
            else:
                axes[row, col].text(0.5, 0.5, f'{model_name}\nNot Available', 
                                    ha='center', va='center', transform=axes[row, col].transAxes)

    plt.tight_layout()
    plt.show()

def visualize_flow_comparison_sph(models_dict: Dict, ds, hyperParameterDict: Dict,
                                  trajectory_index: int,
                                  frames_per_trajectory: int = 126,
                                  frames: List[int] = [0, 150, 500, 1200, 2200, 3000],
                                  figsize: Tuple[int, int] = (16, 15)) -> None:
    """
    Flow visualization using the same approach as working density visualization
    Use sphMath.visualizeParticles for background + manual streamlines from density gradients
    """
    # Import SPH operations 
    from BasisConvolution.sph.sphOps import sphOperationStates
    from BasisConvolution.sph.neighborhood import neighborSearch
    from scipy.interpolate import griddata

    trajectory_start_idx = trajectory_index * frames_per_trajectory

    # Calculate actual dataset indices with proper bounds checking
    actual_frames = []
    valid_frames = []
    max_valid_idx = len(ds) - 1

    for f in frames:
        dataset_idx = trajectory_start_idx + f

        # Check if this frame index is valid in the dataset
        if dataset_idx <= max_valid_idx:
            actual_frames.append(dataset_idx)
            valid_frames.append(f)
        else:
            # If frame is beyond dataset bounds, use the last valid frame from this trajectory
            last_valid_frame = min(frames_per_trajectory - 1, max_valid_idx - trajectory_start_idx)
            if last_valid_frame >= 0:
                dataset_idx = trajectory_start_idx + last_valid_frame
                actual_frames.append(dataset_idx)
                valid_frames.append(last_valid_frame)
                print(f"Warning: Frame {f} exceeds dataset bounds, using frame {last_valid_frame} instead")

    n_frames = len(frames)
    fig, axes = plt.subplots(n_frames, 4, figsize=figsize)
    if n_frames == 1:
        axes = axes.reshape(1, -1)

    for model in models_dict.values():
        model.eval()

    model_names = ['linear', 'ffourier', 'chebyshev']

    def plot_sph_flow_field(ax, currentState, config, density_field, title):
        """Plot flow field derived FROM the density field - different densities create different flows"""
        try:
            positions = currentState['fluid']['positions'].cpu().numpy()
            density_np = density_field.cpu().numpy()

            # Create grid for visualization
            domain_min = positions.min(axis=0) - 0.02
            domain_max = positions.max(axis=0) + 0.02

            x_range = np.linspace(domain_min[0], domain_max[0], 48)
            y_range = np.linspace(domain_min[1], domain_max[1], 48)
            X, Y = np.meshgrid(x_range, y_range)

            # Normalize density to [0,1] to reveal spatial patterns (aggressive approach)
            density_min = density_np.min()
            density_max = density_np.max()
            if density_max > density_min:  # Avoid division by zero
                density_normalized = (density_np - density_min) / (density_max - density_min)
            else:
                density_normalized = np.zeros_like(density_np)

            # Map normalized density field to grid for background
            density_grid = griddata(positions, density_normalized, (X, Y), method='cubic', fill_value=0)
            density_grid = np.nan_to_num(density_grid, nan=0.0)

            # Create colored background with normalized density (viridis colormap)
            im = ax.pcolormesh(X, Y, density_grid, shading='auto', cmap='viridis', alpha=0.9, vmin=0, vmax=1)

            # Compute flow field FROM density using improved gradient method
            # Use a smoother, physics-inspired approach for computing flow from density
            if len(positions) > 50:
                # Method: Compute density gradient but smooth it properly
                subsample = max(1, len(positions) // 800)
                pos_sub = positions[::subsample] 
                dens_sub = density_np[::subsample]

                if len(pos_sub) > 20:
                    # Compute density on grid first for smoother gradients
                    density_grid_hires = griddata(pos_sub, dens_sub, (X, Y), method='cubic', fill_value=0)
                    density_grid_hires = np.nan_to_num(density_grid_hires, nan=0.0)

                    # Apply gaussian smoothing to reduce noise
                    from scipy.ndimage import gaussian_filter
                    density_smooth = gaussian_filter(density_grid_hires, sigma=1.0)

                    # Compute gradients on the smooth grid (this is the pressure gradient approach)
                    dy, dx = np.gradient(density_smooth)

                    # Convert gradient to flow field (negative for flow direction, scaled for visibility)
                    U = -dx * 0.5  # Scale factor for reasonable arrow lengths
                    V = -dy * 0.5

                    # Check for significant gradients
                    magnitude = np.sqrt(U**2 + V**2)
                    max_mag = np.percentile(magnitude[magnitude > 0], 90) if np.any(magnitude > 0) else 0

                    if max_mag > 1e-6:
                        # Create streamlines showing flow derived from density differences
                        stream = ax.streamplot(X, Y, U, V, 
                                               color='black', 
                                               linewidth=0.7,
                                               density=0.6,  # Less dense for cleaner appearance
                                               arrowstyle='->', 
                                               arrowsize=0.7)

            ax.set_title(title, fontsize=11)
            ax.set_xlim(domain_min[0], domain_max[0])
            ax.set_ylim(domain_min[1], domain_max[1])
            ax.set_aspect('equal')

        except Exception as e:
            print(f"Error creating density-derived flow field for {title}: {e}")
            # Fallback visualization with normalized density
            positions = currentState['fluid']['positions'].cpu().numpy()
            density_np = density_field.cpu().numpy()
            # Normalize density for fallback too
            density_min = density_np.min()
            density_max = density_np.max()
            if density_max > density_min:
                density_normalized = (density_np - density_min) / (density_max - density_min)
            else:
                density_normalized = np.zeros_like(density_np)
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=density_normalized, 
                                 cmap='viridis', s=1, alpha=0.8, vmin=0, vmax=1)
            ax.set_title(f'{title} (Fallback)', fontsize=11)
            ax.set_aspect('equal')

    for row, (frame_offset, frame_idx) in enumerate(zip(valid_frames, actual_frames)):
        try:
            config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
                frame_idx, ds, hyperParameterDict)

            gt = trajectoryStates[0]['fluid']['target']
            gt_density = gt[:, 0] if gt.shape[1] > 0 else gt.flatten()
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            # Create empty plots for this row
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f'Error loading\nFrame {frame_offset}', 
                                    ha='center', va='center', transform=axes[row, col].transAxes,
                                    color='red', fontsize=10)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        # Convert SFBC state to proper particle state (same as working density function)
        particles = convertSFBCStateToParticleState(currentState)

        # Use domain bounds from config (same as working function)
        domain = su.DomainDescription(
            min=config['domain']['minExtent'],
            max=config['domain']['maxExtent'], 
            periodic=config['domain']['periodicity'],
            dim=config['domain']['dim']
        )

        # Ground truth flow: use sphMath for background + manual streamlines for flow
        try:
            # Use sphMath for density background (same as working function)
            if particles.kinds.shape[0] > gt_density.shape[0]:
                boundary_count = particles.kinds.shape[0] - gt_density.shape[0]
                boundary_gt = torch.zeros(boundary_count, dtype=gt_density.dtype, device=gt_density.device)
                combined_gt_density = torch.cat([gt_density, boundary_gt], dim=0)
            else:
                combined_gt_density = gt_density

            # Create background using sphMath (same as working function)
            sp.visualizeParticles(fig, axes[row, 0],
                                 particles=particles,
                                 domain=domain,
                                 quantity=combined_gt_density,  # No normalization needed!
                                 which='fluid',
                                 cmap='viridis',
                                 markerSize=1.0,
                                 domainEpsilon=0.05,
                                 plotDomain=True,
                                 cbar=True)  # Enable colorbar for density scale

            # Add flow arrows on top using density gradients
            if 'neighborhood' not in currentState['fluid']:
                currentState['fluid']['neighborhood'] = neighborSearch(currentState['fluid'], config['domain'], config)

            # Compute SPH gradient for flow arrows
            density_gradient = sphOperationStates(
                currentState['fluid'], 
                currentState['fluid'], 
                (gt_density, gt_density), 
                operation='gradient',
                neighborhood=currentState['fluid']['neighborhood'],
                gradientMode='symmetric'
            )

            # Add streamlines on existing plot
            positions = currentState['fluid']['positions'].cpu().numpy()
            gradient_np = density_gradient.cpu().numpy()

            # Create streamlines from gradient
            if gradient_np.shape[0] > 50:
                domain_bounds = [float(domain.min[0]), float(domain.max[0]), 
                                 float(domain.min[1]), float(domain.max[1])]
                x_range = np.linspace(domain_bounds[0], domain_bounds[1], 32)
                y_range = np.linspace(domain_bounds[2], domain_bounds[3], 32)
                X, Y = np.meshgrid(x_range, y_range)

                # Subsample and interpolate gradients
                subsample = max(1, len(positions) // 1000)
                pos_sub = positions[::subsample]
                grad_sub = gradient_np[::subsample]

                if len(grad_sub) > 20:
                    U = griddata(pos_sub, -grad_sub[:, 0], (X, Y), method='linear', fill_value=0)
                    V = griddata(pos_sub, -grad_sub[:, 1], (X, Y), method='linear', fill_value=0)
                    U = np.nan_to_num(U, nan=0.0)
                    V = np.nan_to_num(V, nan=0.0)

                    magnitude = np.sqrt(U**2 + V**2)
                    if np.percentile(magnitude[magnitude > 0], 90) > 1e-6:
                        axes[row, 0].streamplot(X, Y, U, V, color='black', linewidth=0.8,
                                               density=0.8, arrowstyle='->', arrowsize=0.8)

            axes[row, 0].set_title(f'Ground Truth Flow (traj={trajectory_index}, f={frame_offset})')

        except Exception as e:
            print(f"Error creating ground truth flow: {e}")
            # Fallback - just density without flow
            sp.visualizeParticles(fig, axes[row, 0],
                                 particles=particles,
                                 domain=domain,
                                 quantity=combined_gt_density,
                                 which='fluid',
                                 cmap='viridis',
                                 markerSize=2.0,
                                 domainEpsilon=0.05,
                                 plotDomain=True,
                                 cbar=True)  # Enable colorbar
            axes[row, 0].set_title(f'GT Flow (Error)')

        # Model prediction flows
        for col, model_name in enumerate(model_names, 1):
            if model_name in models_dict:
                try:
                    with torch.no_grad():
                        prediction = runInference(currentState, config, models_dict[model_name], verbose=False)

                    pred_density = prediction[:, 0] if prediction.shape[1] > 0 else prediction.flatten()

                    # Same approach for model predictions
                    if particles.kinds.shape[0] > pred_density.shape[0]:
                        boundary_count = particles.kinds.shape[0] - pred_density.shape[0]
                        boundary_pred = torch.zeros(boundary_count, dtype=pred_density.dtype, device=pred_density.device)
                        combined_pred_density = torch.cat([pred_density, boundary_pred], dim=0)
                    else:
                        combined_pred_density = pred_density

                    # Background using sphMath (no normalization needed)
                    sp.visualizeParticles(fig, axes[row, col],
                                         particles=particles,
                                         domain=domain,
                                         quantity=combined_pred_density,
                                         which='fluid',
                                         cmap='viridis',
                                         markerSize=1.0,
                                         domainEpsilon=0.05,
                                         plotDomain=True,
                                         cbar=True)  # Enable colorbar for density scale

                    # Add flow arrows from model's density gradients
                    pred_gradient = sphOperationStates(
                        currentState['fluid'], 
                        currentState['fluid'], 
                        (pred_density, pred_density), 
                        operation='gradient',
                        neighborhood=currentState['fluid']['neighborhood'],
                        gradientMode='symmetric'
                    )

                    # Add streamlines for this model
                    pred_grad_np = pred_gradient.cpu().numpy()
                    if pred_grad_np.shape[0] > 50:
                        subsample = max(1, len(positions) // 1000)
                        pos_sub = positions[::subsample]
                        grad_sub = pred_grad_np[::subsample]

                        if len(grad_sub) > 20:
                            U = griddata(pos_sub, -grad_sub[:, 0], (X, Y), method='linear', fill_value=0)
                            V = griddata(pos_sub, -grad_sub[:, 1], (X, Y), method='linear', fill_value=0)
                            U = np.nan_to_num(U, nan=0.0)
                            V = np.nan_to_num(V, nan=0.0)

                            magnitude = np.sqrt(U**2 + V**2)
                            if np.percentile(magnitude[magnitude > 0], 90) > 1e-6:
                                axes[row, col].streamplot(X, Y, U, V, color='black', linewidth=0.8,
                                                         density=0.8, arrowstyle='->', arrowsize=0.8)

                    axes[row, col].set_title(f'{model_name} Flow (traj={trajectory_index}, f={frame_offset})')

                except Exception as e:
                    print(f"Error creating {model_name} flow: {e}")
                    axes[row, col].text(0.5, 0.5, f'{model_name}\nFlow Error', 
                                        ha='center', va='center', transform=axes[row, col].transAxes,
                                        color='red', fontsize=10)
            else:
                axes[row, col].text(0.5, 0.5, f'{model_name}\nNot Available', 
                                    ha='center', va='center', transform=axes[row, col].transAxes)

    plt.suptitle(f'Density Flow Visualization: Trajectory {trajectory_index}', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def visualize_velocity_field_single_trajectory(ds, hyperParameterDict: Dict,
                                               trajectory_index: int,
                                               frames_per_trajectory: int = 126,
                                               frames: List[int] = [10, 60, 113],
                                               figsize: Tuple[int, int] = (18, 6)) -> None:
    """
    Visualize velocity field for a single test trajectory at specific frames
    Red = fastest particles, Blue = slowest particles
    """
    trajectory_start_idx = trajectory_index * frames_per_trajectory
    
    actual_frames = []
    valid_frames = []
    max_valid_idx = len(ds) - 1
    
    for f in frames:
        dataset_idx = trajectory_start_idx + f
        if dataset_idx <= max_valid_idx:
            actual_frames.append(dataset_idx)
            valid_frames.append(f)
        else:
            last_valid_frame = min(frames_per_trajectory - 1, max_valid_idx - trajectory_start_idx)
            if last_valid_frame >= 0:
                dataset_idx = trajectory_start_idx + last_valid_frame
                actual_frames.append(dataset_idx)
                valid_frames.append(last_valid_frame)
    
    n_frames = len(valid_frames)
    fig, axes = plt.subplots(1, n_frames, figsize=figsize)
    
    if n_frames == 1:
        axes = [axes]
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.15)
    
    for col, (frame_offset, frame_idx) in enumerate(zip(valid_frames, actual_frames)):
        try:
            config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
                frame_idx, ds, hyperParameterDict)
            
            positions = currentState['fluid']['positions'].cpu().numpy()
            velocities = currentState['fluid']['velocities'].cpu().numpy()
            velocity_magnitude = np.linalg.norm(velocities, axis=1)
            
            scatter = axes[col].scatter(positions[:, 0], positions[:, 1], 
                                       c=velocity_magnitude, 
                                       cmap='coolwarm', 
                                       s=3.0, 
                                       alpha=0.8)
            
            axes[col].set_xlim(config['domain']['minExtent'][0].item(), 
                              config['domain']['maxExtent'][0].item())
            axes[col].set_ylim(config['domain']['minExtent'][1].item(), 
                              config['domain']['maxExtent'][1].item())
            axes[col].set_aspect('equal')
            axes[col].set_title(f'Frame {frame_offset}', fontsize=12, fontweight='bold')
            axes[col].tick_params(labelsize=8)
            
            cbar = plt.colorbar(scatter, ax=axes[col])
            cbar.ax.tick_params(labelsize=8)
            
        except Exception as e:
            print(f"Error processing frame {frame_offset}: {e}")
            axes[col].text(0.5, 0.5, f'Error\nFrame {frame_offset}', 
                          ha='center', va='center', transform=axes[col].transAxes,
                          color='red', fontsize=10)
    
    fig.suptitle(f'Velocity Field: Trajectory {trajectory_index}', fontsize=14, fontweight='bold')
    plt.show()


def debug_domain_particle_alignment(currentState, config, frame_name="Debug"):
    """
    Debug function to visualize the relationship between domain boundaries and particle positions
    """
    import matplotlib.patches as patches

    # Convert SFBC state to particle state
    particles = convertSFBCStateToParticleState(currentState)

    # Get bounds
    actual_min = particles.positions.min(dim=0)[0]
    actual_max = particles.positions.max(dim=0)[0]
    config_min = config['domain']['minExtent'] 
    config_max = config['domain']['maxExtent']

    # Create debug plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot particles
    fluid_mask = particles.kinds == 0
    boundary_mask = particles.kinds == 1

    if torch.sum(fluid_mask) > 0:
        fluid_pos = particles.positions[fluid_mask].cpu().numpy()
        ax.scatter(fluid_pos[:, 0], fluid_pos[:, 1], c='blue', s=1, alpha=0.7, label='Fluid particles')

    if torch.sum(boundary_mask) > 0:
        boundary_pos = particles.positions[boundary_mask].cpu().numpy()
        ax.scatter(boundary_pos[:, 0], boundary_pos[:, 1], c='red', s=2, alpha=0.8, label='Boundary particles')

    # Draw domain boundaries
    domain_width = config_max[0] - config_min[0]
    domain_height = config_max[1] - config_min[1]
    domain_rect = patches.Rectangle(
        (config_min[0], config_min[1]), domain_width, domain_height,
        linewidth=3, edgecolor='blue', facecolor='none', linestyle='--',
        label='Config domain boundary'
    )
    ax.add_patch(domain_rect)

    # Draw actual particle bounding box
    particle_width = actual_max[0] - actual_min[0]
    particle_height = actual_max[1] - actual_min[1]
    particle_rect = patches.Rectangle(
        (actual_min[0], actual_min[1]), particle_width, particle_height,
        linewidth=3, edgecolor='green', facecolor='none', linestyle='-',
        label='Actual particle bounds'
    )
    ax.add_patch(particle_rect)

    ax.set_xlim(config_min[0] - 0.1, config_max[0] + 0.1)
    ax.set_ylim(config_min[1] - 0.1, config_max[1] + 0.1)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f'{frame_name}: Domain vs Particle Position Analysis')
    ax.grid(True, alpha=0.3)

    # Add text with gap information
    gap_left = actual_min[0] - config_min[0]
    gap_right = config_max[0] - actual_max[0] 
    gap_bottom = actual_min[1] - config_min[1]
    gap_top = config_max[1] - actual_max[1]

    ax.text(0.02, 0.98, f'Gaps: L={gap_left:.3f}, R={gap_right:.3f}, B={gap_bottom:.3f}, T={gap_top:.3f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
def visualize_clean_density_comparison(models_dict: Dict, ds, hyperParameterDict: Dict,
                                     trajectory_index: int,
                                     frames_per_trajectory: int = 126,
                                     frames: List[int] = [10, 30, 70, 100],
                                     figsize: Tuple[int, int] = (12, 12)) -> None:
    """
    Clean density comparison with minimal decoration
    Four rows: Ground Truth, Linear, FFourier, and Chebyshev predictions for 4 different frames
    """
    trajectory_start_idx = trajectory_index * frames_per_trajectory

    # Calculate actual dataset indices
    actual_frames = []
    valid_frames = []
    max_valid_idx = len(ds) - 1

    for f in frames:
        dataset_idx = trajectory_start_idx + f
        if dataset_idx <= max_valid_idx:
            actual_frames.append(dataset_idx)
            valid_frames.append(f)
        else:
            last_valid_frame = min(frames_per_trajectory - 1, max_valid_idx - trajectory_start_idx)
            if last_valid_frame >= 0:
                dataset_idx = trajectory_start_idx + last_valid_frame
                actual_frames.append(dataset_idx)
                valid_frames.append(last_valid_frame)

    n_frames = len(valid_frames)
    
    # Create figure with 4 rows, space for colorbars
    fig, axes = plt.subplots(4, n_frames, figsize=figsize)
    
    # Adjust spacing to accommodate colorbars
    plt.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.04, 
                       wspace=0.12, hspace=0.18)

    # Set models to evaluation mode
    for model in models_dict.values():
        model.eval()
    
    # Model names for rows 1-3
    model_names = ['linear', 'ffourier', 'chebyshev']
    
    # Store scatter objects for colorbars
    row_scatters = [None, None, None, None]  # One per row

    # Process each frame
    for col, (frame_offset, frame_idx) in enumerate(zip(valid_frames, actual_frames)):
        try:
            config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
                frame_idx, ds, hyperParameterDict)

            gt = trajectoryStates[0]['fluid']['target']
            gt_density = gt[:, 0] if gt.shape[1] > 0 else gt.flatten()
            
            # Convert SFBC state to particle state
            particles = convertSFBCStateToParticleState(currentState)
            
            # Set up domain
            domain = su.DomainDescription(
                min=config['domain']['minExtent'],
                max=config['domain']['maxExtent'], 
                periodic=config['domain']['periodicity'],
                dim=config['domain']['dim']
            )

            # Prepare ground truth density for all particles
            if particles.kinds.shape[0] > gt_density.shape[0]:
                boundary_count = particles.kinds.shape[0] - gt_density.shape[0]
                boundary_gt = torch.zeros(boundary_count, dtype=gt_density.dtype, device=gt_density.device)
                combined_gt_density = torch.cat([gt_density, boundary_gt], dim=0)
            else:
                combined_gt_density = gt_density

            # Row 0: Ground Truth (with ticks and colorbar support)
            scatter_gt = sp.visualizeParticles(fig, axes[0, col],
                                              particles=particles,
                                              domain=domain,
                                              quantity=combined_gt_density,
                                              which='fluid',
                                              cmap='viridis',
                                              markerSize=2.5,
                                              domainEpsilon=0.01,
                                              plotDomain=False,
                                              cbar=False)
            
            # Keep axis ticks but remove labels
            axes[0, col].set_xlabel('')
            axes[0, col].set_ylabel('')
            axes[0, col].tick_params(labelsize=6)
            
            # Store scatter for colorbar (only from last column)
            if col == n_frames - 1:
                row_scatters[0] = scatter_gt

            # Rows 1-3: Model predictions (Linear, FFourier, Chebyshev)
            for row, model_name in enumerate(model_names, 1):
                if model_name in models_dict:
                    with torch.no_grad():
                        prediction = runInference(currentState, config, models_dict[model_name], verbose=False)

                    pred_density = prediction[:, 0] if prediction.shape[1] > 0 else prediction.flatten()

                    # Prepare prediction density for all particles
                    if particles.kinds.shape[0] > pred_density.shape[0]:
                        boundary_count = particles.kinds.shape[0] - pred_density.shape[0]
                        boundary_pred = torch.zeros(boundary_count, dtype=pred_density.dtype, device=pred_density.device)
                        combined_pred_density = torch.cat([pred_density, boundary_pred], dim=0)
                    else:
                        combined_pred_density = pred_density

                    scatter_pred = sp.visualizeParticles(fig, axes[row, col],
                                                         particles=particles,
                                                         domain=domain,
                                                         quantity=combined_pred_density,
                                                         which='fluid',
                                                         cmap='viridis',
                                                         markerSize=2.5,
                                                         domainEpsilon=0.01,
                                                         plotDomain=False,
                                                         cbar=False)
                    
                    # Keep axis ticks but remove labels
                    axes[row, col].set_xlabel('')
                    axes[row, col].set_ylabel('')
                    axes[row, col].tick_params(labelsize=6)
                    
                    # Store scatter for colorbar (only from last column)
                    if col == n_frames - 1:
                        row_scatters[row] = scatter_pred
                else:
                    axes[row, col].text(0.5, 0.5, f'{model_name.title()}\nNot Available', 
                                      ha='center', va='center', transform=axes[row, col].transAxes,
                                      fontsize=10, color='red')

        except Exception as e:
            print(f"Error processing frame {frame_offset}: {e}")
            # Create empty plots on error but keep ticks
            for row in range(4):
                axes[row, col].tick_params(labelsize=6)
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center', 
                                   transform=axes[row, col].transAxes, color='red')

    # Add row titles for all 4 rows
    fig.text(0.005, 0.875, 'Ground Truth', rotation=90, fontsize=12, fontweight='bold', 
             ha='center', va='center')
    fig.text(0.005, 0.625, 'Linear', rotation=90, fontsize=12, fontweight='bold',
             ha='center', va='center')
    fig.text(0.005, 0.375, 'FFourier', rotation=90, fontsize=12, fontweight='bold',
             ha='center', va='center')
    fig.text(0.005, 0.125, 'Chebyshev', rotation=90, fontsize=12, fontweight='bold',
             ha='center', va='center')
    
    # Add colorbars at the end of each row
    row_positions = [0.875, 0.620, 0.365, 0.110]  # Vertical positions for each row
    row_titles = ['Ground Truth', 'Linear', 'FFourier', 'Chebyshev']
    
    for row_idx, (scatter, position, title) in enumerate(zip(row_scatters, row_positions, row_titles)):
        if scatter is not None:
            # Create colorbar axis for this row (shorter height)
            cbar_ax = fig.add_axes([0.91, position, 0.02, 0.16])  # [left, bottom, width, height]
            
            try:
                # Try to get the color mappable from sphMath visualization
                if hasattr(scatter, 'get_array') and scatter.get_array() is not None:
                    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')
                    cbar.ax.yaxis.set_ticks_position('right')
                    cbar.ax.set_yticks([cbar.vmin, cbar.vmax])
                    cbar.ax.tick_params(labelsize=6)
                else:
                    # Fallback - create a generic colorbar
                    import matplotlib.cm as cm
                    import matplotlib.colors as colors
                    norm = colors.Normalize(vmin=0, vmax=1)
                    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
                    cbar.ax.yaxis.set_ticks_position('right')
                    cbar.ax.set_yticks([0, 1])
                    cbar.ax.tick_params(labelsize=6)
            except Exception as e:
                print(f"Could not create colorbar for {title}: {e}")

    plt.show()

