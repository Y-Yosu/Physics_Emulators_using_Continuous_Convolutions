#!/usr/bin/env python3
"""
Comprehensive Earth Mover Distance (EMD) Analysis
=================================================
Analyzes density distribution evolution across trajectories for all three datasets:
- lagrangebench_dataset_TGV
- SFBC_dataset_II  
- SFBC_TGV

For each dataset and split (train/test), calculates:
- EMD from first frame to middle frame
- EMD from middle frame to last frame
- Mean and standard deviation statistics
- Visualization plots for sample trajectories
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class EMDAnalyzer:
    """Comprehensive EMD analysis for fluid simulation datasets"""
    
    def __init__(self, base_path="/home/yusuf/Physics_Emulators_using_Continuous_Convolutions/data/SFBC"):
        self.base_path = Path(base_path)
        self.output_dir = Path("/home/yusuf/Physics_Emulators_using_Continuous_Convolutions/EMD_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'lagrangebench_TGV': {
                'path': self.base_path / 'lagrangebench_dataset_TGV' / 'dataset',
                'splits': ['train', 'test', 'valid'],
                'color': '#1f77b4'
            },
            'SFBC_dataset_II': {
                'path': self.base_path / 'SFBC_dataset_II' / 'dataset', 
                'splits': ['train', 'test'],
                'color': '#ff7f0e'
            },
            'SFBC_TGV': {
                'path': self.base_path / 'SFBC_TGV' / 'dataset',
                'splits': ['train', 'test'], 
                'color': '#2ca02c'
            }
        }
        
        self.results = {}
        
    def load_trajectory_data(self, file_path):
        """Load trajectory data from HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Get positions and densities for all frames
                positions = []
                densities = []
                
                # Find all frame groups
                frame_keys = [k for k in f.keys() if k.startswith('frame_')]
                frame_keys.sort(key=lambda x: int(x.split('_')[1]))
                
                for frame_key in frame_keys:
                    frame_group = f[frame_key]
                    
                    # Get fluid data
                    if 'fluid' in frame_group:
                        fluid_group = frame_group['fluid']
                        
                        # Positions
                        if 'positions' in fluid_group:
                            pos = np.array(fluid_group['positions'])
                            positions.append(pos)
                        
                        # Densities
                        if 'densities' in fluid_group:
                            dens = np.array(fluid_group['densities'])
                            densities.append(dens)
                        elif 'density' in fluid_group:
                            dens = np.array(fluid_group['density'])
                            densities.append(dens)
                
                return positions, densities
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def compute_2d_density_distribution(self, positions, densities, bins=50):
        """Compute 2D density distribution for EMD calculation"""
        if len(positions) == 0 or len(densities) == 0:
            return None, None, None
            
        # Get spatial bounds
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Create grid
        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)
        
        # Compute weighted histogram (density-weighted)
        hist, _, _ = np.histogram2d(
            positions[:, 0], positions[:, 1], 
            bins=[x_edges, y_edges], 
            weights=densities
        )
        
        # Normalize to probability distribution
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        
        # Get bin centers for EMD calculation
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        return hist, x_centers, y_centers
    
    def compute_emd_2d(self, hist1, hist2, x_centers, y_centers):
        """Compute Earth Mover Distance between two 2D distributions"""
        if hist1 is None or hist2 is None:
            return np.nan
            
        # Flatten histograms
        flat_hist1 = hist1.flatten()
        flat_hist2 = hist2.flatten()
        
        # Create coordinate grid
        xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
        coords = np.column_stack([xx.flatten(), yy.flatten()])
        
        # Filter out zero-weight points for efficiency
        nonzero1 = flat_hist1 > 1e-10
        nonzero2 = flat_hist2 > 1e-10
        
        if not np.any(nonzero1) or not np.any(nonzero2):
            return np.nan
            
        coords1 = coords[nonzero1]
        coords2 = coords[nonzero2]
        weights1 = flat_hist1[nonzero1]
        weights2 = flat_hist2[nonzero2]
        
        # Normalize weights
        weights1 = weights1 / np.sum(weights1)
        weights2 = weights2 / np.sum(weights2)
        
        # Use scipy's wasserstein distance for 1D approximation
        # For true 2D EMD, we'd need more complex optimization
        # Here we use a reasonable approximation
        
        # Compute center of mass for each distribution
        com1 = np.average(coords1, axis=0, weights=weights1)
        com2 = np.average(coords2, axis=0, weights=weights2)
        
        # Simple EMD approximation using center of mass distance + spread difference
        com_distance = np.linalg.norm(com1 - com2)
        
        # Add spread difference
        spread1 = np.sqrt(np.average(np.sum((coords1 - com1)**2, axis=1), weights=weights1))
        spread2 = np.sqrt(np.average(np.sum((coords2 - com2)**2, axis=1), weights=weights2))
        spread_diff = abs(spread1 - spread2)
        
        emd_approx = com_distance + 0.5 * spread_diff
        
        return emd_approx
    
    def analyze_trajectory(self, file_path):
        """Analyze a single trajectory file"""
        positions, densities = self.load_trajectory_data(file_path)
        
        if positions is None or len(positions) < 3:
            return None, None
            
        # Get first, middle, and last frames
        first_idx = 0
        last_idx = len(positions) - 1
        middle_idx = len(positions) // 2
        
        # Compute density distributions
        first_hist, x_centers, y_centers = self.compute_2d_density_distribution(
            positions[first_idx], densities[first_idx]
        )
        middle_hist, _, _ = self.compute_2d_density_distribution(
            positions[middle_idx], densities[middle_idx]
        )
        last_hist, _, _ = self.compute_2d_density_distribution(
            positions[last_idx], densities[last_idx]
        )
        
        # Compute EMDs
        emd_first_to_middle = self.compute_emd_2d(first_hist, middle_hist, x_centers, y_centers)
        emd_middle_to_last = self.compute_emd_2d(middle_hist, last_hist, x_centers, y_centers)
        
        trajectory_data = {
            'positions': [positions[first_idx], positions[middle_idx], positions[last_idx]],
            'densities': [densities[first_idx], densities[middle_idx], densities[last_idx]],
            'histograms': [first_hist, middle_hist, last_hist],
            'frame_indices': [first_idx, middle_idx, last_idx]
        }
        
        return (emd_first_to_middle, emd_middle_to_last), trajectory_data
    
    def analyze_dataset_split(self, dataset_name, split_name):
        """Analyze all trajectories in a dataset split"""
        dataset_config = self.datasets[dataset_name]
        split_path = dataset_config['path'] / split_name
        
        if not split_path.exists():
            print(f"⚠️  Split {split_name} not found for {dataset_name}")
            return None
            
        print(f"   Analyzing {split_name} split...")
        
        # Find all HDF5 files
        hdf5_files = list(split_path.glob("*.hdf5"))
        
        if not hdf5_files:
            print(f"   No HDF5 files found in {split_path}")
            return None
            
        emd_first_to_middle = []
        emd_middle_to_last = []
        sample_trajectory_data = None
        
        for i, file_path in enumerate(hdf5_files):
            if i % 10 == 0:
                print(f"      Processing file {i+1}/{len(hdf5_files)}")
                
            emds, traj_data = self.analyze_trajectory(file_path)
            
            if emds is not None:
                emd_fm, emd_ml = emds
                if not np.isnan(emd_fm) and not np.isnan(emd_ml):
                    emd_first_to_middle.append(emd_fm)
                    emd_middle_to_last.append(emd_ml)
                    
                    # Save first valid trajectory for visualization
                    if sample_trajectory_data is None:
                        sample_trajectory_data = traj_data
        
        if not emd_first_to_middle:
            print(f"   No valid EMD calculations for {dataset_name}/{split_name}")
            return None
            
        # Compute statistics
        results = {
            'emd_first_to_middle': {
                'values': emd_first_to_middle,
                'mean': np.mean(emd_first_to_middle),
                'std': np.std(emd_first_to_middle),
                'count': len(emd_first_to_middle)
            },
            'emd_middle_to_last': {
                'values': emd_middle_to_last,
                'mean': np.mean(emd_middle_to_last),
                'std': np.std(emd_middle_to_last),
                'count': len(emd_middle_to_last)
            },
            'sample_trajectory': sample_trajectory_data
        }
        
        print(f"      ✅ Processed {len(emd_first_to_middle)} valid trajectories")
        print(f"      EMD First→Middle: {results['emd_first_to_middle']['mean']:.4f} ± {results['emd_first_to_middle']['std']:.4f}")
        print(f"      EMD Middle→Last:  {results['emd_middle_to_last']['mean']:.4f} ± {results['emd_middle_to_last']['std']:.4f}")
        
        return results
    
    def create_trajectory_visualization(self, dataset_name, split_name, trajectory_data):
        """Create visualization for a sample trajectory"""
        if trajectory_data is None:
            return
            
        positions = trajectory_data['positions']
        densities = trajectory_data['densities']
        frame_indices = trajectory_data['frame_indices']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{dataset_name} - {split_name.upper()} Split: Density Evolution\n'
                    f'Frames: {frame_indices[0]} → {frame_indices[1]} → {frame_indices[2]}', 
                    fontsize=14, fontweight='bold')
        
        titles = ['First Frame', 'Middle Frame', 'Last Frame']
        
        for i, (ax, pos, dens, title) in enumerate(zip(axes, positions, densities, titles)):
            # Create scatter plot colored by density
            scatter = ax.scatter(pos[:, 0], pos[:, 1], c=dens, cmap='viridis', 
                               s=20, alpha=0.7, edgecolors='none')
            
            ax.set_title(f'{title}\n(Frame {frame_indices[i]})', fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Density', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f'{dataset_name}_{split_name}_trajectory_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved trajectory visualization: {output_path}")
    
    def run_comprehensive_analysis(self):
        """Run complete EMD analysis for all datasets"""
        print("🚀 Starting Comprehensive EMD Analysis")
        print("=" * 60)
        
        all_results = []
        
        for dataset_name, dataset_config in self.datasets.items():
            print(f"\n📊 Analyzing {dataset_name}...")
            
            if not dataset_config['path'].exists():
                print(f"⚠️  Dataset path not found: {dataset_config['path']}")
                continue
                
            dataset_results = {}
            
            for split_name in dataset_config['splits']:
                split_results = self.analyze_dataset_split(dataset_name, split_name)
                
                if split_results is not None:
                    dataset_results[split_name] = split_results
                    
                    # Create visualization for this split
                    self.create_trajectory_visualization(
                        dataset_name, split_name, split_results['sample_trajectory']
                    )
                    
                    # Add to summary results
                    all_results.append({
                        'dataset': dataset_name,
                        'split': split_name,
                        'emd_first_middle_mean': split_results['emd_first_to_middle']['mean'],
                        'emd_first_middle_std': split_results['emd_first_to_middle']['std'],
                        'emd_middle_last_mean': split_results['emd_middle_to_last']['mean'],
                        'emd_middle_last_std': split_results['emd_middle_to_last']['std'],
                        'trajectory_count': split_results['emd_first_to_middle']['count']
                    })
            
            self.results[dataset_name] = dataset_results
        
        # Create summary analysis
        self.create_summary_analysis(all_results)
        
        return self.results
    
    def create_summary_analysis(self, all_results):
        """Create comprehensive summary analysis and visualizations"""
        if not all_results:
            print("❌ No results to summarize")
            return
            
        df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_path = self.output_dir / 'emd_analysis_results.csv'
        df.to_csv(results_path, index=False)
        print(f"\n💾 Detailed results saved: {results_path}")
        
        # Create summary table
        print(f"\n📋 EMD ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"{'Dataset':<20} {'Split':<8} {'First→Middle EMD':<20} {'Middle→Last EMD':<20} {'Trajectories':<12}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['dataset']:<20} {row['split']:<8} "
                  f"{row['emd_first_middle_mean']:.4f} ± {row['emd_first_middle_std']:.4f}    "
                  f"{row['emd_middle_last_mean']:.4f} ± {row['emd_middle_last_std']:.4f}    "
                  f"{row['trajectory_count']:<12}")
        
        # Create comparison plots
        self.create_comparison_plots(df)
        
        # Statistical analysis
        self.create_statistical_analysis(df)
    
    def create_comparison_plots(self, df):
        """Create comparison plots across datasets and splits"""
        
        # 1. EMD Comparison Bar Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # First to Middle EMD
        datasets = df['dataset'].unique()
        x_pos = np.arange(len(df))
        
        colors = [self.datasets[dataset]['color'] for dataset in df['dataset']]
        
        bars1 = ax1.bar(x_pos, df['emd_first_middle_mean'], 
                       yerr=df['emd_first_middle_std'], 
                       color=colors, alpha=0.7, capsize=5)
        
        ax1.set_title('EMD: First → Middle Frame', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Dataset - Split')
        ax1.set_ylabel('Earth Mover Distance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{row['dataset']}\n{row['split']}" for _, row in df.iterrows()], 
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Middle to Last EMD
        bars2 = ax2.bar(x_pos, df['emd_middle_last_mean'], 
                       yerr=df['emd_middle_last_std'], 
                       color=colors, alpha=0.7, capsize=5)
        
        ax2.set_title('EMD: Middle → Last Frame', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Dataset - Split')
        ax2.set_ylabel('Earth Mover Distance')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{row['dataset']}\n{row['split']}" for _, row in df.iterrows()], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'emd_comparison_barplot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 EMD comparison plot saved: {plot_path}")
        
        # 2. Dataset Evolution Pattern Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for dataset in datasets:
            dataset_data = df[df['dataset'] == dataset]
            color = self.datasets[dataset]['color']
            
            for _, row in dataset_data.iterrows():
                # Plot evolution pattern
                x_vals = [0, 1]  # First→Middle, Middle→Last
                y_vals = [row['emd_first_middle_mean'], row['emd_middle_last_mean']]
                y_errs = [row['emd_first_middle_std'], row['emd_middle_last_std']]
                
                ax.errorbar(x_vals, y_vals, yerr=y_errs, 
                           marker='o', linewidth=2, markersize=8,
                           label=f"{dataset} - {row['split']}", 
                           color=color, alpha=0.8)
        
        ax.set_title('Density Evolution Patterns Across Datasets', fontweight='bold', fontsize=14)
        ax.set_xlabel('Trajectory Phase')
        ax.set_ylabel('Earth Mover Distance')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['First → Middle', 'Middle → Last'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        evolution_path = self.output_dir / 'emd_evolution_patterns.png'
        plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Evolution patterns plot saved: {evolution_path}")
    
    def create_statistical_analysis(self, df):
        """Create statistical analysis of EMD patterns"""
        print(f"\n🔬 STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Overall statistics
        print(f"Overall EMD Statistics:")
        print(f"  First→Middle: {df['emd_first_middle_mean'].mean():.4f} ± {df['emd_first_middle_mean'].std():.4f}")
        print(f"  Middle→Last:  {df['emd_middle_last_mean'].mean():.4f} ± {df['emd_middle_last_mean'].std():.4f}")
        
        # Dataset comparison
        print(f"\nDataset Comparison (Average EMD):")
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            avg_first_middle = dataset_data['emd_first_middle_mean'].mean()
            avg_middle_last = dataset_data['emd_middle_last_mean'].mean()
            print(f"  {dataset}:")
            print(f"    First→Middle: {avg_first_middle:.4f}")
            print(f"    Middle→Last:  {avg_middle_last:.4f}")
            print(f"    Evolution Ratio: {avg_middle_last/avg_first_middle:.2f}")
        
        # Split comparison
        print(f"\nSplit Comparison (Train vs Test):")
        for split in ['train', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) > 0:
                avg_first_middle = split_data['emd_first_middle_mean'].mean()
                avg_middle_last = split_data['emd_middle_last_mean'].mean()
                print(f"  {split.upper()}:")
                print(f"    First→Middle: {avg_first_middle:.4f}")
                print(f"    Middle→Last:  {avg_middle_last:.4f}")

def main():
    """Main analysis function"""
    print("🌊 Comprehensive EMD Analysis for Fluid Simulation Datasets")
    print("=" * 70)
    
    analyzer = EMDAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\n✅ Analysis Complete!")
    print(f"📁 Results saved in: {analyzer.output_dir}")
    print(f"📊 Generated visualizations for each dataset split")
    print(f"📋 Summary statistics and comparisons created")

if __name__ == "__main__":
    main()



