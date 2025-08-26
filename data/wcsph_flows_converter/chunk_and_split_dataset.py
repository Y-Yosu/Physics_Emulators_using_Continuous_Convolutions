#!/usr/bin/env python3
"""
WCSPH Flows to SFBC Dataset Converter
=====================================

This script converts the original wcsph_flows test trajectories into SFBC format
by chunking them into smaller trajectories and splitting into train/test sets.

Original data: 4 long trajectories from wcsph_flows (4096 frames each)
Target: 128 trajectories (85 train + 43 test) with 128 frames each

Usage:
    python chunk_and_split_dataset.py
"""

import h5py
import numpy as np
import os
import random
from pathlib import Path
import argparse
from typing import List, Dict, Tuple


class WCSPHToSFBCConverter:
    def __init__(self, 
                 source_path: str,
                 output_base: str,
                 frames_per_chunk: int = 128,
                 train_count: int = 85,
                 test_count: int = 43,
                 random_seed: int = 42):
        
        self.source_path = source_path
        self.output_base = output_base
        self.frames_per_chunk = frames_per_chunk
        self.train_count = train_count
        self.test_count = test_count
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directories
        self.train_dir = os.path.join(output_base, "train")
        self.test_dir = os.path.join(output_base, "test")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        print(f"WCSPH to SFBC Dataset Converter")
        print(f"================================")
        print(f"Source: {source_path}")
        print(f"Output: {output_base}")
        print(f"Chunk size: {frames_per_chunk} frames")
        print(f"Target: {train_count} train + {test_count} test = {train_count + test_count} total")
        print(f"Random seed: {random_seed}")

    def analyze_source_files(self) -> List[Dict]:
        """Analyze source HDF5 files and calculate possible chunks"""
        source_files = sorted([f for f in os.listdir(self.source_path) if f.endswith('.hdf5')])
        
        if len(source_files) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {self.source_path}")
        
        print(f"\nFound {len(source_files)} source files:")
        
        file_info = []
        total_chunks = 0
        
        for i, filename in enumerate(source_files):
            filepath = os.path.join(self.source_path, filename)
            
            with h5py.File(filepath, 'r') as f:
                # Get timestep count
                if 'simulationExport' not in f:
                    print(f"  Warning: {filename} has no simulationExport, skipping")
                    continue
                
                timestep_keys = [key for key in f['simulationExport'].keys() if key.isdigit()]
                frame_count = len(timestep_keys)
                
                # Calculate possible chunks
                possible_chunks = frame_count // self.frames_per_chunk
                remaining_frames = frame_count % self.frames_per_chunk
                
                info = {
                    'filename': filename,
                    'filepath': filepath,
                    'frame_count': frame_count,
                    'possible_chunks': possible_chunks,
                    'remaining_frames': remaining_frames
                }
                
                file_info.append(info)
                total_chunks += possible_chunks
                
                print(f"  {i}: {filename}")
                print(f"      Frames: {frame_count}")
                print(f"      Chunks: {possible_chunks} (+ {remaining_frames} remaining)")
        
        print(f"\nTotal chunks: {total_chunks}")
        print(f"Target chunks: {self.train_count + self.test_count}")
        
        if total_chunks < self.train_count + self.test_count:
            raise ValueError(f"Not enough chunks! Need {self.train_count + self.test_count}, got {total_chunks}")
        
        return file_info

    def convert_chunk_to_sfbc(self, source_file: str, start_frame: int, end_frame: int, output_file: str):
        """Convert a chunk of wcsph_flows data to SFBC format"""
        
        with h5py.File(source_file, 'r') as src:
            # Get timestep keys
            timestep_keys = sorted([key for key in src['simulationExport'].keys() if key.isdigit()], key=int)
            
            # Select the chunk of timesteps
            chunk_timesteps = timestep_keys[start_frame:end_frame]
            
            if len(chunk_timesteps) == 0:
                raise ValueError(f"No timesteps found in range {start_frame}:{end_frame}")
            
            with h5py.File(output_file, 'w') as dst:
                # Copy important file-level attributes for SFBC compatibility
                important_attrs = ['targetNeighbors', 'restDensity', 'radius', 'support', 
                                 'supportRadius', 'area', 'c0', 'EOSgamma', 'defaultKernel', 
                                 'initialDt', 'spacing', 'packing', 'fluidGravity', 'boundaryScheme',
                                 'densityScheme', 'integrationScheme', 'simulationScheme']
                
                for attr_name in important_attrs:
                    if attr_name in src.attrs:
                        dst.attrs[attr_name] = src.attrs[attr_name]
                
                # Set default SFBC attributes if missing
                if 'targetNeighbors' not in dst.attrs:
                    dst.attrs['targetNeighbors'] = 45.228  # From README
                if 'restDensity' not in dst.attrs:
                    dst.attrs['restDensity'] = 1000.0
                if 'support' not in dst.attrs and 'supportRadius' not in dst.attrs:
                    dst.attrs['support'] = 0.079519
                if 'radius' not in dst.attrs:
                    dst.attrs['radius'] = 0.00390625  # 1/(128*2) from README
                if 'area' not in dst.attrs:
                    dst.attrs['area'] = 0.000061035  # From README
                if 'fluidGravity' not in dst.attrs:
                    dst.attrs['fluidGravity'] = np.array([0.0, -9.81], dtype=np.float32)  # Default 2D gravity
                
                # Add other common SFBC defaults
                if 'c0' not in dst.attrs:
                    dst.attrs['c0'] = 100.0  # Speed of sound
                if 'EOSgamma' not in dst.attrs:
                    dst.attrs['EOSgamma'] = 7.0  # Tait EOS gamma
                if 'defaultKernel' not in dst.attrs:
                    dst.attrs['defaultKernel'] = 'wendland2'
                
                # Create simulationExport group
                sim_export = dst.create_group('simulationExport')
                
                # Convert each timestep
                for new_idx, old_timestep in enumerate(chunk_timesteps):
                    src_data = src['simulationExport'][old_timestep]
                    
                    # Create new timestep group (0-indexed with 5-digit padding)
                    new_timestep = f"{new_idx:05d}"
                    dst_timestep = sim_export.create_group(new_timestep)
                    
                    # Copy timestep attributes
                    if 'dt' in src_data.attrs:
                        dst_timestep.attrs['dt'] = src_data.attrs['dt']
                    if 'time' in src_data.attrs:
                        dst_timestep.attrs['time'] = src_data.attrs['time']
                    if 'timestep' in src_data.attrs:
                        dst_timestep.attrs['timestep'] = new_idx  # Use new index
                    
                    # Copy essential data with SFBC naming convention
                    required_fields = ['fluidPosition', 'fluidVelocity', 'fluidDensity']
                    optional_fields = ['UID', 'boundaryDensity', 'boundaryVelocity', 'fluidGravity', 
                                     'fluidAcceleration', 'fluidPressure', 'fluidDpdt']
                    
                    for field in required_fields:
                        if field in src_data:
                            dst_timestep.create_dataset(field, data=src_data[field][:])
                        else:
                            print(f"Warning: Missing required field {field} in {old_timestep}")
                    
                    for field in optional_fields:
                        if field in src_data:
                            dst_timestep.create_dataset(field, data=src_data[field][:])
                    
                    # Add required SFBC fields that might be missing
                    if 'fluidPosition' in src_data:
                        fluid_positions = src_data['fluidPosition'][:]
                        n_particles = len(fluid_positions)
                        
                        # Add fluidSupport (constant support radius)
                        if 'fluidSupport' not in src_data:
                            # Use support radius from file attributes (already copied to dst)
                            support_radius = dst.attrs.get('support', dst.attrs.get('supportRadius', 0.079519))
                            
                            dst_timestep.create_dataset('fluidSupport', 
                                                      data=np.full(n_particles, support_radius, dtype=np.float32))
                        else:
                            dst_timestep.create_dataset('fluidSupport', data=src_data['fluidSupport'][:])
                        
                        # Add fluidArea (constant particle area)
                        if 'fluidArea' not in src_data:
                            particle_area = dst.attrs.get('area', 0.000061035)  # Use value from README
                            
                            dst_timestep.create_dataset('fluidArea', 
                                                      data=np.full(n_particles, particle_area, dtype=np.float32))
                        else:
                            dst_timestep.create_dataset('fluidArea', data=src_data['fluidArea'][:])
                        
                        # Create UID if missing
                        if 'UID' not in src_data:
                            dst_timestep.create_dataset('UID', 
                                                      data=np.arange(n_particles, dtype=np.int64))
                        
                        # Add fluidGravity if missing (SFBC requires this)
                        if 'fluidGravity' not in src_data:
                            # Default gravity: [0, -9.81] for 2D or [0, 0, -9.81] for 3D
                            dims = fluid_positions.shape[1]  # Number of spatial dimensions
                            if dims == 2:
                                gravity_vector = np.array([0.0, -9.81], dtype=np.float32)
                            else:  # 3D
                                gravity_vector = np.array([0.0, 0.0, -9.81], dtype=np.float32)
                            
                            # Create gravity field: same vector for all particles
                            gravity_field = np.tile(gravity_vector, (n_particles, 1))
                            dst_timestep.create_dataset('fluidGravity', data=gravity_field)
                
                # Copy initial conditions if available
                if 'initial' in src:
                    dst.copy(src['initial'], 'initial')
                
                # Copy config if available
                if 'config' in src:
                    dst.copy(src['config'], 'config')
                
                # Add metadata
                metadata = dst.create_group('metadata')
                metadata.attrs['original_file'] = os.path.basename(source_file)
                metadata.attrs['original_start_frame'] = start_frame
                metadata.attrs['original_end_frame'] = end_frame - 1
                metadata.attrs['chunk_length'] = len(chunk_timesteps)
                metadata.attrs['conversion_type'] = 'wcsph_flows_to_sfbc'
                metadata.attrs['converter_version'] = '1.0'

    def generate_chunks(self, file_info: List[Dict]) -> List[Dict]:
        """Generate all possible chunks from source files"""
        all_chunks = []
        chunk_counter = 0
        
        for file_info_item in file_info:
            filepath = file_info_item['filepath']
            filename = file_info_item['filename']
            possible_chunks = file_info_item['possible_chunks']
            
            print(f"\nProcessing {filename} -> {possible_chunks} chunks")
            
            for chunk_idx in range(possible_chunks):
                start_frame = chunk_idx * self.frames_per_chunk
                end_frame = start_frame + self.frames_per_chunk
                
                chunk_info = {
                    'source_file': filepath,
                    'source_filename': filename,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'chunk_id': chunk_counter,
                    'output_filename': f"trajectory_{chunk_counter:03d}.hdf5"
                }
                
                all_chunks.append(chunk_info)
                chunk_counter += 1
                
                print(f"  Chunk {chunk_counter-1}: frames {start_frame}-{end_frame-1}")
        
        return all_chunks

    def split_train_test(self, all_chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Randomly split chunks into train and test sets"""
        
        # Randomly shuffle chunks
        random.shuffle(all_chunks)
        
        # Split into train and test
        train_chunks = all_chunks[:self.train_count]
        test_chunks = all_chunks[self.train_count:self.train_count + self.test_count]
        
        print(f"\nSplit summary:")
        print(f"  Train chunks: {len(train_chunks)}")
        print(f"  Test chunks: {len(test_chunks)}")
        print(f"  Total used: {len(train_chunks) + len(test_chunks)}")
        print(f"  Unused: {len(all_chunks) - len(train_chunks) - len(test_chunks)}")
        
        # Show distribution by source file
        for split_name, chunks in [("Train", train_chunks), ("Test", test_chunks)]:
            print(f"\n{split_name} distribution by source:")
            sources = {}
            for chunk in chunks:
                src = chunk['source_filename']
                sources[src] = sources.get(src, 0) + 1
            for src, count in sorted(sources.items()):
                print(f"  {src}: {count} chunks")
        
        return train_chunks, test_chunks

    def convert_split(self, chunks: List[Dict], output_dir: str, split_name: str):
        """Convert a list of chunks and save to output directory"""
        
        print(f"\n🔄 Converting {split_name} trajectories...")
        
        for i, chunk in enumerate(chunks):
            output_path = os.path.join(output_dir, f"trajectory_{i:03d}.hdf5")
            
            print(f"{split_name} {i+1}/{len(chunks)}: {chunk['source_filename']} "
                  f"[{chunk['start_frame']}:{chunk['end_frame']}] -> trajectory_{i:03d}.hdf5")
            
            try:
                self.convert_chunk_to_sfbc(
                    source_file=chunk['source_file'],
                    start_frame=chunk['start_frame'],
                    end_frame=chunk['end_frame'],
                    output_file=output_path
                )
            except Exception as e:
                print(f"❌ Error converting {split_name} chunk {i}: {e}")
                raise

    def verify_dataset(self):
        """Verify the created dataset"""
        print(f"\n🔍 Verifying created dataset...")
        
        train_files = [f for f in os.listdir(self.train_dir) if f.endswith('.hdf5')]
        test_files = [f for f in os.listdir(self.test_dir) if f.endswith('.hdf5')]
        
        print(f"Created files:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Test: {len(test_files)} files")
        print(f"  Total: {len(train_files) + len(test_files)} files")
        
        # Verify a few files
        def verify_trajectory(filepath):
            """Verify a trajectory file"""
            try:
                with h5py.File(filepath, 'r') as f:
                    if 'simulationExport' not in f:
                        return False, "No simulationExport"
                    
                    timesteps = [k for k in f['simulationExport'].keys() if k.isdigit()]
                    frame_count = len(timesteps)
                    
                    if frame_count != self.frames_per_chunk:
                        return False, f"Expected {self.frames_per_chunk} frames, got {frame_count}"
                    
                    # Check first frame
                    first_frame = f['simulationExport']['00000']
                    required_keys = ['fluidPosition', 'fluidVelocity', 'fluidDensity']
                    missing_keys = [k for k in required_keys if k not in first_frame]
                    
                    if missing_keys:
                        return False, f"Missing keys: {missing_keys}"
                    
                    particle_count = first_frame['fluidPosition'].shape[0]
                    return True, f"{frame_count} frames, {particle_count} particles"
            except Exception as e:
                return False, f"Error: {e}"
        
        # Verify first few files
        print(f"\nVerification samples:")
        for split, files, dir_path in [("Train", train_files[:3], self.train_dir), 
                                       ("Test", test_files[:3], self.test_dir)]:
            print(f"\n{split}:")
            for filename in files:
                filepath = os.path.join(dir_path, filename)
                success, message = verify_trajectory(filepath)
                status = "✅" if success else "❌"
                print(f"  {status} {filename}: {message}")
        
        return len(train_files), len(test_files)

    def run(self):
        """Run the complete conversion process"""
        print(f"\n{'='*50}")
        print(f"Starting WCSPH to SFBC conversion...")
        print(f"{'='*50}")
        
        # Step 1: Analyze source files
        file_info = self.analyze_source_files()
        
        # Step 2: Generate all chunks
        all_chunks = self.generate_chunks(file_info)
        print(f"\n✅ Generated {len(all_chunks)} chunk definitions")
        
        # Step 3: Split into train/test
        train_chunks, test_chunks = self.split_train_test(all_chunks)
        
        # Step 4: Convert train set
        self.convert_split(train_chunks, self.train_dir, "Train")
        print(f"✅ Train conversion complete: {len(train_chunks)} trajectories")
        
        # Step 5: Convert test set
        self.convert_split(test_chunks, self.test_dir, "Test")
        print(f"✅ Test conversion complete: {len(test_chunks)} trajectories")
        
        # Step 6: Verify
        train_count, test_count = self.verify_dataset()
        
        print(f"\n🎯 CONVERSION COMPLETE!")
        print(f"{'='*50}")
        print(f"📊 Dataset Summary:")
        print(f"  Source: 4 wcsph_flows trajectories")
        print(f"  Created: {train_count} train + {test_count} test = {train_count + test_count} total")
        print(f"  Frames per trajectory: {self.frames_per_chunk}")
        print(f"  Output location: {self.output_base}")
        print(f"  Random seed: {self.random_seed}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Convert WCSPH flows to SFBC format')
    parser.add_argument('--source', type=str, 
                       default='/home/yusuf/Physics_Emulators_using_Continuous_Convolutions/data/wcsph_flows/testing/noObstacle',
                       help='Source directory containing wcsph_flows HDF5 files')
    parser.add_argument('--output', type=str,
                       default='/home/yusuf/Physics_Emulators_using_Continuous_Convolutions/data/SFBC/SFBC_TGV/dataset',
                       help='Output directory for SFBC dataset')
    parser.add_argument('--frames-per-chunk', type=int, default=128,
                       help='Number of frames per trajectory chunk')
    parser.add_argument('--train-count', type=int, default=85,
                       help='Number of training trajectories')
    parser.add_argument('--test-count', type=int, default=43,
                       help='Number of test trajectories')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.source):
        print(f"❌ Error: Source directory does not exist: {args.source}")
        return 1
    
    if args.train_count + args.test_count > 128:
        print(f"❌ Error: Total trajectories ({args.train_count + args.test_count}) exceeds expected maximum (128)")
        return 1
    
    # Create converter and run
    converter = WCSPHToSFBCConverter(
        source_path=args.source,
        output_base=args.output,
        frames_per_chunk=args.frames_per_chunk,
        train_count=args.train_count,
        test_count=args.test_count,
        random_seed=args.random_seed
    )
    
    try:
        converter.run()
        return 0
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 