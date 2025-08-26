"""
Debug script to compare fresh trained model vs loaded model
"""
import torch
import numpy as np
from BasisConvolution.util.augment import loadAugmentedFrame
from BasisConvolution.util.network import runInference

def debug_save_load_issue(fresh_model, basis_name, save_dir, test_ds, hyperParameterDict):
    """
    Compare fresh trained model vs loaded model outputs
    """
    print(f"\n🔬 DEBUGGING {basis_name.upper()} SAVE/LOAD ISSUE")
    print("="*50)
    
    # Get test data
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(
        0, test_ds, hyperParameterDict, unrollLength=0)
    
    # Test fresh model
    fresh_model.eval()
    with torch.no_grad():
        fresh_pred = runInference([currentState], [config], fresh_model, verbose=False)[0]
    
    print(f"✅ FRESH MODEL OUTPUT:")
    print(f"   Range: [{fresh_pred.min():.3f}, {fresh_pred.max():.3f}]")
    print(f"   Mean: {fresh_pred.mean():.3f}, Std: {fresh_pred.std():.3f}")
    
    # Save model
    from model_utils import save_model, save_hyperparameters
    save_model(fresh_model, basis_name, save_dir)
    save_hyperparameters(hyperParameterDict, save_dir)
    
    # Load model
    from model_utils import load_model
    loaded_model = load_model(basis_name, save_dir, device=hyperParameterDict['device'])
    
    # Test loaded model
    loaded_model.eval()
    with torch.no_grad():
        loaded_pred = runInference([currentState], [config], loaded_model, verbose=False)[0]
    
    print(f"🚨 LOADED MODEL OUTPUT:")
    print(f"   Range: [{loaded_pred.min():.3f}, {loaded_pred.max():.3f}]")
    print(f"   Mean: {loaded_pred.mean():.3f}, Std: {loaded_pred.std():.3f}")
    
    # Compare outputs
    diff = torch.abs(fresh_pred - loaded_pred).max().item()
    print(f"\n📊 COMPARISON:")
    print(f"   Max difference: {diff:.6f}")
    
    if diff > 1e-4:
        print(f"🚨 MODELS ARE DIFFERENT! Max diff: {diff}")
        return False
    else:
        print(f"✅ Models are identical (diff < 1e-4)")
        return True
    
def debug_model_architecture(fresh_model, loaded_model, basis_name):
    """
    Compare model architectures
    """
    print(f"\n🏗️  ARCHITECTURE COMPARISON - {basis_name}")
    print("="*50)
    
    # Compare state dicts
    fresh_keys = set(fresh_model.state_dict().keys())
    loaded_keys = set(loaded_model.state_dict().keys())
    
    print(f"Fresh model keys: {len(fresh_keys)}")
    print(f"Loaded model keys: {len(loaded_keys)}")
    
    missing_in_loaded = fresh_keys - loaded_keys
    extra_in_loaded = loaded_keys - fresh_keys
    
    if missing_in_loaded:
        print(f"🚨 Missing in loaded: {missing_in_loaded}")
    if extra_in_loaded:
        print(f"🚨 Extra in loaded: {extra_in_loaded}")
    
    # Compare weights for common keys
    common_keys = fresh_keys & loaded_keys
    for key in list(common_keys)[:5]:  # Check first 5 layers
        fresh_weight = fresh_model.state_dict()[key]
        loaded_weight = loaded_model.state_dict()[key]
        diff = torch.abs(fresh_weight - loaded_weight).max().item()
        print(f"   {key}: diff = {diff:.6f}")
        if diff > 1e-6:
            print(f"     🚨 Weight mismatch!")

def debug_hyperparameters(original_hyperParams, save_dir):
    """
    Compare original vs loaded hyperparameters
    """
    print(f"\n📋 HYPERPARAMETER COMPARISON")
    print("="*50)
    
    from model_utils import load_hyperparameters
    loaded_hyperParams = load_hyperparameters(save_dir, device=original_hyperParams['device'])
    
    # Compare critical hyperparameters
    critical_keys = ['basisFunctions', 'basisTerms', 'layers', 'activation', 
                    'fluidFeatureCount', 'boundaryFeatureCount', 'dimension']
    
    for key in critical_keys:
        if key in original_hyperParams and key in loaded_hyperParams:
            orig_val = original_hyperParams[key]
            loaded_val = loaded_hyperParams[key]
            if orig_val != loaded_val:
                print(f"🚨 {key}: {orig_val} → {loaded_val}")
            else:
                print(f"✅ {key}: {orig_val}")
        else:
            print(f"⚠️  {key}: missing in one of the dicts")
