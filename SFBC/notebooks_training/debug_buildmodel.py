"""
Debug buildModel behavior with different hyperparameters
"""
import torch
from BasisConvolution.util.network import buildModel

def debug_buildModel_comparison(fresh_hyperParams, loaded_hyperParams):
    """
    Compare what buildModel creates with different hyperparameters
    """
    print(f"\n🏗️  DEBUGGING buildModel ARCHITECTURE DIFFERENCES")
    print("="*70)
    
    # Show critical hyperparameters
    critical_keys = ['basisFunctions', 'basisTerms', 'layers', 'activation', 
                    'convLayer', 'fluidFeatureCount', 'boundaryFeatureCount']
    
    print(f"📋 FRESH HYPERPARAMETERS:")
    for key in critical_keys:
        if key in fresh_hyperParams:
            val = fresh_hyperParams[key]
            if key == 'convLayer' and isinstance(val, dict):
                print(f"   {key}:")
                for subkey, subval in val.items():
                    print(f"      {subkey}: {subval}")
            else:
                print(f"   {key}: {val}")
        else:
            print(f"   {key}: MISSING")
    
    print(f"\n📋 LOADED HYPERPARAMETERS:")
    for key in critical_keys:
        if key in loaded_hyperParams:
            val = loaded_hyperParams[key]
            if key == 'convLayer' and isinstance(val, dict):
                print(f"   {key}:")
                for subkey, subval in val.items():
                    print(f"      {subkey}: {subval}")
            else:
                print(f"   {key}: {val}")
        else:
            print(f"   {key}: MISSING")
    
    # Create models and compare
    print(f"\n🏗️  CREATING MODELS:")
    
    # Fresh model
    print(f"Creating with FRESH hyperparams...")
    fresh_model, _, _ = buildModel(fresh_hyperParams, verbose=True)
    fresh_keys = list(fresh_model.state_dict().keys())
    print(f"   Fresh model: {len(fresh_keys)} parameters")
    for key in fresh_keys:
        shape = fresh_model.state_dict()[key].shape
        print(f"      {key}: {shape}")
    
    # Loaded model  
    print(f"\nCreating with LOADED hyperparams...")
    loaded_model, _, _ = buildModel(loaded_hyperParams, verbose=True)
    loaded_keys = list(loaded_model.state_dict().keys())
    print(f"   Loaded model: {len(loaded_keys)} parameters")
    for key in loaded_keys:
        shape = loaded_model.state_dict()[key].shape
        print(f"      {key}: {shape}")
    
    # Compare
    print(f"\n⚖️  ARCHITECTURE COMPARISON:")
    if fresh_keys == loaded_keys:
        print(f"✅ Architectures are IDENTICAL")
    else:
        print(f"❌ Architectures are DIFFERENT")
        missing_in_loaded = set(fresh_keys) - set(loaded_keys)
        extra_in_loaded = set(loaded_keys) - set(fresh_keys)
        if missing_in_loaded:
            print(f"   Missing in loaded: {missing_in_loaded}")
        if extra_in_loaded:
            print(f"   Extra in loaded: {extra_in_loaded}")

def find_hyperparameter_differences(fresh_hyperParams, loaded_hyperParams):
    """
    Find all differences between hyperparameter dictionaries
    """
    print(f"\n🔍 COMPLETE HYPERPARAMETER DIFF:")
    print("="*70)
    
    all_keys = set(fresh_hyperParams.keys()) | set(loaded_hyperParams.keys())
    
    for key in sorted(all_keys):
        fresh_val = fresh_hyperParams.get(key, "MISSING")
        loaded_val = loaded_hyperParams.get(key, "MISSING")
        
        if fresh_val != loaded_val:
            print(f"❌ {key}:")
            print(f"   Fresh:  {fresh_val}")
            print(f"   Loaded: {loaded_val}")
        else:
            print(f"✅ {key}: {fresh_val}")
