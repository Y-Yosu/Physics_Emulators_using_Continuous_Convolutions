"""
Detailed debugging for hyperparameter save/load
"""
import torch
import json

def debug_hyperparameter_save_load(original_hyperParams, save_dir):
    """
    Debug exactly what gets saved vs loaded
    """
    print(f"\n🔬 DETAILED HYPERPARAMETER DEBUG")
    print("="*60)
    
    # Check what's in original hyperParams
    print(f"📋 ORIGINAL HYPERPARAMETERS:")
    critical_keys = ['basisFunctions', 'basisTerms', 'layers', 'activation', 
                    'windowFunction', 'fluidFeatureCount', 'boundaryFeatureCount']
    
    for key in critical_keys:
        if key in original_hyperParams:
            print(f"   ✅ {key}: {original_hyperParams[key]}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    print(f"\n💾 SAVING HYPERPARAMETERS...")
    from model_utils import save_hyperparameters
    hyperparams_path = save_hyperparameters(original_hyperParams, save_dir)
    
    # Directly load and check what was saved
    print(f"\n📂 CHECKING SAVED FILE CONTENTS:")
    saved_data = torch.load(hyperparams_path, map_location='cpu', weights_only=False)
    
    for key in critical_keys:
        if key in saved_data:
            print(f"   ✅ {key}: {saved_data[key]}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    # Load through model_utils function
    print(f"\n📥 LOADING THROUGH model_utils:")
    from model_utils import load_hyperparameters
    loaded_hyperParams = load_hyperparameters(save_dir, device=original_hyperParams['device'])
    
    for key in critical_keys:
        if key in loaded_hyperParams:
            print(f"   ✅ {key}: {loaded_hyperParams[key]}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    return loaded_hyperParams

def test_buildModel_with_missing_params(hyperParams):
    """
    Test what happens when buildModel is called with missing parameters
    """
    print(f"\n🏗️  TESTING buildModel WITH CURRENT HYPERPARAMS:")
    print("="*60)
    
    try:
        from BasisConvolution.util.network import buildModel
        model, optimizer, scheduler = buildModel(hyperParams, verbose=True)
        print(f"✅ buildModel succeeded")
        
        # Check if the model has the expected structure
        state_dict_keys = list(model.state_dict().keys())
        print(f"   Model has {len(state_dict_keys)} parameters")
        for key in state_dict_keys[:3]:  # Show first 3 keys
            print(f"   - {key}: {model.state_dict()[key].shape}")
            
    except Exception as e:
        print(f"❌ buildModel failed: {e}")
        import traceback
        traceback.print_exc()

def compare_model_creation(fresh_hyperParams, loaded_hyperParams):
    """
    Compare models created with fresh vs loaded hyperparams
    """
    print(f"\n⚖️  COMPARING MODEL CREATION:")
    print("="*60)
    
    from BasisConvolution.util.network import buildModel
    
    # Create with fresh hyperparams
    print("Creating with FRESH hyperparams...")
    fresh_model, _, _ = buildModel(fresh_hyperParams, verbose=False)
    fresh_keys = list(fresh_model.state_dict().keys())
    print(f"   Fresh model: {len(fresh_keys)} parameters")
    
    # Create with loaded hyperparams  
    print("Creating with LOADED hyperparams...")
    loaded_model, _, _ = buildModel(loaded_hyperParams, verbose=False)
    loaded_keys = list(loaded_model.state_dict().keys())
    print(f"   Loaded model: {len(loaded_keys)} parameters")
    
    # Compare architectures
    if fresh_keys == loaded_keys:
        print("✅ Model architectures are identical")
        
        # Compare shapes
        for key in fresh_keys:
            fresh_shape = fresh_model.state_dict()[key].shape
            loaded_shape = loaded_model.state_dict()[key].shape
            if fresh_shape != loaded_shape:
                print(f"❌ Shape mismatch {key}: {fresh_shape} vs {loaded_shape}")
        
    else:
        print("❌ Model architectures are DIFFERENT!")
        missing_in_loaded = set(fresh_keys) - set(loaded_keys)
        extra_in_loaded = set(loaded_keys) - set(fresh_keys)
        if missing_in_loaded:
            print(f"   Missing in loaded: {missing_in_loaded}")
        if extra_in_loaded:
            print(f"   Extra in loaded: {extra_in_loaded}")
