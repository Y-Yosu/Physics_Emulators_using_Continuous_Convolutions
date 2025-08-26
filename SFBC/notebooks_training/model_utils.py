import os
import torch
import json
from BasisConvolution.util.network import buildModel

def save_model(model, basis_name, save_dir):
    """
    Save a single trained model
    
    Args:
        model: Trained PyTorch model
        basis_name: Name of basis function (e.g., 'ffourier', 'linear', 'chebyshev')
        save_dir: Directory to save the model
        
    Returns:
        str: Path to saved model file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model_filename = f"{basis_name}_model.pth"
    model_path = os.path.join(save_dir, model_filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'basis_function': basis_name
    }, model_path)
    
    print(f"✅ Saved {basis_name} model to {model_path}")
    return model_path

def save_hyperparameters(hyperParameterDict, save_dir):
    """
    Save hyperparameters for all models in a directory
    
    Args:
        hyperParameterDict: Hyperparameter dictionary
        save_dir: Directory to save the hyperparameters
        
    Returns:
        str: Path to saved hyperparameters file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    hyperparams_path = os.path.join(save_dir, "hyperparameters.pth")
    
    torch.save(hyperParameterDict, hyperparams_path)
    
    print(f"✅ Saved hyperparameters to {hyperparams_path}")
    return hyperparams_path

def load_hyperparameters(save_dir, device='cuda'):
    """
    Load hyperparameters from a directory
    
    Args:
        save_dir: Directory where hyperparameters were saved
        device: Device to set in hyperparameters ('cuda' or 'cpu')
        
    Returns:
        dict: Hyperparameter dictionary
    """
    hyperparams_path = os.path.join(save_dir, "hyperparameters.pth")
    
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_path}")
    
    hyperParameterDict = torch.load(hyperparams_path, map_location=device, weights_only=False)
    hyperParameterDict['device'] = device  # Update device
    
    print(f"✅ Loaded hyperparameters from {hyperparams_path}")
    return hyperParameterDict

def load_model(basis_name, save_dir, device='cuda'):
    """
    Load a single trained model (automatically loads hyperparameters from same directory)
    
    Args:
        basis_name: Name of basis function (e.g., 'ffourier', 'linear', 'chebyshev')
        save_dir: Directory where model was saved
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    model_filename = f"{basis_name}_model.pth"
    model_path = os.path.join(save_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load hyperparameters from same directory
    hyperParameterDict = load_hyperparameters(save_dir, device)
    
    # Peek at the saved model to determine architecture based on actual state_dict keys
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    
    # GraphNetwork models have keys like "messageProcessors", "skipConnections"
    # BasisNetwork models have keys like "convs", "fcs"  
    is_old_graphnetwork = any('messageProcessors' in key or 'skipConnections' in key for key in state_dict_keys)
    
    if is_old_graphnetwork:
        # Fix activation='default' issue from models saved without config files
        dictList = ['inputEncoder', 'outputDecoder', 'edgeMLP', 'vertexMLP', 'fcLayerMLP', 'inputEdgeEncoder', 'messageMLP']
        for d in dictList:
            if d in hyperParameterDict and 'activation' in hyperParameterDict[d]:
                if hyperParameterDict[d]['activation'] == 'default':
                    hyperParameterDict[d]['activation'] = hyperParameterDict['activation']
        
        # Create GraphNetwork directly for old saved models
        from BasisConvolution.convNetv3 import GraphNetwork
        from BasisConvolution.detail.windows import getWindowFunction
        
        fluidFeatureCount = hyperParameterDict['fluidFeatureCount']
        boundaryFeaturecount = hyperParameterDict['boundaryFeatureCount']
        layers = hyperParameterDict['layers']
        coordinateMapping = hyperParameterDict['coordinateMapping']
        windowFunction = getWindowFunction(hyperParameterDict['windowFunction'])
        
        outputDecoder = hyperParameterDict['outputDecoder'] if hyperParameterDict['outputDecoderActive'] else None
        inputEncoder = hyperParameterDict['inputEncoder'] if hyperParameterDict['inputEncoderActive'] else None
        vertexMLP = hyperParameterDict['vertexMLP'] if hyperParameterDict['vertexMLPActive'] else None
        edgeMLP = hyperParameterDict['edgeMLP'] if hyperParameterDict['edgeMLPActive'] else None
        fcMLP = hyperParameterDict['fcLayerMLP'] if hyperParameterDict['fcLayerMLPActive'] else None
        inputEdgeEncoder = hyperParameterDict['inputEdgeEncoder'] if hyperParameterDict['inputEdgeEncoderActive'] else None
        inputBasisEncoder = hyperParameterDict['inputBasisEncoder'] if hyperParameterDict['inputBasisEncoderActive'] else None
        convLayerDict = hyperParameterDict['convLayer']
        normalization = hyperParameterDict['normalization']
        outputScaling = hyperParameterDict['outputScaling'] if 'outputScaling' in hyperParameterDict else 1/128
        
        model = GraphNetwork(
            fluidFeatures = fluidFeatureCount, boundaryFeatures = boundaryFeaturecount, dim = hyperParameterDict['dimension'], layers = hyperParameterDict['layers'], activation = hyperParameterDict['activation'],
            coordinateMapping=coordinateMapping, windowFn = windowFunction, 
            vertexMLP = vertexMLP,
            edgeMLP = edgeMLP, edgeMode = hyperParameterDict['edgeMode'],
            outputDecoder = outputDecoder, inputEncoder = inputEncoder, 
            skipLayerMLP = fcMLP, skipLayerMode = hyperParameterDict['skipLayerMode'], skipConnectionMode = hyperParameterDict['skipConnectionMode'],
            verbose = False,
            inputEdgeEncoder=inputEdgeEncoder, basisEncoder=inputBasisEncoder, normalization=normalization,
            convLayer = convLayerDict, messageMLP = hyperParameterDict['messageMLP'],
            activationOnNode = hyperParameterDict['activationOnNode'],
            outputScaling = outputScaling
        ).to(device)
        
        print(f"✅ Created GraphNetwork for old saved model: {basis_name}")
    else:
        # Use BasisNetwork for newer models
        model, _, _ = buildModel(hyperParameterDict, verbose=False)
        print(f"✅ Created BasisNetwork for newer model: {basis_name}")
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Loaded {basis_name} model from {model_path}")
    return model

def save_all_models(models_dict, save_dir):
    """
    Save all models from a dictionary (convenience function)
    
    Args:
        models_dict: Dictionary mapping basis function names to trained models
        save_dir: Directory to save the models
        
    Returns:
        list: List of saved model paths
    """
    saved_paths = []
    for basis_name, model in models_dict.items():
        model_path = save_model(model, basis_name, save_dir)
        saved_paths.append(model_path)
    
    print(f"\n✅ All {len(models_dict)} models saved to {save_dir}")
    return saved_paths

def load_all_models(basis_names, save_dir, device='cuda'):
    """
    Load multiple models (convenience function)
    
    Args:
        basis_names: List of basis function names to load
        save_dir: Directory where models were saved
        device: Device to load models on ('cuda' or 'cpu')
        
    Returns:
        dict: Dictionary mapping basis function names to loaded models
    """
    models_dict = {}
    for basis_name in basis_names:
        try:
            model = load_model(basis_name, save_dir, device)
            models_dict[basis_name] = model
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
    
    print(f"\n✅ Loaded {len(models_dict)} models: {list(models_dict.keys())}")
    return models_dict

def list_saved_models(save_dir):
    """
    List all saved models in a directory
    
    Args:
        save_dir: Directory to check for saved models
        
    Returns:
        list: List of basis function names found
    """
    if not os.path.exists(save_dir):
        print(f"Directory not found: {save_dir}")
        return []
    
    model_files = [f for f in os.listdir(save_dir) if f.endswith('_model.pth')]
    basis_names = [f.replace('_model.pth', '') for f in model_files]
    
    if basis_names:
        print(f"Found saved models: {basis_names}")
    else:
        print("No saved models found")
    
    return basis_names

def load_models_for_visualization(save_dir, device='cuda'):
    """
    Load models for visualization - automatically finds all models and returns models_dict
    
    Args:
        save_dir: Directory where models were saved
        device: Device to load models on ('cuda' or 'cpu')
        
    Returns:
        tuple: (models_dict, hyperParameterDict)
            - models_dict: Dictionary ready for visualization functions
            - hyperParameterDict: Hyperparameters for dataset loading
    """
    available_models = list_saved_models(save_dir)
    
    if not available_models:
        raise FileNotFoundError(f"No models found in {save_dir}")
    
    # Load hyperparameters for dataset loading
    hyperParameterDict = load_hyperparameters(save_dir, device)
    
    models_dict = {}
    for basis_name in available_models:
        try:
            model = load_model(basis_name, save_dir, device)
            models_dict[basis_name] = model
        except FileNotFoundError as e:
            print(f"⚠️  Failed to load {basis_name}: {e}")
    
    print(f"\n🎯 Ready for visualization with models: {list(models_dict.keys())}")
    return models_dict, hyperParameterDict 