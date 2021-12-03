import torch.nn as nn

# TODO: more complex weight initialization function
# * choice of initialization strategy
# * choice of bias terms
# * choice of different initialization for different types (Linear, Conv, etc.)


def basic_weight_init(module: nn.Module):
    """weight_init(model) or model.apply(weight_init)

    This function initializes the weights of a module using xavier_normal.

    Args:
        module (nn.Module): PyTorch module to be initialized.
    Returns:
        None
    """
    if type(module) in [nn.Linear, ]:
        nn.init.xavier_normal_(module.weight)

def basic_weight_init_glorut_uniform(module: nn.Module):
    """weight_init(model) or model.apply(weight_init)

    This function initializes the weights of a module using xavier_uniform.

    Args:
        module (nn.Module): PyTorch module to be initialized.
    Returns:
        None
    """
    if type(module) in [nn.Linear, ]:
        nn.init.xavier_uniform_(module.weight)

def basic_weight_init_he_normal_relu(module: nn.Module):
    """weight_init(model) or model.apply(weight_init)

    This function initializes the weights of a module using kaiming_normal.

    Args:
        module (nn.Module): PyTorch module to be initialized.
    Returns:
        None
    """
    if type(module) in [nn.Linear, ]:
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(module.bias)

def basic_weight_init_he_uniform_relu(module: nn.Module):
    """weight_init(model) or model.apply(weight_init)

    This function initializes the weights of a module using kaiming_uniform.

    Args:
        module (nn.Module): PyTorch module to be initialized.
    Returns:
        None
    """
    if type(module) in [nn.Linear, ]:
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(module.bias)

        
