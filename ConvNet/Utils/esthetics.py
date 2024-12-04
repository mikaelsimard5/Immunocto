def generate_model_name(config):
    """
    Generate a descriptive model name based on the configuration parameters.

    Args:
        config (dict): Configuration dictionary containing settings for the model.

    Returns:
        str: Generated model name.
    """
    # Extract sub-configurations with defaults to empty dictionaries
    basemodel = config.get("BASEMODEL", {})
    advancedmodel = config.get("ADVANCEDMODEL", {})
    augmentation = config.get("AUGMENTATION", {})
    data = config.get("DATA", {})
    optimizer = config.get("OPTIMIZER", {})
    regularization = config.get("REGULARIZATION", {})
    scheduler = config.get("SCHEDULER", {})

    def safe_get(dictionary, key, default="NA"):
        """Helper function to safely fetch a value or return a default."""
        value = dictionary.get(key, default)
        return value if value is not None else default

    # Construct the model name
    model_name = "_".join(filter(None, [
        f"{safe_get(basemodel, 'backbone')}",
        f"BS{safe_get(basemodel, 'batch_size')}",
        f"{safe_get(basemodel, 'loss_function', '')[:2]}Loss",
        f"{safe_get(basemodel, 'precision')}",
        f"{str(safe_get(basemodel, 'target_pixel_size')).replace('.', 'p')}um",
        f"{safe_get(basemodel, 'target_patch_size', ['NA'])}px",
        f"E{safe_get(advancedmodel, 'max_epochs')}",
        f"{'pretrain' if safe_get(advancedmodel, 'pretrained') else 'nopretrain'}",
        f"seed{safe_get(advancedmodel, 'random_seed')}",
        f"sigma{str(safe_get(augmentation, 'colour_sigma')).replace('.', 'p')}",
        f"T{str(safe_get(data, 'train_size')).replace('.', 'p')}",
        f"V{str(safe_get(data, 'val_size')).replace('.', 'p')}",
        f"eps{safe_get(optimizer, 'eps')}",
        f"lr{safe_get(optimizer, 'lr')}",
        f"fold{safe_get(data, 'train_fold', ['NA'])}",
        f"ls{str(safe_get(regularization, 'label_smoothing')).replace('.', 'p')}",
        f"{safe_get(scheduler, 'type')}"
    ]))

    return model_name
