def generate_model_name(config):

    # Extract relevant fields from the configuration with default fallback to 'NA'
    basemodel = config.get("BASEMODEL", {})
    advancedmodel = config.get("ADVANCEDMODEL", {})
    augmentation = config.get("AUGMENTATION", {})
    data = config.get("DATA", {})
    optimizer = config.get("OPTIMIZER", {})
    regularization = config.get("REGULARIZATION", {})
    scheduler = config.get("SCHEDULER", {})

    # Helper function to safely get a value or default to 'NA'
    def safe_get(dictionary, key, default="NA"):
        return dictionary.get(key, default)

    # Build the model name
    model_name = (
        f"{safe_get(basemodel, 'Backbone')}"
        f"_BS{safe_get(basemodel, 'Batch_Size')}"
        f"_{safe_get(basemodel, 'Loss_Function', '')[:2]}Loss"
        f"_{safe_get(basemodel, 'Precision')}"
        f"_{str(safe_get(basemodel, 'Target_Pixel_Size')).replace('.', 'p')}um"
        f"_{safe_get(basemodel, 'Target_Patch_Size', ['NA'])[0]}px"
        f"_E{safe_get(advancedmodel, 'Max_Epochs')}"
        f"_{'pretrain' if safe_get(advancedmodel, 'Pretrained') else 'nopretrain'}"
        f"_seed{safe_get(advancedmodel, 'Random_Seed')}"
        f"_sigma{str(safe_get(augmentation, 'Colour_Sigma')).replace('.', 'p')}"
        f"_T{str(safe_get(data, 'Train_Size')).replace('.', 'p')}"
        f"_V{str(safe_get(data, 'Val_Size')).replace('.', 'p')}"
        f"_eps{safe_get(optimizer, 'eps')}"
        f"_lr{safe_get(optimizer, 'lr')}"
        f"_fold{safe_get(data, 'Fold', ['NA'])[0]}"
        f"_ls{str(safe_get(regularization, 'Label_Smoothing')).replace('.', 'p')}"
        f"_{safe_get(scheduler, 'type')}"
    )

    return model_name