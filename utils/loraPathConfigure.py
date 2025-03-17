class pathConfigurator:
    def __init__(self):
        pass
    def getPath(self,config,pkg,version):
        # module,layer,r,alpha
        module = config["target_modules"]
        layer = config["target_layers"]
        r = config["r"]
        alpha = config["alpha"]
        knowledge_type = config["knowledge_type"]
        learning_rate = config["learning_rate"]
        epoch = config["num_epochs"]    
        module_str = "_".join(str(m) for m in module)
        layer_str = "_".join(str(l) for l in layer)
        path = f"{config["save_path_base"]}/lora_models/{pkg}/{pkg}_{version}_{knowledge_type}_{module_str}_{layer_str}_{r}_{alpha}_{learning_rate}_{epoch}.pth"
        return path