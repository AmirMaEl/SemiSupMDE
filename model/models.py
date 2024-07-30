
def create_model(model, accelerator):
    print(model)
    if model == 'test':
        from .test_model import TestModel
        model = TestModel()  
    elif model == 'train':
        from .task_model import TNetModel
        model = TNetModel()
    elif model == 'traint2':
        from .t2model import T2NetModel
        model = T2NetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(accelerator=accelerator)
    print("model [%s] was created." % (model.name()))
    return model
