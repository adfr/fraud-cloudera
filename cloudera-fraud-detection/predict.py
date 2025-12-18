import cml.models_v1 as models

@models.cml_model
def predict(args):
    return {"message": "hello", "input": args}
