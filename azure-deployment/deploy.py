import os
import uuid

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model

model_dir = "model_artifacts"
service_name = "houses-price-predictor-service"

env_file_path = os.path.join(os.path.dirname(__file__), "conda_env.yaml")
config_file_path = os.path.join(os.path.dirname(__file__), "config.json")
model_path = os.path.join(os.path.dirname(__file__), "..", model_dir)
features_path = os.path.join(os.path.dirname(__file__), "..", "feature_defs.pkl")
model_card_path = os.path.join(os.path.dirname(__file__), "..", "model_card.json")


ws = Workspace.from_config(path=config_file_path)
env = Environment.from_conda_specification(name="pycaret-env", file_path=env_file_path)
model = Model.register(
    workspace=ws,
    model_path=model_path,
    model_name="houses-price-predictor-model",
)

inference_config = InferenceConfig(entry_script="score.py", environment=env)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=4)

service = Model.deploy(
    workspace=ws,
    name=service_name,
    overwrite=True,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
print(service.get_logs())
