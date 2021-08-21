from src.constants import DEMO_MODEL_CKPT, class_dic
from src.pl_model import LitModel

from web_app.utils import model_inference, setup_annotation, setup_parameters

model = LitModel.load_from_checkpoint(checkpoint_path=DEMO_MODEL_CKPT)

image = setup_parameters()
out_image = image.copy()
proba = model_inference(model, out_image)

setup_annotation(proba, image)
