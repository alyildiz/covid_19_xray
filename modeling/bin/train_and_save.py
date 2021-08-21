import torch
from pytorch_lightning import Trainer
from src.constants import SAVE_CHECKPOINTS_PATH
from src.load_data import get_data
from src.pl_datamodule import Covid19DataModule
from src.pl_model import LitModel


def main():
    train_x, val_x, test_x, train_y, val_y, test_y = get_data()
    data_module = Covid19DataModule(
        batch_size=16, train_x=train_x, val_x=val_x, test_x=test_x, train_y=train_y, val_y=val_y, test_y=test_y
    )
    model = LitModel(input_shape=(3, 224, 224), num_classes=3)
    trainer = Trainer(max_epochs=20, gpus=torch.cuda.device_count(), log_every_n_steps=5)
    trainer.fit(model, data_module)
    trainer.save_checkpoint(SAVE_CHECKPOINTS_PATH)


if __name__ == "__main__":
    main()
