import hydra 
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

def train(cfg : DictConfig):

    #instantiate DataModule
    dm = hydra.utils.instantiate(cfg.datamodule.data)
    dm.train_transforms = hydra.utils.instantiate(cfg.datamodule.train_transforms)
    dm.val_transforms = hydra.utils.instantiate(cfg.datamodule.val_transforms)

    model = hydra.utils.instantiate(cfg.module,num_samples = dm.num_samples )

    online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=cfg.module.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=cfg.datamodule.name
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint, online_evaluator, lr_monitor]



    logger = WandbLogger(project = "SimClr clustering")
    trainer = hydra.utils.instantiate(cfg.trainer, logger= logger, callbacks=callbacks)

    trainer.tune(model, dm)
    trainer.fit(model, dm)

    return True
