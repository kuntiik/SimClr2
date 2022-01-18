import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning import LightningModule

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = self.init_model()
        self.projection = Projection(input_dim=self.hparams.hidden_mlp, hidden_dim=self.hparams.hidden_mlp, output_dim=self.hparams.feat_dim)

        global_batch_size = self.hparams.num_nodes * self.hparams.gpus * self.hparams.batch_size if self.hparams.gpus > 0 else self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def init_model(self):
        if self.hparams.arch == "resnet18":
            backbone = resnet18
        elif self.hparams.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.hparams.first_conv, maxpool1=self.hparams.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        return self.encoder(x)[-1]

    def shared_step(self, batch):
        if self.hparams.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.hparams.weight_decay)
        else:
            params = self.parameters()

        if self.hparams.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.hparams.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def _get_negative_mask(self, sim, out_1, out_1_dist, out_2):
        negative_mask = torch.ones_like(sim, dtype=torch.bool, device=self.device)
        for i in range(out_1.shape[0]):
            negative_mask[i, i] = False
        for i in range(out_2.shape[0]):
            negative_mask[i + out_1.shape[0], i + out_1_dist.shape[0]] = False
        return negative_mask

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        # neg = sim.sum(dim=-1)
        # row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        # neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        #TODO verify this
        neg = (sim * self._get_negative_mask(sim, out_1, out_1_dist, out_2)).sum(dim=-1)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss