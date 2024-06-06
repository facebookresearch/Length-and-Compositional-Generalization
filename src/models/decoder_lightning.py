import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import hydra
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Subset
import collections.abc
import src.metrics
import os
import jsonlines
import json
import torch
from omegaconf import OmegaConf
from src.utils.metrics import pad_label_label
import numpy as np
from torch.nn import MSELoss
import torchmetrics
import code
import gc
from sklearn.linear_model import LinearRegression
# from memory_profiler import profile

class DecoderLightningModule(LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["datamodule",])
                
        # if loading a pretrained model, but need to change some of the parameters
        if self.hparams.get('substitute_config'):
            self._update_params(self.hparams, self.hparams.substitute_config)

        # print(self.hparams)
        # recursively replace any key in self.hparams python dictionary that contains encoder with decoder
        # for key in list(self.hparams.keys()):
        #     if "encoder" in key:
        #         new_key = key.replace("encoder", "decoder")
        #         self.hparams[new_key] = self.hparams.pop(key)
        #         # do this for key and values of this key as well
        #         for k,v in self.hparams[new_key].items():
        #             if "encoder" in str(k) or "encoder" in str(v):
        #                 new_k = k.replace("encoder", "decoder")
        #                 import omegaconf
        #                 if not isinstance(v, omegaconf.dictconfig.DictConfig):
        #                     new_v = v.replace("encoder", "decoder")
        #                 else:
        #                     new_v = v
        #                 self.hparams[new_key][new_k] = new_v

        #                 print(new_k, new_v)
        # if "encoder_config" in self.hparams["set_decoder"].keys():
        #     self.hparams["set_decoder"]["decoder_config"] = self.hparams["set_decoder"].pop("encoder_config")
        # print(self.hparams)
        # print(self.hparams.set_decoder)
        self.batch_size = self.hparams.dataset_parameters.batch_size
        self.set_decoder = hydra.utils.instantiate(self.hparams.set_decoder, _recursive_ = False)
        # self.model.load_state_dict(torch.load(ckpt_path))
        self.training_output = []
        self.testing_output = []
        self.validation_output = []
        self.log_to_terminal = True


    def _update_params(self, params, new_params):
        # for when you load pretrained model and want to update the params, 
        for k, v in new_params.items():
            if isinstance(v, collections.abc.Mapping):
                params[k] = self._update_params(params.get(k, {}), v)
            else:
                params[k] = v
        return params

    def forward(self, batch, stage='train'):

        x_set = batch['input'].squeeze(0)
        y_set = batch['output'].squeeze(0)
        masks = batch['mask'].squeeze(0)
        outputs = self.set_decoder(x_set=x_set, mask=masks, y_set=y_set) # ['loss', 'y_pred']

        # self.log(name=f'{stage}/loss', value=outputs["loss"], batch_size=self.batch_size, prog_bar=True, sync_dist=True)  

        return outputs       
    

    def on_train_start(self):

        # pass the whole dataset through the model and save the hidden representations
        # for each sequence in the dataset
        hidden_gts = []
        hidden_preds = []
        for batch in self.trainer.datamodule.val_dataloader():
            # batch = next(iter(self.trainer.datamodule.train_dataloader())) # {'input', 'output', 'mask', 'hidden'}
            hidden_gt = batch["hidden"].squeeze(0) # [batch_size, seq_max_length, hidden_dim]
            hidden_pred = self.set_decoder(batch["input"].squeeze(0).to(self.device), batch["mask"].squeeze(0).to(self.device), batch["output"].squeeze(0).to(self.device))["hidden"] # [batch_size, seq_max_length, hidden_dim]
            hidden_gts.append(hidden_gt)
            hidden_preds.append(hidden_pred)
        hidden_gt = torch.cat(hidden_gts, dim=0) # [num_batches x batch_size, seq_max_length, hidden_dim]
        hidden_pred = torch.cat(hidden_preds, dim=0) # [num_batches x batch_size, seq_max_length, hidden_dim]
        
        r2_scores = []
        for j in range(hidden_gt.shape[1]):
            reg = LinearRegression(fit_intercept=True).fit(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy())
            r2_scores.append(reg.score(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy()))

        self.log(name="val/r2_score_start", value=np.mean(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="val/r2_score_std_start", value=np.std(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="val/r2_score_min_start", value=np.min(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="val/r2_score_max_start", value=np.max(r2_scores), prog_bar=True, sync_dist=True)

        # print the scores to the termnial
        if self.log_to_terminal:
            print("======= Starting Scores =========")
            print(f"r2_scores: {r2_scores}")
            print(f"mean r2_score: {np.mean(r2_scores)}")
            print(f"std r2_score: {np.std(r2_scores)}")
            print(f"min r2_score: {np.min(r2_scores)}")
            print(f"max r2_score: {np.max(r2_scores)}")


        # write the tensors to pt files for later use
        torch.save(hidden_gt, "hidden_gt_start.pt")
        torch.save(hidden_pred, "hidden_pred_start.pt")
            
    def training_step(self, batch, batch_idx):

        # batch: {'input', 'output', 'mask'}
        outputs = self(batch) # ['loss', 'y_pred']
        loss = outputs["loss"]

        self.log(name=f'train/loss', value=outputs["loss"], batch_size=self.batch_size, prog_bar=False, sync_dist=True)
        self.training_output.append({"y_true": batch["output"].squeeze(0)})

        # every 200 batches print 10 samples of the batch along with the model predictions
        # outputs["y_pred"]: batch_size x seq_max_length-1 x y_dim
        # batch["output"]: batch_size x seq_max_length-1 x y_dim
        if batch_idx % 200 == 0:
            k = 10
            outputs_ = batch["output"].squeeze(0)[:k] # k x seq_max_length-1 x y_dim
            masks_ = batch["mask"].squeeze(0)[:k] # k x seq_max_length-1 x seq_max_length
            lengths = batch["lengths"].squeeze(0)[:k] # k
            pred_outputs_ = outputs["y_pred"][:k]
            # inputs.reshape(-1, inputs.shape[-1])[seq_mask.reshape(-1) == 1].shape
            if self.log_to_terminal:
                print(f"\n================== Train ==================")
                for i in range(len(outputs_)):
                    if masks_[i].sum() == 0:
                        continue
                    # outputs_[i]: seq_max_length-1 x y_dim, filter those that are all zeros in the y_dim
                    non_zero_indices = outputs_[i].sum(dim=1) != 0
                    print(f"\nlength {lengths[i]} ground truth, prediction:\n{torch.round(outputs_[i, non_zero_indices], decimals=3)}\n{torch.round(pred_outputs_[i, non_zero_indices], decimals=3)}")
                    print("")

        return loss

    
        
    def on_train_epoch_end(self):

        # compute the baseline loss. Baseline loss uses the mse loss between the ground truth and the mean of the ground truth
        y_true = torch.cat([training_output["y_true"] for training_output in self.training_output], dim=0) # [num_epochs x batch_size, seq_max_length, y_dim]
        y_mean = y_true.mean(dim=0, keepdim=True).repeat(y_true.shape[0], 1, 1) # [1, seq_max_length, y_dim]
        losses = []
        for i in range(y_true.shape[1]):
            losses.append(MSELoss()(y_mean[:, i], y_true[:, i]))
        losses = torch.stack(losses)
        loss = losses.mean()
        self.log(name="train/baseline_loss", value=loss, prog_bar=True, sync_dist=True)
        
        return


    def validation_step(self, batch, batch_idx):

        outputs = self(batch, stage='val') # ['loss', 'y_pred']

        self.log(name=f'val/loss', value=outputs["loss"], batch_size=self.batch_size, prog_bar=False, sync_dist=True)  

        # evaluate the linear relationship between hidden representations learned by the model and the ground truth
        hidden_gt = batch["hidden"].squeeze(0) # [batch_size, seq_max_length, hidden_dim]
        hidden_pred = outputs["hidden"] # [batch_size, seq_max_length, hidden_dim]
        # mask = batch["mask"].squeeze(0) # [batch_size, seq_max_length, seq_max_length]
        # non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float() # [batch_size, seq_max_length, 1]
        # indices_bool = non_zero_mask.squeeze().bool() # [batch_size, seq_max_length]
        # r2_scores = []
        # for j in range(hidden_gt.shape[1]):
        #     reg = LinearRegression(fit_intercept=True).fit(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy())
        #     r2_scores.append(reg.score(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy()))

        # self.log(name="val/r2_score", value=np.mean(r2_scores), prog_bar=True, sync_dist=True)
        self.validation_output.append({"hidden_gt": hidden_gt, "hidden_pred": hidden_pred, "y_true": batch["output"].squeeze(0)})

        # every 200 batches print 10 samples of the batch along with the model predictions
        if batch_idx % 500 == 0 and not self.trainer.state.stage == "sanity_check":
            k = 10
            outputs_ = batch["output"].squeeze(0)[:k]
            masks_ = batch["mask"].squeeze(0)[:k]
            lengths = batch["lengths"].squeeze(0)[:k] # k
            pred_outputs_ = outputs["y_pred"][:k]
            print(f"\n================== Val ==================")
            with jsonlines.open("val_predictions.jsonl", mode="a") as writer:
                writer.write(f"======================== epoch: {self.trainer.current_epoch} ========================")
                for i in range(len(outputs_)):
                    if masks_[i].sum() == 0:
                        continue
                    # non_zero_indices = outputs_[i].sum(dim=1) != 0
                    # the above breaks when outputs_[i] has -1,1 values, instead we should check if 
                    # all entries are zero or not, that is robust
                    non_zero_indices = torch.eq(outputs_[i], 0).sum(dim=1) != outputs_[i].shape[1]

                    if self.log_to_terminal:
                        print(f"\nlength {lengths[i]} ground truth, prediction:\n{torch.round(outputs_[i, non_zero_indices], decimals=3)}\n{torch.round(pred_outputs_[i, non_zero_indices], decimals=3)}")
                        print("=========================================")
                    writer.write(f"length {lengths[i]}")
                    for j in range(lengths[i]):
                        writer.write(f"------ {j} ------")
                        writer.write(f"gtout: {[round(num, 5) for num in outputs_[i, non_zero_indices].tolist()[j]]}\n")
                        writer.write(f"preds: {[round(num, 5) for num in pred_outputs_[i, non_zero_indices].tolist()[j]]}\n")
                    writer.write("=========================================")


        return outputs["loss"]
    
    def test_step(self, batch, batch_idx):

        outputs = self(batch, stage='test') # ['loss', 'y_pred']
        hidden_gt = batch["hidden"].squeeze(0) # [batch_size, seq_max_length, hidden_dim]
        hidden_pred = outputs["hidden"] # [batch_size, seq_max_length, hidden_dim]

        # every 200 batches print 10 samples of the batch along with the model predictions
        if batch_idx % 500 == 0:
            k = 10
            outputs_ = batch["output"].squeeze(0)[:k]
            masks_ = batch["mask"].squeeze(0)[:k]
            lengths = batch["lengths"].squeeze(0)[:k] # k
            pred_outputs_ = outputs["y_pred"][:k]
            print(f"\n================== Val ==================")
            with jsonlines.open("val_predictions.jsonl", mode="a") as writer:
                writer.write(f"======================== epoch: {self.trainer.current_epoch} ========================")
                for i in range(len(outputs_)):
                    if masks_[i].sum() == 0:
                        continue
                    non_zero_indices = outputs_[i].sum(dim=1) != 0
                    if self.log_to_terminal:
                        print(f"\nlength {lengths[i]} ground truth, prediction:\n{torch.round(outputs_[i, non_zero_indices], decimals=3)}\n{torch.round(pred_outputs_[i, non_zero_indices], decimals=3)}")
                        print("=========================================")
                    writer.write(f"gtout: {[round(num, 5) for num in outputs_[i, non_zero_indices].tolist()[0]]}\n")
                    writer.write(f"preds: {[round(num, 5) for num in pred_outputs_[i, non_zero_indices].tolist()[0]]}\n")
                    writer.write("=========================================")

        self.testing_output.append({"hidden_gt": hidden_gt, "hidden_pred": hidden_pred, "y_true": batch["output"].squeeze(0), "loss": outputs["loss"]})

        return outputs["loss"]
    

    # @profile
    def on_validation_epoch_end(self):
        
        # self.validation_output contains tensor of hidden_gt and hidden_pred of size [batch_size, seq_max_length, hidden_dim]
        # evaluate the linear relationship between hidden representations learned by the model and the ground truth
        hidden_gt = torch.cat([val_output["hidden_gt"] for val_output in self.validation_output], dim=0) # [num_epochs x batch_size, seq_max_length, hidden_dim]
        hidden_pred = torch.cat([val_output["hidden_pred"] for val_output in self.validation_output], dim=0) # [num_epochs x batch_size, seq_max_length, hidden_dim]
        r2_scores = []
        for j in range(hidden_gt.shape[1]):
            reg = LinearRegression(fit_intercept=True).fit(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy())
            r2_scores.append(reg.score(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy()))

        self.log(name="val/r2_score", value=np.mean(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="val/r2_score_std", value=np.std(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="val/r2_score_min", value=np.min(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="val/r2_score_max", value=np.max(r2_scores), prog_bar=True, sync_dist=True)

        # compute the baseline loss. Baseline loss uses the mse loss between the ground truth and the mean of the ground truth
        y_true = torch.cat([val_output["y_true"] for val_output in self.validation_output], dim=0) # [num_epochs x batch_size, seq_max_length, y_dim]
        y_mean = y_true.mean(dim=0, keepdim=True).repeat(y_true.shape[0], 1, 1) # [1, seq_max_length, y_dim]
        losses = []
        for i in range(y_true.shape[1]):
            losses.append(MSELoss()(y_mean[:, i], y_true[:, i]))
        losses = torch.stack(losses)
        loss = losses.mean()
        self.log(name="val/baseline_loss", value=loss, prog_bar=True, sync_dist=True)

        # every 5 epochs write the hidden representations to pt files for future use. Use the current epoch in the name
        if self.trainer.current_epoch % 5 == 0:
            torch.save(hidden_gt, f"hidden_gt_{self.trainer.current_epoch}.pt")
            torch.save(hidden_pred, f"hidden_pred_{self.trainer.current_epoch}.pt")


        self.validation_output = []

        return
    
    def on_test_epoch_end(self):

        # compute the mean of the losses gathered from the test step
        losses = [test_output["loss"] for test_output in self.testing_output] # [num_batches]
        loss = torch.tensor(losses).mean()
        self.log(name="test/loss", value=loss, prog_bar=True, sync_dist=True)

        # self.testing_output contains tensor of hidden_gt and hidden_pred of size [batch_size, seq_max_length, hidden_dim]
        # evaluate the linear relationship between hidden representations learned by the model and the ground truth
        hidden_gt = torch.cat([test_output["hidden_gt"] for test_output in self.testing_output], dim=0) # [num_epochs x batch_size, seq_max_length, hidden_dim]
        hidden_pred = torch.cat([test_output["hidden_pred"] for test_output in self.testing_output], dim=0) # [num_epochs x batch_size, seq_max_length, hidden_dim]
        r2_scores = []
        for j in range(hidden_gt.shape[1]):
            reg = LinearRegression(fit_intercept=True).fit(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy())
            r2_scores.append(reg.score(hidden_pred[:, j, :].detach().cpu().numpy(), hidden_gt[:, j, :].detach().cpu().numpy()))

        self.log(name="test/r2_score", value=np.mean(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="test/r2_score_std", value=np.std(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="test/r2_score_min", value=np.min(r2_scores), prog_bar=True, sync_dist=True)
        self.log(name="test/r2_score_max", value=np.max(r2_scores), prog_bar=True, sync_dist=True)

        # compute the baseline loss. Baseline loss uses the mse loss between the ground truth and the mean of the ground truth
        y_true = torch.cat([test_output["y_true"] for test_output in self.testing_output], dim=0) # [num_epochs x batch_size, seq_max_length, y_dim]
        y_mean = y_true.mean(dim=0, keepdim=True).repeat(y_true.shape[0], 1, 1) # [1, seq_max_length, y_dim]
        losses = []
        for i in range(y_true.shape[1]):
            losses.append(MSELoss()(y_mean[:, i], y_true[:, i]))
        losses = torch.stack(losses)
        loss = losses.mean()
        self.log(name="test/baseline_loss", value=loss, prog_bar=True, sync_dist=True)

        self.testing_output = []

        return        

    def configure_optimizers(self):

        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)

        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                , params
                                                                )
        
        if self.hparams.get("scheduler_config"):
            # for pytorch scheduler objects, we should use utils.instantiate()
            if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
                scheduler = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer)

            # for transformer function calls, we should use utils.call()
            elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
                scheduler = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer)
            
            else:
                raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
                
            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict["scheduler"] = scheduler

            return [optimizer], [scheduler_dict]
        else:
            # no scheduling
            return [optimizer]
