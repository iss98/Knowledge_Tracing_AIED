from torch.nn.modules import PairwiseDistance
from tqdm import  tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy


from cfg import get_cfg
from preprocess import get_ds
from model import get_model
 

if __name__ == "__main__":
    cfg = get_cfg()
    wandb.init(project=cfg.prjname, name = f"{cfg.dscor}_{cfg.dsitem}_{cfg.dstime}_{cfg.model}_{cfg.addds}",config=cfg)

    ds = get_ds(cfg.model, cfg.addds)(cfg)
    if ds.max_item != cfg.item :
        print("number of item does not match")
        raise ValueError

    test_len = round(len(ds)*cfg.testsplit)
    val_len = round((len(ds) - test_len) * cfg.valsplit)
    tv_ds, test_ds = random_split(ds, [len(ds)-test_len, test_len])
    train_ds, val_ds = random_split(tv_ds, [len(ds)-test_len-val_len, val_len])

    train_dl = DataLoader(train_ds, batch_size = cfg.bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size = cfg.bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size = cfg.bs, shuffle=True)

    model = get_model(cfg.model)(ds.input_size, cfg)
    model.to(cfg.device)

    optimizer = optim.SGD(model.parameters(), lr = cfg.lr, momentum = cfg.mt)
    scheduler = ReduceLROnPlateau(optimizer, mode = "max", factor = cfg.fac, patience = cfg.pat, threshold = cfg.thr)
    
    loss_fn = nn.BCELoss().to(cfg.device)
    AUC_fn = AUROC(task="binary")
    ACC_fn = BinaryAccuracy(threshold = 0.5)
    
    test_every = cfg.testevery

    best_auc = 0
    fname = f"save/{cfg.dscor}_{cfg.dsitem}_{cfg.dstime}_{cfg.model}_{cfg.addds}.pt"

    for ep in tqdm(range(cfg.epoch)):
        model.train()
        loss_ep = []
        #train
        for batch in tqdm(train_dl):
            
            iir = batch["input_interaction"].to(cfg.device)
            if cfg.model != "DKT":
                iit = batch["input_item"].to(cfg.device)
            if cfg.model == "TRKT":
                temb = batch["input_text_emb"].to(cfg.device)
            tit = batch["target_item"].to(cfg.device)
            trp = batch["target_response"].to(cfg.device)
            
            optimizer.zero_grad()
            if cfg.model == "DKT":
                pred = model(iir)
            elif cfg.model == "TRKT":
                pred = model(iit, iir, temb)
            else:
                pred = model(iit, iir)
            
            pred = model.posterior_predict(pred, tit)
            loss = loss_fn(pred, trp)

            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            if cfg.model != "DKT":
              del iit
            del iir, tit, trp, pred, loss
        
        auc_ep = []
        acc_ep = []
        #validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dl):

                iir = batch["input_interaction"].to(cfg.device)
                if cfg.model != "DKT":
                    iit = batch["input_item"].to(cfg.device)
                if cfg.model == "TRKT":
                    temb = batch["input_text_emb"].to(cfg.device)
                tit = batch["target_item"].to(cfg.device)
                trp = batch["target_response"].detach().cpu()

                if cfg.model == "DKT":
                    pred = model(iir)
                elif cfg.model == "TRKT":
                    pred = model(iit, iir, temb)
                else:
                    pred = model(iit, iir)

                pred = model.posterior_predict(pred, tit).detach().cpu()
                auc = AUC_fn(pred, trp)
                acc = ACC_fn(pred, trp)

                auc_ep.append(auc)
                acc_ep.append(acc)
                if cfg.model != "DKT":
                  del iit
                del iir, tit, trp, pred, auc, acc

        scheduler.step(np.mean(auc_ep))
        

        if (ep + 1) % test_every == 0:
            model.eval()
            t_auc = []
            t_acc = []
            with torch.no_grad():
                for batch in tqdm(test_dl):

                    iir = batch["input_interaction"].to(cfg.device)
                    if cfg.model != "DKT":
                        iit = batch["input_item"].to(cfg.device)
                    if cfg.model == "TRKT":
                        temb = batch["input_text_emb"].to(cfg.device)
                    tit = batch["target_item"].to(cfg.device)
                    trp = batch["target_response"].detach().cpu()

                    if cfg.model == "DKT":
                        pred = model(iir)
                    elif cfg.model == "TRKT":
                        pred = model(iit, iir, temb)
                    else:
                        pred = model(iit, iir)

                    pred = model.posterior_predict(pred, tit).detach().cpu()
                    auc = AUC_fn(pred, trp)
                    acc = ACC_fn(pred, trp)

                    t_auc.append(auc)
                    t_acc.append(acc)
                    if cfg.model != "DKT":
                      del iit
                    del iir, tit, trp, pred, auc, acc

            wandb.log({"test_acc": np.mean(t_acc), "test_auc": np.mean(t_auc)},commit=False)
            if np.mean(t_auc) > best_auc :
                wandb.run.summary["best_auc"] = np.mean(t_auc)
                best_auc = np.mean(t_auc)
                torch.save(model.state_dict(), fname)
        
        wandb.log({"loss": np.mean(loss_ep), "acc": np.mean(acc_ep), "auc": np.mean(auc_ep), "ep": ep})