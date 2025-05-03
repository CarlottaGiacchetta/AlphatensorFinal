from mcts import MCTS
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from tqdm import trange, tqdm
import os
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt
#from torch.cuda.amp import GradScaler, autocast
from torch import amp


class AlphaZero:

    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.use_amp = args.get("use_amp", False) and args["device"].startswith("cuda")
        print(self.use_amp)
        self.scaler = amp.GradScaler(device ='cuda', enabled=self.use_amp)   # <-- NUOVO per fare ottimizzazione
        if self.use_amp:
           torch.set_float32_matmul_precision("medium")


    def selfPlay(self):
        memory = []
        state = self.game.get_initial_state()

        while True: # fino a quando non finisce
            
            best_action_tokens = self.mcts.search(state)

            stato_encoded = self.game.get_encoded_state(state)
            scalars = self.game.get_scalar()       
            memory.append((stato_encoded, scalars, best_action_tokens))

            # lo stato cambia! --> perform_action
            state = self.game.perform_action(best_action_tokens)
            reward, done = self.game.get_value_and_terminated(state)
            
            if done:
                final_reward = reward
                break
        
        traiettorie = []
        for st_enc, sc, tokens in memory:
            traiettorie.append((st_enc, sc, tokens, final_reward))

        return traiettorie
    
    def train(self, memory):
        random.shuffle(memory)
        B = self.args["batch_size"]
        batches = list(range(0, len(memory), B))
        print(f"lunghezza memoria: {len(memory)}")
        for i in tqdm(batches, desc="Training batches"):
            batch = memory[i : i + B]
            states, scalars, actions, values = zip(*batch)

            # 1) batchizza
            xx = torch.stack(states).to(self.model.device).float()  # (B, dim_t, S, S, S)
            ss = torch.stack(scalars).to(self.model.device).float()  # (B, 1)
            gg = torch.stack(actions).to(self.model.device).long()   # (B, n_steps)
            vv = torch.tensor(values,dtype=torch.float32, device=self.model.device).view(-1,1) # (B) → (B,1)

            # ──────────────────────────────────────────────────────
            # Forward & Loss (mixed precision)
            # ──────────────────────────────────────────────────────
            with amp.autocast(enabled=self.use_amp, device = self.model.device):
                pol_loss, val_loss = self.model.fwd_train(xx, ss, gg, vv)
                loss = pol_loss + val_loss

            # Back‑prop + Opt step
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()                 

            '''OLD VERSIONE -> NOT OPTMIZED
            # 2) forward + compute losses
            pol_loss, val_loss = self.model.fwd_train(xx, ss, gg, vv)
            loss = pol_loss + val_loss
            print(f"Loss loop interno selfplay: {loss}")
            # 3) backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()'''

    def _save_pt_and_plot(self, losses: list[float], tag: str):
        """
        Salva:
        • checkpoint del modello  → models/{tag}.pt
        • loss vector             → figures/{tag}.npy
        • plot PNG                → figures/{tag}.png
        """
        os.makedirs(self.args["models_path"], exist_ok=True)
        os.makedirs("figures",                 exist_ok=True)

        # --- checkpoint ---
        ckpt_path = os.path.join(self.args["models_path"], f"{tag}.pt")
        torch.save(self.model.state_dict(), ckpt_path)

        # --- vettore loss ---
        losses_arr = np.asarray(losses, dtype=np.float32)
        npy_path   = os.path.join("figures", f"{tag}.npy")
        np.save(npy_path, losses_arr)

        # --- plot PNG ---
        plt.figure()
        plt.plot(losses_arr)
        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.title(f"{tag} loss")
        plt.tight_layout()
        png_path = os.path.join("figures", f"{tag}.png")
        plt.savefig(png_path)
        plt.close()

        print(f"✓ checkpoint  {ckpt_path}")
        print(f"✓ vettore     {npy_path}")
        print(f"✓ grafico     {png_path}")


    def learn_pretrain(self, sup_datasets, pretrain_epochs=1, tag="pretrain_loss"):

        if pretrain_epochs == 0:
            print("Skippo pretrain.. già presente")
            return
        sup_ds = ConcatDataset(sup_datasets)
        loader = DataLoader(
            sup_ds,
            batch_size=self.args["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,                  # single-process ⇒ no mmap
            pin_memory=True,
            persistent_workers=False,
        )
        #if torch.__version__ >= "2.0":
        #    self.model = torch.compile(self.model)
        print(f"=== Pre-training supervisionato: {pretrain_epochs} epoche su {len(sup_ds)} campioni ===")
        self.model.train()

        batch_losses = []
        for ep in range(pretrain_epochs):
            epoch_loss = 0.0
            for st, sc, tok, rew in tqdm(loader, desc=f"Epoca {ep+1}/{pretrain_epochs}", leave=False):
                st, sc, tok, rew = (
                    st.to(self.model.device),
                    sc.to(self.model.device),
                    tok.to(self.model.device),
                    rew.to(self.model.device),
                )
                with amp.autocast(device_type = "cuda", enabled=self.use_amp):
                    pol_loss, val_loss = self.model.fwd_train(st, sc, tok, rew)
                    loss = pol_loss + val_loss
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                '''OLD VERISON
                pol_loss, val_loss = self.model.fwd_train(st, sc, tok, rew)
                loss = pol_loss + val_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()'''
                epoch_loss += loss.item()
            print(f"loss medio = {epoch_loss/len(loader):.4f}")
            batch_losses.append(epoch_loss/len(loader))
        self._save_pt_and_plot(batch_losses, tag)
    
    def learn(self, sup_datasets: list[Dataset], pretrain_epochs: int = 5):
        # 1) Supervisd pre‑training
        sup_ds = ConcatDataset(sup_datasets)
        sup_loader = DataLoader(
            sup_ds,
            batch_size=self.args["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0, 
            pin_memory=True,
        )

        print(f"=== Pre‑training supervisionato: {pretrain_epochs} epoche su {len(sup_ds)} campioni ===")
        self.model.train()
        for ep in range(pretrain_epochs):
            epoch_loss = 0.0
            for st, sc, tok, rew in sup_loader:
                st, sc, tok, rew = (
                    st.to(self.model.device),
                    sc.to(self.model.device),
                    tok.to(self.model.device),
                    rew.to(self.model.device),
                )
                pol_loss, val_loss = self.model.fwd_train(st, sc, tok, rew)
                loss = pol_loss + val_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Pretrain epoca {ep+1}/{pretrain_epochs}  loss medio = {epoch_loss/len(sup_loader):.4f}")

        # 2) Self‑play + aggiornamento iterativo
        for iteration in range(self.args["num_iterations"]):
            # genera tutta la memoria via self‑play
            memory = []
            
            #! TODO: parallelo
            self.model.eval()
            for _ in trange(self.args["num_selfPlay_iterations"], desc="Self‑play"):
                memory += self.selfPlay()

            # allena la rete sulla memoria raccolta
            self.model.train()
            for ep in range(self.args["num_epochs"]):
                # il metodo train si aspetta un elenco di 4‑tuple e lo spacchetta correttamente
                self.train(memory)
                print(f"Iter {iteration}  Epoch {ep+1}/{self.args['num_epochs']}  memoria size = {len(memory)}")

            # salva il checkpoint
            os.makedirs(self.args["models_path"], exist_ok=True)
            torch.save(
                self.model.state_dict(),
                f"{self.args['models_path']}/model_iter{iteration}.pt",
            )
            torch.save(
                self.optimizer.state_dict(),
                f"{self.args['models_path']}/optim_iter{iteration}.pt",
            )

        print("=== Training completato ===")