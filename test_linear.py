# test_strassen.py
import os
import torch
from model import TensorModel
from alpha import AlphaZero
from tensor_game import TensorGame


def load_latest_checkpoint(models_path: str) -> str:
    """Restituisce il percorso dell’ultimo modello model_iter*.pt."""
    files = [f for f in os.listdir(models_path)
             if f.startswith("model_iter") and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"Nessun checkpoint in {models_path}")
    iters = sorted(int(f.split("model_iter")[-1].split(".pt")[0]) for f in files)
    return os.path.join(models_path, f"model_iter{iters[-1]}.pt")


def test_strassen(models_path: str, args: dict, search_games: int = 10):
    # 1) gioco e modello
    game = TensorGame(args)
    model = TensorModel(dim_3d=4, dim_t=args["R_limit"], dim_s=1, dim_c=16,n_steps=12, n_logits=3, n_samples=4, device=args["device"]).to(args["device"])

    # 2) carica l’ultimo checkpoint (compatibile con GNN)
    ckpt = load_latest_checkpoint(models_path)
    print(f"🔄  Carico checkpoint: {ckpt}")
    sd = torch.load(ckpt, map_location=args["device"])
    model.load_state_dict(sd, strict=False)       # ← chiavi extra ignorate
    model.eval()

    # 3) AlphaZero solo inferenza
    alpha = AlphaZero(model, optimizer=None, game=game, args=args)

    # 4) self-play ripetuti
    successes = 0
    print("Inizio a giocare...")
    for i in range(1, search_games + 1):
        traj = alpha.selfPlay()
        final_reward = traj[-1][-1].item()
        solved = (final_reward == 0)
        print(f"[{i:02}/{search_games}] mosse={len(traj):2d}  "
              f"reward={final_reward:3}  solved={solved}")
        successes += solved

    print(f"\n✅  Strassen risolto in {successes}/{search_games} run")


# ------------------------------------------------------------
if __name__ == "__main__":
    # stessi hyper-parametri usati nel training
    args = {
        "C": 2,
        "num_searches": 50,
        "num_iterations": 10,
        "num_selfPlay_iterations": 10,
        "num_epochs": 10,
        "num_epochs_pretrain": 2,
        "batch_size": 32,
        "models_path": "models",
        "device": "mps",#"cuda" if torch.cuda.is_available() else "cpu",
        "T": 0,
        "R_limit": 8,
    }

    test_strassen(models_path=args["models_path"], args=args, search_games=20)
