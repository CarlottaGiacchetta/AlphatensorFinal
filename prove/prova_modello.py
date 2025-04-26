from model import TensorModel
import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataset import SyntheticDemoDataset, StrassenAugDataset

model = TensorModel(dim_3d=4, dim_t=8, dim_s=1, dim_c=16, n_steps=12, n_logits=3, n_samples=4, device="mps").to("mps")


# Esempio di concatenazione
ds = ConcatDataset([
    SyntheticDemoDataset(5, 8, "cpu"),
    StrassenAugDataset(100, 8, "cpu"),
])
dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
for st, sc, tok, rew in dl:
    st = st.to("mps")
    sc = sc.to("mps")
    tok= tok.to("mps")
    rew= rew.to("mps")

# training loop d’esempio
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(20):
    for st, sc, tok, rew in dl:
        st  = st.to("mps")
        sc  = sc.to("mps")
        tok = tok.to("mps")
        rew = rew.to("mps")

        pol_loss, val_loss = model.fwd_train(st, sc, tok, rew)
        loss = pol_loss + val_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"E{epoch:02d}  loss={loss.item():.3f}")
