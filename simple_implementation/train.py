import torch
import torch.nn as nn
import argparse
import random

from data import CzechSpeechDS
from data import LogMelFeaturizer
from data import collate_function
from model import Decoder
from model import CNN
from torch.utils.data import DataLoader, Subset

from data import N_MELS, BLANK_ID

class Trainer:
    def __init__(self, model, vocab, epochs, lr, train_loader,val_loader=None, *, weight_decay=1e-2, warmup_frac=0.15,
                 grad_clip=5.0, device="cpu", last_path = "last.pt", ckpt_path="best.pt", blank = BLANK_ID,
                 evaluator=None):

        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            
        self.model = model.to(self.device)
        self.vocab = vocab
        
        self.best_score = float("inf")
        self.start_epoch = 0                  # 0 = fresh run; set by load(resume=True)
        self.last_path = last_path 
        
        self.train_loader = train_loader
        self.decoder = Decoder(vocab)
        
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.grad_clip = grad_clip
        self.blank = blank
        self.ckpt_path = ckpt_path
        self.evaluator = evaluator

        self.ctc = nn.CTCLoss(blank, zero_infinity=True)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
  
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.lr, total_steps=self.epochs * len(self.train_loader), pct_start=self.warmup_frac)
        # policy for changing the learning rate - The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate 
        # and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate
        # changes the learning rate after every batch
        # pct start =  The percentage of the cycle (in number of steps) spent increasing the learning rate.
         

    def train_epoch(self, epoch):
        
        self.model.train()
        total = 0.0
        
        n_ok = 0  # batches that actually updated the model
        n_skipped = 0  # batches skipped for a non-finite loss
        n_infeasible = 0  # clips CTC cannot align
        
        for features, feat_lens, target_tokens, target_lengths in self.train_loader:
            log_probabilities, output_lengths = self.model(features.to(self.device), feat_lens) # returns defined in the forward function in Model
            
            n_infeasible += (output_lengths.cpu() < target_lengths.cpu()).sum().item()
            
            loss = self.ctc(log_probabilities.transpose(0, 1), target_tokens, output_lengths, target_lengths)  
            
            if not torch.isfinite(loss):
                n_skipped += 1
                self.optimizer.zero_grad()
                self.sched.step()          # keep the LR schedule aligned with the step count
                continue
            
            self.optimizer.zero_grad() # resets the gradients before processing the next batch
            loss.backward()
            # prevents exploding gradients by scaling down all parameter gradients proportionally 
            # if their combined L2 norm exceeds 5.0
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) 
            
            self.optimizer.step()
            self.sched.step()
            total += loss.item()
            
        avg_loss = total / max(n_ok, 1)
        print(f"epoch {epoch}  train CTC loss {avg_loss:.3f} | "
          f"infeasible clips {n_infeasible} | skipped batches {n_skipped}")
        
        return avg_loss
        

    @torch.no_grad()
    def validate(self, show=3):
        
        if self.val_loader is None: return {}
         
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        n_skipped = 0
        n_infeasible = 0 
        w_err = w_tot = c_err = c_tot = 0
        shown = 0
        
        for features, f_lens, target_tokens, target_lengths in self.val_loader:
            log_probabilities, output_lengths = self.model(features.to(self.device), f_lens)
            
            n_infeasible += (output_lengths.cpu() < target_lengths.cpu()).sum().item()
            
            loss = self.ctc(log_probabilities.transpose(0, 1), target_tokens, output_lengths, target_lengths) 
            
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            
            total_loss += loss.item() 
            num_batches += 1
            
            refs = self.decoder.decode_targets(target_tokens, target_lengths)
            
            for k, ref in enumerate(refs):
                hypothesis = self.decoder.greedy_decode(log_probabilities[k].cpu(), output_lengths[k].item())
               
                w_err += self.edit_distance(ref.split(), hypothesis.split())
                w_tot += len(ref.split())
                c_err += self.edit_distance(ref, hypothesis)
                c_tot += len(ref)
                
                if shown < show:
                    print(f"  REF: {ref}\n  HYP: {hypothesis}\n")
                    shown += 1
        val_loss = total_loss / max(num_batches, 1)
        metrics = {
            "val_loss": val_loss,
            "wer": w_err / max(w_tot, 1),
            "cer": c_err / max(c_tot, 1),
        }
        
        print(f"val_loss: {metrics['val_loss']:.4f} | WER: {metrics['wer']:.2%}| CER: {metrics['cer']:.2%}  | infeasible clips {n_infeasible} | skipped batches {n_skipped}")
        return val_loss

    def edit_distance(self, a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]

    def fit(self):

        best_loss = float("inf")
        
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            loss = self.train_epoch(epoch)
            if (self.val_loader is not None):
                val_loss = self.validate()
            else:
                val_loss = loss
            
            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss
                self.save(self.ckpt_path, epoch) 
                
            self.save(self.last_path, epoch) 

    def save(self, path, epoch):

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.sched.state_dict(), 
            "epoch": epoch,
            "best_score": self.best_score,
            "vocab": self.vocab,
        }, path)

    def load(self, path, resume=False):

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        if checkpoint["vocab"] != self.vocab:
            raise ValueError("checkpoint vocabulary differs from the current vocab.json")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if resume:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.sched.load_state_dict(checkpoint["scheduler_state_dict"])
            self.best_score = checkpoint["best_score"]
            self.start_epoch = checkpoint["epoch"]
        
        return checkpoint["epoch"]

MODELS = {"cnn": CNN}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16) # batch size
    ap.add_argument("--lr", type=float, default=1e-3) # learning rate
    ap.add_argument("--overfit", type=int, default=0, help="train+eval on N utts")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    
    featurizer = LogMelFeaturizer()
    train_ds = CzechSpeechDS("train", featurizer)
    val_ds   = CzechSpeechDS("validation", featurizer)
    vocab = train_ds.vocab  
    
    if args.overfit:
        idx = random.Random(0).sample(range(len(train_ds)), args.overfit)
        train_ds = Subset(train_ds, idx)
        val_ds = train_ds
    
    
    train_loader = DataLoader(train_ds, batch_size=args.bs,num_workers=args.workers, shuffle = True, collate_fn=collate_function) 
    
    val_loader   = DataLoader(val_ds, batch_size=args.bs, num_workers=args.workers, shuffle=False,
                          collate_fn=collate_function)

    device = args.device
    model = MODELS[args.model](vocab)
    
    trainer = Trainer(model, vocab, args.epochs, args.lr, train_loader, val_loader=val_loader, device = device)
    
    trainer.fit() 
    

if __name__ == "__main__":
    main()