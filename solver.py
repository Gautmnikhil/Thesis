import torch
import numpy as np
import os
from tqdm import tqdm
from model.MTFAE import MTFA
from data_factory.data_loader import ProcessedUserLoader
from torch.utils.data import DataLoader

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        # Use updated ProcessedUserLoader
        self.train_loader = DataLoader(ProcessedUserLoader(self.data_path, mode='train', win_size=self.win_size, step=1),
                                       batch_size=self.batch_size, shuffle=True)
        self.vali_loader = DataLoader(ProcessedUserLoader(self.data_path, mode='val', win_size=self.win_size, step=1),
                                      batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(ProcessedUserLoader(self.data_path, mode='test', win_size=self.win_size, step=1),
                                      batch_size=self.batch_size, shuffle=False)
        self.thre_loader = DataLoader(ProcessedUserLoader(self.data_path, mode='thre', win_size=self.win_size, step=1),
                                      batch_size=self.batch_size, shuffle=False)

        # Update: Device selection for both GPU and CPU
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.model = MTFA(win_size=self.win_size, seq_size=self.seq_size, c_in=self.input_c, c_out=self.output_c, d_model=self.d_model, e_layers=self.e_layers, fr=self.fr, tr=self.tr, dev=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()

        loss_list = []
        with torch.no_grad():
            for i, input_data in enumerate(vali_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)

                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                        my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                            tematt[u])))

                    con_loss += (torch.mean(
                        my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                                tematt[u].detach())) + torch.mean(
                        my_kl_loss(tematt[u].detach(),
                                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                adv_loss = adv_loss / len(freatt)
                con_loss = con_loss / len(freatt)

                loss_list.append((con_loss - adv_loss).item())

        return np.average(loss_list)

    def train(self):
        print("======================TRAIN MODE======================")

        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in tqdm(range(self.num_epochs)):
            loss_list = []

            self.model.train()
            with tqdm(total=train_steps) as pbar:
                for i, input_data in enumerate(self.train_loader):

                    self.optimizer.zero_grad()

                    input = input_data.float().to(self.device)

                    tematt, freatt = self.model(input)

                    adv_loss = 0.0
                    con_loss = 0.0

                    for u in range(len(freatt)):
                        adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                            my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                                    tematt[u])))
                        con_loss += (torch.mean(my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach())) + torch.mean(
                            my_kl_loss(tematt[u].detach(), (
                                    freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                    adv_loss = adv_loss / len(freatt)
                    con_loss = con_loss / len(freatt)

                    loss = con_loss - adv_loss
                    loss_list.append(loss.item())

                    pbar.update(1)

                    loss.backward()
                    self.optimizer.step()

            train_loss = np.average(loss_list)
            vali_loss = self.vali(self.vali_loader)

            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

    def test(self):
        # Update: Use map_location to load model on CPU if CUDA is not available
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
                map_location=self.device
            )
        )
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        # (1) find the threshold
        attens_energy = []
        with torch.no_grad():
            if len(self.thre_loader) == 0:
                print("Threshold loader is empty. Cannot proceed with threshold calculation.")
                return

            for i, input_data in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    if u == 0:
                        adv_loss = my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss = my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                    else:
                        adv_loss += my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss += my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature

                metric = torch.softmax((adv_loss + con_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        if len(attens_energy) == 0:
            print("No attention energy data to concatenate. Exiting test.")
            return

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)