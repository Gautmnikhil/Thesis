import torch
import numpy as np
import os
from tqdm import tqdm
from model.MTFAE import MTFA
from data_factory.data_loader import ProcessedUserLoader
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        # Load model state
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
                map_location=self.device
            )
        )
        self.model.eval()
        
        # Set temperature based on user
        if self.dataset == 'activity_user1':
            temperature = 3  # Lower temperature for User 1
        else:
            temperature = 5  # Default temperature

        print("======================TEST MODE======================")

        # (1) Find the threshold
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
                    adv_loss += my_kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                    con_loss += my_kl_loss(
                        (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                        tematt[u].detach()) * temperature

                metric = adv_loss + con_loss
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        if len(attens_energy) == 0:
            print("No attention energy data to concatenate. Exiting test.")
            return

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # Set threshold based on user
        if self.dataset == 'activity_user1':
            thresh = np.percentile(test_energy, 85)  # Lower threshold for User 1
        else:
            thresh = np.percentile(test_energy, 90)  # Default threshold

        print("Threshold :", thresh)

        # (2) Evaluate model on the test set
        print("Evaluating model on test data...")
        predictions = []
        ground_truths = []
        anomalies = []

        with torch.no_grad():
            for i, input_data in enumerate(self.test_loader):
                input = input_data.float().to(self.device)

                # Model prediction
                tematt, freatt = self.model(input)

                # Generate prediction (as an anomaly score)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss += my_kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                    con_loss += my_kl_loss(
                        (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                        tematt[u].detach()) * temperature

                metric = adv_loss + con_loss
                cri = metric.detach().cpu().numpy()

                # Save predictions and ground truth
                predictions.extend(cri)
                ground_truths.extend(input_data[:, -1].cpu().numpy())  # Assuming labels are part of the input data

                # Save anomalies with time intervals
                for j, score in enumerate(cri):
                    if np.any(score > thresh):  # Compare each element of cri to the threshold
                        anomalies.append({"Index": i * self.batch_size + j, "Score": score.tolist()})  # Record index and score of anomaly

        # Ensure both are numpy arrays
        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)

        # Save anomalies to a CSV file in the result folder
        anomalies_df = pd.DataFrame(anomalies)
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        anomalies_file_path = os.path.join(result_dir, f"{self.dataset}_anomalies.csv")
        anomalies_df.to_csv(anomalies_file_path, index=False)
        print(f"Anomalies saved to: {anomalies_file_path}")

        # Reduce ground_truths to a single binary label per sample
        # If any element in the row is greater than 0.5, mark it as 1 (anomalous), otherwise 0 (normal)
        ground_truths_binary = np.any(ground_truths > 0.5, axis=1).astype(int)

        # Predictions are already binary, so ensure they are 1D as well
        predictions_binary = np.any(predictions > thresh, axis=1).astype(int)

        # Debugging information to ensure they match
        print("Ground Truths Binary Shape:", ground_truths_binary.shape)
        print("Predictions Binary Shape:", predictions_binary.shape)
        print("Ground Truths Binary Example:", ground_truths_binary[:5])
        print("Predictions Binary Example:", predictions_binary[:5])

        # Calculate evaluation metrics
        # Assuming binary classification: normal (0) or anomaly (1)
        accuracy = accuracy_score(ground_truths_binary, predictions_binary)
        precision = precision_score(ground_truths_binary, predictions_binary, zero_division=1)
        recall = recall_score(ground_truths_binary, predictions_binary, zero_division=1)
        f1 = f1_score(ground_truths_binary, predictions_binary, zero_division=1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Plotting Evaluation Metrics
        metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values(), color='skyblue')
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.title('Evaluation Metrics')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Visualization Example: Anomaly Scores Plot
        indices = [anomaly['Index'] for anomaly in anomalies]
        scores = [anomaly['Score'] for anomaly in anomalies]
        self.plot_anomaly_scores(indices, scores, thresh)

        # Interactive Plot using Plotly
        self.interactive_anomaly_plot(indices, scores, thresh)

    def plot_anomaly_scores(self, indices, scores, threshold):
        plt.figure(figsize=(10, 6))
        plt.plot(indices, scores, label='Anomaly Scores', color='b', alpha=0.7, linewidth=2)

        # Highlight anomalies by marking points above the threshold
        anomalies_indices = []
        anomalies_scores = []

        # Ensure consistent size for anomalies_indices and anomalies_scores
        for i, score in zip(indices, scores):
            if isinstance(score, (float, int)) and score > threshold:
                anomalies_indices.append(i)
                anomalies_scores.append(score)

        if anomalies_indices and anomalies_scores:
            plt.scatter(anomalies_indices, anomalies_scores, color='red', marker='o', label='Anomalies')

        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Index')
        plt.ylabel('Score')
        plt.title('Anomaly Scores Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def interactive_anomaly_plot(self, indices, scores, threshold):
        trace1 = go.Scatter(
            x=indices,
            y=scores,
            mode='lines',
            name='Anomaly Scores',
            line=dict(color='blue', width=2)
        )

        # Add markers for anomalies above the threshold
        anomalies_indices = []
        anomalies_scores = []

        for i, score in zip(indices, scores):
            if isinstance(score, (float, int)) and score > threshold:
                anomalies_indices.append(i)
                anomalies_scores.append(score)

        trace2 = go.Scatter(
            x=anomalies_indices,
            y=anomalies_scores,
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        )

        layout = go.Layout(title='Anomaly Scores Over Time', xaxis=dict(title='Index'), yaxis=dict(title='Score'))
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        pyo.plot(fig)
