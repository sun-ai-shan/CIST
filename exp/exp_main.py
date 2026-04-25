import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')

BATCH_SIZE = 32


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args.output_attention = args.output_attention
        self.best_attention_weights = None

    def save_attention_weights(self, attns, epoch, batch, is_best=False):
        folder_path = './attention_weights/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if isinstance(attns, list):
            attns_cpu = [attn.detach().cpu().numpy() for attn in attns]
            avg_attention = np.mean(attns_cpu, axis=0)
        else:
            attns_cpu = attns.detach().cpu().numpy()
            avg_attention = attns_cpu

        np.save(
            os.path.join(folder_path, f'attention_weights_epoch{epoch + 1}_batch{batch + 1}.npy'),
            avg_attention,
        )

        if is_best:
            np.save(
                os.path.join(folder_path, f'best_attention_weights_epoch{epoch + 1}_batch{batch + 1}.npy'),
                avg_attention,
            )
            self.best_attention_weights = avg_attention

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if len(batch_y.shape) == 2:
                    batch_y = batch_y.unsqueeze(-1)
                    dec_inp = torch.zeros_like(batch_y[:, :self.args.pred_len]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len], dec_inp], dim=1).float().to(self.device)
                elif len(batch_y.shape) == 3:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    raise ValueError(f"Unexpected batch_y shape: {batch_y.shape}")

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, dec_inp)
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs, attns = self.model(batch_x, dec_inp)
                    else:
                        outputs = self.model(batch_x, dec_inp)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, root_path, data_path):
        full_data_path = setting.split('_')[4]
        print(f"Data path: {full_data_path}")

        train_data, test_data, vali_data, all_data, self.y_scaler = loaddata(
            full_data_path, seq_len=self.args.seq_len
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )
        vali_loader = torch.utils.data.DataLoader(
            dataset=vali_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_loss = float('inf')
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, attns = self.model(batch_x, dec_inp)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs, attns = self.model(batch_x, dec_inp)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if loss < best_loss:
                    best_loss = loss
                    self.save_attention_weights(attns, epoch, i, is_best=True)
                else:
                    self.save_attention_weights(attns, epoch, i, is_best=False)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        full_data_path = setting.split('_')[4]
        train_data, test_data, vali_data, all_data, self.y_scaler = loaddata(full_data_path)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, dec_inp)
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs, attns = self.model(batch_x, dec_inp)
                    else:
                        outputs = self.model(batch_x, dec_inp)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd_arr = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_arr, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('Test shape after reshape:', preds.shape, trues.shape)

        if hasattr(self, 'y_scaler'):
            preds_original = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
            trues_original = self.y_scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

            mae, mse, rmse, mape, mspe = metric(preds_original, trues_original)
            print('[Original scale] mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))

            mae_norm, mse_norm, rmse_norm, mape_norm, mspe_norm = metric(preds, trues)
            print('[Normalized scale] mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(
                mse_norm, mae_norm, rmse_norm, mape_norm, mspe_norm))

            np.save(folder_path + 'pred_original.npy', preds_original)
            np.save(folder_path + 'true_original.npy', trues_original)
            np.save(folder_path + 'metrics_original.npy', np.array([mae, mse, rmse, mape, mspe]))
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('[Normalized scale] mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        if self.best_attention_weights is not None:
            top_attention = self.best_attention_weights[:20, :20]
            np.save(os.path.join(folder_path, 'top_20_attention_weights.npy'), top_attention)

        return

    def predict(self, setting, load=False):
        full_data_path = setting.split('_')[4]
        train_data, test_data, vali_data, all_data, self.y_scaler = loaddata(full_data_path)

        pred_loader = torch.utils.data.DataLoader(
            dataset=all_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        attention_weights = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, dec_inp)
                        else:
                            outputs = self.model(batch_x, dec_inp)
                else:
                    if self.args.output_attention:
                        outputs, attns = self.model(batch_x, dec_inp)
                    else:
                        outputs = self.model(batch_x, dec_inp)

                true = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                true = true.detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if self.args.output_attention:
                    if isinstance(attns, list):
                        for attn in attns:
                            attention_weights.append(attn.detach().cpu().numpy())
                    else:
                        attention_weights.append(attns.detach().cpu().numpy())

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if hasattr(self, 'y_scaler'):
            preds_array = np.array(preds)
            preds_original = self.y_scaler.inverse_transform(
                preds_array.reshape(-1, 1)
            ).reshape(preds_array.shape)
            np.save(folder_path + 'real_prediction_original.npy', preds_original)
            print("Predictions saved (original scale and normalized scale)")
        else:
            print("Predictions saved (normalized scale)")

        np.save(folder_path + 'real_prediction.npy', preds)

        if self.args.output_attention:
            np.save(os.path.join(folder_path, 'attention_weights.npy'), attention_weights)

        return


def loaddata(datapath, seq_len=10):
    import esm

    def encode_clade_onehot(clade_str, max_length=20):
        """Encode a clade string as a one-hot vector."""
        chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.')
        char_to_idx = {char: i for i, char in enumerate(sorted(chars))}

        import pandas as pd
        if pd.isna(clade_str) or clade_str == '<null>':
            clade_str = 'unknown'

        padded = str(clade_str).ljust(max_length, '0')[:max_length]
        encoding = np.zeros((max_length, len(chars)))
        for i, char in enumerate(padded):
            if char in char_to_idx:
                encoding[i, char_to_idx[char]] = 1
            else:
                encoding[i, 0] = 1

        return encoding.flatten()

    print("Loading ESM-1b model...")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def get_esm_embedding(sequence, max_length=1022):
        """Get protein sequence embedding using ESM-1b."""
        import pandas as pd
        if pd.isna(sequence) or sequence is None:
            return np.zeros(1280)

        sequence = str(sequence).strip()
        if len(sequence) == 0:
            return np.zeros(1280)
        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33]
        sequence_representations = token_representations[0, 1:len(sequence) + 1]
        sequence_embedding = sequence_representations.mean(0)
        return sequence_embedding.cpu().numpy()

    import pandas as pd
    data = pd.read_csv(datapath)
    print(f"Processing {len(data)} sequences...")

    print("Data validation:")
    print(f"Total samples: {len(data)}")
    print(f"rcdata column null count: {data['rcdata'].isnull().sum()}")
    print(f"clade column null count: {data['clade'].isnull().sum()}")
    print(f"ab column null count: {data['ab'].isnull().sum()}")
    print("\nSample data:")
    print(data[['rcdata', 'clade', 'ab']].head())
    print(f"\nrcdata column dtype: {data['rcdata'].dtype}")

    print("Encoding clade information...")
    clade_max_length = 20
    clade_encodings = []
    for i, clade in enumerate(data['clade']):
        if i % 100 == 0:
            print(f"Processing clade {i + 1}/{len(data)}")
        clade_encodings.append(encode_clade_onehot(clade, clade_max_length))

    clade_features = np.array(clade_encodings)
    print(f"Clade encoding shape: {clade_features.shape}")

    print("Processing protein sequences with ESM-1b...")
    embeddings = []
    invalid_sequences = 0
    for i, sequence in enumerate(data['rcdata']):
        if i % 100 == 0:
            print(f"Processing sequence {i + 1}/{len(data)}")
        if pd.isna(sequence) or sequence is None:
            print(f"Warning: Empty sequence at index {i}, using zero vector")
            invalid_sequences += 1
        embedding = get_esm_embedding(sequence)
        embeddings.append(embedding)

    if invalid_sequences > 0:
        print(f"Warning: Found {invalid_sequences} invalid sequences, replaced with zero vectors")

    esm_features = np.array(embeddings)
    y = data['ab'].values

    print(f"ESM embedding shape: {esm_features.shape}")
    print(f"Clade encoding shape: {clade_features.shape}")

    X_combined = np.concatenate([esm_features, clade_features], axis=1)
    print(f"Combined features shape: {X_combined.shape}")
    print(f"Combined features range: {X_combined.min()} to {X_combined.max()}")

    scaler = MinMaxScaler()
    X_combined = scaler.fit_transform(X_combined)

    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    print(f"ab range (original): {data['ab'].min()} to {data['ab'].max()}")
    print(f"ab range (normalized): {y.min()} to {y.max()}")

    num_samples = len(X_combined) - seq_len + 1
    X_seq = []
    y_seq = []
    for i in range(num_samples):
        X_seq.append(X_combined[i:i + seq_len])
        y_seq.append(y[i + seq_len - 1])

    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32).reshape(-1, 1, 1)
    print(f"Final sequence data shape: {X_seq.shape}")

    N = len(X_seq)
    train_size = int(N * 0.6)
    val_size = int(N * 0.1)

    train_data = torch.utils.data.TensorDataset(X_seq[:train_size], y_seq[:train_size])
    vali_data = torch.utils.data.TensorDataset(
        X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
    )
    test_data = torch.utils.data.TensorDataset(X_seq[train_size + val_size:], y_seq[train_size + val_size:])
    all_data = torch.utils.data.TensorDataset(X_seq, y_seq)

    return train_data, test_data, vali_data, all_data, y_scaler