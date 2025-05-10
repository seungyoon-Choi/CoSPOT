import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from torch import nn, optim
from torch.optim import lr_scheduler
from utils.metrics import metric, cumavg
import ptwt, pywt

from tqdm import tqdm


plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, is_train=0):
    total_loss = []
    total_mae_loss = []


    output_set, label_set = None, None
    if is_train == 0:

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                if args.model == 'FT_online':
                    fourier_batch_x = torch.fft.rfft(batch_x, dim=1)
                    fourier_batch_x = torch.abs(fourier_batch_x)
                    #normalize
                    fourier_batch_x = fourier_batch_x.to(torch.bfloat16)

                    fourier_recent_x = torch.fft.rfft(batch_x[:,-args.patch_len:,:], dim=1)
                    fourier_recent_x = torch.abs(fourier_recent_x)
                    fourier_recent_x = fourier_recent_x.to(torch.bfloat16)

                    wavelet_batch_x = ptwt.wavedec(batch_x.reshape(-1,batch_x.shape[2], batch_x.shape[1]), pywt.Wavelet('haar'), mode='periodic', level=args.wavelet_level)
                    wavelet_batch_x = wavelet_batch_x[0]
                    wavelet_batch_x = wavelet_batch_x.reshape(-1,wavelet_batch_x.shape[2], wavelet_batch_x.shape[1])
                    wavelet_batch_x = wavelet_batch_x.to(torch.bfloat16)

                elif args.model == 'WT_online':
                    wavelet_batch_x = ptwt.wavedec(batch_x.reshape(-1,batch_x.shape[2], batch_x.shape[1]), pywt.Wavelet('haar'), mode='periodic', level=args.wavelet_level)
                    wavelet_batch_x = wavelet_batch_x[0]
                    wavelet_batch_x = wavelet_batch_x.reshape(-1,wavelet_batch_x.shape[2], wavelet_batch_x.shape[1])
                    wavelet_batch_x = wavelet_batch_x.to(torch.bfloat16)


                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)
                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        if args.model == 'FT_online':
                            outputs, pseudo_label = model(batch_x, fourier_batch_x, batch_x_mark, dec_inp, batch_y_mark, args.prompt_type, args.frequency_ratio, wavelet_batch_x)
                        elif args.model == 'WT_online':
                            outputs, pseudo_label = model(batch_x, wavelet_batch_x, batch_x_mark, dec_inp, batch_y_mark, args.prompt_type, args.frequency_ratio)
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                #outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                mae_loss = mae_metric(pred, true)

                total_loss.append(loss.item())
                total_mae_loss.append(mae_loss.item())

            total_loss = np.average(total_loss)
            total_mae_loss = np.average(total_mae_loss)

    elif is_train != 0:
        model.train()

        trained_parameters = []
        for p in model.parameters():
            p.requires_grad = False
        # for param in model.reprogramming_layer.parameters():
        #     param.requires_grad = True
        # model.learnable_prompt.requires_grad = True
        for param in model.output_projection.parameters():
            param.requires_grad = True
        # for param in model.patch_embedding.parameters():
        #     param.requires_grad = True
        # for param in model.fourier_projection.parameters():
        #     param.requires_grad = True
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
        model_optim = optim.AdamW(trained_parameters, lr = args.learning_rate)
        #train_steps = len(test_loader)
        # scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
        #                                     steps_per_epoch=train_steps,
        #                                     pct_start=args.pct_start,
        #                                     epochs=1,
        #                                     max_lr=args.learning_rate)
        #train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            if args.model == 'FT_online':
                fourier_batch_x = torch.fft.rfft(batch_x, dim=1)
                fourier_batch_x = torch.abs(fourier_batch_x)
                #normalize
                fourier_batch_x = fourier_batch_x.to(torch.bfloat16)

                fourier_recent_x = torch.fft.rfft(batch_x[:,-args.patch_len:,:], dim=1)
                fourier_recent_x = torch.abs(fourier_recent_x)
                fourier_recent_x = fourier_recent_x.to(torch.bfloat16)

                wavelet_batch_x = ptwt.wavedec(batch_x.reshape(-1,batch_x.shape[2], batch_x.shape[1]), pywt.Wavelet('haar'), mode='periodic', level=args.wavelet_level)
                wavelet_batch_x = wavelet_batch_x[0]
                wavelet_batch_x = wavelet_batch_x.reshape(-1,wavelet_batch_x.shape[2], wavelet_batch_x.shape[1])
                wavelet_batch_x = wavelet_batch_x.to(torch.bfloat16)

            elif args.model == 'WT_online':
                wavelet_batch_x = ptwt.wavedec(batch_x.reshape(-1,batch_x.shape[2], batch_x.shape[1]), pywt.Wavelet('haar'), mode='periodic', level=args.wavelet_level)
                wavelet_batch_x = wavelet_batch_x[0]
                wavelet_batch_x = wavelet_batch_x.reshape(-1,wavelet_batch_x.shape[2], wavelet_batch_x.shape[1])
                wavelet_batch_x = wavelet_batch_x.to(torch.bfloat16)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if args.model == 'FT_online':
                        outputs, pseudo_label = model(batch_x, fourier_batch_x, batch_x_mark, dec_inp, batch_y_mark, args.prompt_type, args.frequency_ratio, wavelet_batch_x)
                    elif args.model == 'WT_online':
                        outputs, pseudo_label = model(batch_x, wavelet_batch_x, batch_x_mark, dec_inp, batch_y_mark, args.prompt_type, args.frequency_ratio)
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            #outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pseudo_y = torch.cat((batch_y[:,:1,:],pseudo_label[:,1:,:]),dim=1)
            loss = criterion(outputs, batch_y)

            #pred = outputs.detach()
            #true = batch_y.detach()
            if i >= len(vali_loader)-200:
                output_set = torch.cat((output_set, outputs), dim=1) if output_set != None else outputs
                label_set = torch.cat((label_set, batch_y), dim=1) if label_set != None else batch_y

            mse_loss = criterion(outputs, batch_y)
            # loss = criterion(outputs, batch_y) + mae_metric(outputs, batch_y)
            mae_loss = mae_metric(outputs, batch_y)
            
            accelerator.backward(loss)
            model_optim.step()


            mae, mse, rmse, mape, mspe = metric((outputs).detach().cpu().numpy(), batch_y.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe) 


            # loss = criterion(outputs, batch_y)
            # # loss = criterion(outputs, batch_y) + mae_metric(outputs, batch_y)
            # mae_loss = mae_metric(outputs, batch_y)
            
            # accelerator.backward(loss)
            # model_optim.step()

            total_loss.append(mse_loss.item())
            total_mae_loss.append(mae_loss.item())

        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        total_loss = mse
        total_mae_loss = mae


    # total_loss = np.average(total_loss)
    # total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss, output_set, label_set





def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content