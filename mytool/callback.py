# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年11月24日
"""
from typing import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import optuna
from . import plot
from . import tool
from . import mytorch


class Callback:
    def __init__(self, params=None, model=None):
        self.params = params if params is not None else None
        self.model = model if model is not None else None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):  # only for train
        pass

    def on_epoch_end(self, epoch, logs=None):  # only for train
        pass


class CallbackList(Callback):
    def __init__(self, callbacks, params=None, model=None):
        super().__init__(params, model)
        self.callbacks = list(callbacks)
        for cal in self.callbacks:
            if self.params is not None: cal.set_params(self.params)
            if self.model is not None: cal.set_model(self.model)

    def append(self, callback: Union[Callback, List[Callback]]):
        if isinstance(callback, Callback):
            callback = [callback]

        for cal in callback:
            if self.params is not None: cal.set_params(self.params)
            if self.model is not None: cal.set_model(self.model)
            self.callbacks.append(cal)

    def on_train_begin(self, logs=None):
        for cal in self.callbacks:
            cal.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cal in self.callbacks:
            cal.on_train_end(logs)

    def on_train_batch_begin(self, batch, logs=None):
        for cal in self.callbacks:
            cal.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        for cal in self.callbacks:
            cal.on_train_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):  # only for train
        for cal in self.callbacks:
            cal.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):  # only for train
        # special flag for early_stop_callback, if return True, stop training.
        break_flag = None
        for cal in self.callbacks:
            if cal.on_epoch_end(epoch, logs):
                break_flag = True
        return break_flag


class PrintCallback(Callback):
    """
    Print Training logs.
    """

    def __init__(self, total_epochs, steps_per_epoch,
                 batch_sep=100, if_micro_average=True, quiet=False, **kwargs):
        super().__init__(**kwargs)
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_sep = batch_sep
        self.if_micro_average = if_micro_average
        self.quiet = quiet
        self.micro_batch_history = mytorch.History()

    def on_train_begin(self, logs=None):
        if not self.quiet:
            print('Begin Train!')

    def on_train_end(self, logs=None):
        if not self.quiet:
            print('End Train!')

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if self.if_micro_average:
            self.micro_batch_history.update(logs)

        if batch % self.batch_sep == 0:
            print_str = f'    [{batch}/{self.steps_per_epoch}] Batch {batch}'
            if self.if_micro_average:
                logs = self.micro_batch_history.mean()
                self.micro_batch_history.clear()
            for key in logs.keys():
                print_str += f' {key} {logs[key]}'
            if not self.quiet:
                print(print_str)
        else:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        if not self.quiet:
            print('=' * 20, f'[{epoch:04d}/{self.total_epochs:04d}]', ' Epoch ', f'{epoch} ', '=' * 20)

    def on_epoch_end(self, epoch, logs=None):
        print_str = f'Epoch {epoch}'
        for key in logs.keys():
            print_str += f' {key} {logs[key]}'
        if not self.quiet:
            print(print_str)


class TbWriterCallback(Callback):
    """
    Add epoch_logs to SummaryWriter
    """

    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.tb_writer = SummaryWriter(os.path.join(log_dir, tool.get_datetime_name()))

    def on_epoch_end(self, epoch, logs=None):
        for key in logs.keys():
            self.tb_writer.add_scalar('Epoch/' + key, logs[key], epoch)

    def on_train_end(self, logs=None):
        self.tb_writer.close()


class DfSaveCallback(Callback):
    """
    Save epoch_logs to CSV
    """

    def __init__(self, df_save_path, **kwargs):
        super().__init__(**kwargs)
        self.df_save_path = df_save_path
        basedir = os.path.dirname(df_save_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.epoch_history = mytorch.History()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)
        df = pd.DataFrame(self.epoch_history.history)
        df.to_csv(self.df_save_path, index=False, header=True)


class EarlyStoppingCallback(Callback):
    """
    如果有进步，那么patience_flag=patience，更新last_loss或last_mc(if restore_best_weights: save_model_weight)
    如果没有进步，那么patience_flag -=1，
    如果patience_flag==-1，return break_flag=True

    note:
        "进步": loss(down,mode=min), metric(up,mode=max)
        "dowm": cur_loss<last_loss and last_loss-cur_loss>=min_delta
                and baseline>cur_loss and baseline-cur_loss>=min_delta
        "up": cur_mc>last_mc and cur_mc-last_mc>=min_delta
                and cur_mc>baseline and cur_mc-baseline>=min_delta
    """

    def __init__(self, checkpoint_dir, monitor='val_loss', min_delta=0, patience=0,
                 mode='auto', baseline=None, restore_best_weights=False, save_best_only=False, quiet=False, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        if mode == 'auto':
            self.mode = 'min' if monitor.find('loss') != -1 else 'max'
        else:
            self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.save_best_only = save_best_only
        self.quit = quiet

        self.patience_flag = patience
        self.last_loss = float('inf') if self.mode == 'min' else None
        self.last_mc = 0. if self.mode == 'max' else None
        self.last_model_weight_path = None

    def save_weights(self):
        if_last_save = False
        if self.last_model_weight_path is not None:
            # 上次有保存过权重
            if_last_save = True
        # save
        new_model_weight_path = os.path.join(self.checkpoint_dir, tool.get_datetime_name() + '.pth')
        torch.save(self.model.state_dict(), new_model_weight_path)
        # 删除上个文件
        if if_last_save and self.save_best_only \
                and os.path.exists(self.last_model_weight_path):  # if remove last
            os.remove(self.last_model_weight_path)
        # 更新路径
        self.last_model_weight_path = new_model_weight_path

    def on_train_end(self, logs=None):
        if self.restore_best_weights:
            self.model.load_state_dict(torch.load(self.last_model_weight_path))

    def on_epoch_end(self, epoch, logs=None):
        break_flag = False
        if self.mode == 'min':  # loss
            cur_loss = logs[self.monitor]
            if self.baseline:  # 是否有baseline
                if cur_loss < self.last_loss and self.last_loss - cur_loss >= self.min_delta \
                        and self.baseline > cur_loss and self.baseline - cur_loss >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_loss = cur_loss
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True
            else:
                if cur_loss < self.last_loss and self.last_loss - cur_loss >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_loss = cur_loss
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True
        else:  # metric
            cur_mc = logs[self.monitor]
            if self.baseline:
                if cur_mc > self.last_mc and cur_mc - self.last_mc >= self.min_delta \
                        and cur_mc > self.baseline and cur_mc - self.baseline >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_mc = cur_mc
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True
            else:
                if cur_mc > self.last_mc and cur_mc - self.last_mc >= self.min_delta:
                    # 进步
                    self.save_weights()
                    self.patience_flag = self.patience
                    self.last_mc = cur_mc
                else:
                    self.patience_flag -= 1
                    if self.patience_flag == -1: break_flag = True

        if break_flag:
            if not self.quit:
                print('*' * 20, 'EarlyStopping!', '*' * 20)
            return True


class SchedulerWrapCallback(Callback):
    """
    custom scheduler_func has args: (cur_epoch,[cur_lr,[cur_epoch_logs,[util_now_epoch_history]]]).
    It update on epoch end and apply it lr_rate for next epoch.
    """

    def __init__(self, scheduler_func, manual=True, **kwargs):
        super().__init__(**kwargs)
        self.scheduler_func = scheduler_func
        self.manual = manual
        self.epoch_history = mytorch.History()
        self.argcount = self.scheduler_func.__code__.co_argcount

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)
        cur_epoch, cur_lr, cur_epoch_logs = epoch, mytorch.get_optimizer_lr(self.params['optimizer']), logs
        util_now_epoch_history = self.epoch_history
        if self.argcount == 1:
            new_lr = self.scheduler_func(cur_epoch)
        elif self.argcount == 2:
            new_lr = self.scheduler_func(cur_epoch, cur_lr)
        elif self.argcount == 3:
            new_lr = self.scheduler_func(cur_epoch, cur_lr, cur_epoch_logs)
        elif self.argcount == 4:
            new_lr = self.scheduler_func(cur_epoch, cur_lr, cur_epoch_logs, util_now_epoch_history)
        else:
            raise Exception('scheduler_func must have 1/2/3/4 parameters!!')
        if self.manual:
            return
        else:
            assert new_lr is not None
            mytorch.set_optimizer_lr(self.params['optimizer'], new_lr)


class PlotLossMetricTimeLr(Callback):
    def __init__(self, loss_check='loss', metric_check='metric', time_check='time', lr_check='lr', **kwargs):
        super().__init__(**kwargs)
        self.loss_check = loss_check
        self.metric_check = metric_check
        self.time_check = time_check
        self.lr_check = lr_check
        self.epoch_history = mytorch.History()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.epoch_history.history)
        fig = plot.plot_LossMetricTimeLr_with_df(df,self.loss_check,self.metric_check,self.time_check,self.lr_check)
        plt.show()


class TunerRemovePreFileInDir(Callback):
    def __init__(self, dirs=None,  num_check_threshold=20, remove_ratio = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.dirs =[dirs]  if not isinstance(dirs,(tuple,list)) else dirs
        self.num_check_threshold = num_check_threshold
        self.remove_ratio = remove_ratio

    def on_train_end(self, logs=None):
        for dir in self.dirs:
            if dir is None:
                pass
            else:
                tool.remove_prefile_in_dir(dir,self.num_check_threshold,self.remove_ratio)
        return


class MyTunerPrintCallback(Callback):
    def __init__(self, max_trials, executions_per_trial, quiet=False):
        super().__init__()
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.quit = quiet

    def on_train_begin(self, logs=None):
        if not self.quit:
            print('Start Tuner!\n')

    def on_train_end(self, logs=None):
        if not self.quit:
            print('End Tuner!\n')

    def on_train_batch_begin(self, batch, logs=None):
        if not self.quit:
            print(f'Start Repeat #{batch}/{self.executions_per_trial}')

    def on_train_batch_end(self, batch, logs=None):
        #flag, time
        if not self.quit:
            print(f'End Repeat #{batch}/{self.executions_per_trial}',
                  '\tRepeat time: ', logs['time'],
                  '\tRepeat flag: ', logs['flag'])

    def on_epoch_begin(self, epoch, logs=None):  # only for train
        if not self.quit:
            print(f'Start Trial #{epoch}/{self.max_trials}')

    def on_epoch_end(self, epoch, logs=None):  # only for train
        if not self.quit:
            print(f'End Trial #{epoch}/{self.max_trials}',
                  '\tUntil now total time: ', logs['time'],
                  '\tCurrent flag: ', logs['cur_flag'],
                  '\tBest flag: ', logs['best_flag'])

        cur_configdict = logs['cur_configdict']
        best_configdict = logs['best_configdict']
        tool.print_dicts_tablefmt([cur_configdict,best_configdict],['Current','Best'])


class PruningCallback(Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor:str, **kwargs):
        super().__init__(**kwargs)
        self.trial = trial
        self.monitor = monitor
        self.epoch_history = mytorch.History()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_history.update(logs)
        self.trial.report(logs[self.monitor], epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

    def on_train_end(self, logs=None):
        self.trial.set_user_attr('epoch_history',pd.DataFrame(self.epoch_history.history).to_dict('list'))


class StudyStopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int=20, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()