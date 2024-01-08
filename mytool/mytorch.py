# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
import time
import torch
import numpy as np
from mytool import tool
import torch.nn as nn
from torch.utils.data import Dataset
import functools
from .callback import CallbackList, PrintCallback, MyTunerPrintCallback


def get_device():
    """
    :return:
        device: torch.device
    """
    detect_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return torch.device(detect_device)


def tensor_to_device(tensor_or_tensorlist, device):
    """
    map tensor or List[tensor]... to device
    :param tensor_or_tensorlist: tensor or List[tensor]
    :param device: torch device
    :return:
        tensor_or_tensorlist(device)
    """
    if torch.is_tensor(tensor_or_tensorlist):
        return tensor_or_tensorlist.to(device)
    else:
        return list(map(functools.partial(tensor_to_device, device=device), tensor_or_tensorlist))


def parser_compile_kw(compile_kw):
    """
    An auxiliary function
    :param compile_kw: Dict
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
        :key loss_weights[optional]: List[numeric] or None calculate weight loss for every output
    :return:
        loss_fn
        optimizer
        scheduler
        metric_list
    """
    loss_fn, optimizer = compile_kw['loss'], compile_kw['optimizer']
    scheduler = compile_kw['scheduler'] if 'scheduler' in compile_kw.keys() else None
    loss_weights = compile_kw['loss_weights'] if 'loss_weights' in compile_kw.keys() else None
    if 'metric' in compile_kw.keys():
        metric_list = compile_kw['metric'] if isinstance(compile_kw['metric'], (list, tuple)) else [
            compile_kw['metric']]
    else:  # None
        metric_list = []
    return loss_fn, optimizer, scheduler, metric_list, loss_weights


def get_optimizer_lr(optimizer):
    """get current optimizer lr"""
    for p in optimizer.param_groups:
        return p['lr']


def set_optimizer_lr(optimizer, cur_lr):
    """adjust torch.optim.optimizer lr rate"""
    for p in optimizer.param_groups:
        p['lr'] = cur_lr


def compute_loss(inputs, labels, model, loss_fn, device, loss_weights=None):
    """
    cope with a situation which may have more than one tensor(inputs,labels)/loss_fn
    :param inputs: tensor or List[tensor]
    :param labels: tensor or List[labels]
    :param model: torch model
    :param loss_fn: loss_fn or List[loss_fn]
    :param device: torch device
    :param loss_weights: None or List[numeric]
    :return:
        :key loss: for back propagation
        :key outputs: for metric_func
        :key labels: for metric_func
        :key loss_info_dict: Dict
            :key loss:
            :key loss_funcname_1/funcname_1_weight: maybe
            :key loss_funcname_2/funcname_2_weight: maybe
            ...
    """
    # init
    loss_info_dict = {}

    # get outputs and to list(has been to device)
    inputs = tensor_to_device(inputs, device)
    outputs = model(inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]  # rename

    # labels to list and to device
    if not isinstance(labels, (list, tuple)):  # tensor
        labels = [labels]
    labels = tensor_to_device(labels, device)
    assert len(outputs) == len(labels)

    # loss to list and has same dimension
    if not isinstance(loss_fn, (list, tuple)):  # func
        loss_list = [loss_fn for i in outputs]
    elif len(loss_fn) == 1:
        loss_list = [loss_fn[0] for i in outputs]
    elif len(loss_fn) == len(outputs):
        loss_list = loss_fn
    else:
        raise Exception('loss_fn must be correct for outputs!')
    loss_fn = loss_list  # rename
    assert len(loss_fn) == len(outputs)

    # set loss_weights
    if loss_weights is None:
        loss_weights = [1 for i in outputs]  # 同等权重
    assert len(loss_weights) == len(outputs)

    # normalize loss_weights
    lw_sum = np.sum(loss_weights)
    nlw = [torch.tensor(w / lw_sum, device=device) for w in loss_weights]
    loss_weights = nlw  # rename

    # Now outputs/labels/loss_fn/loss_weights are list
    # next step compute loss
    loss = torch.tensor(0.)
    for output_i, (out, lab, lf, lw) in enumerate(zip(outputs, labels, loss_fn, loss_weights)):
        # set func name
        lf_name = tool.get_func_name(lf)
        if len(outputs) > 1:
            lf_name += '_' + str(output_i + 1)

        # cal origin loss
        closs = lf(out, lab)

        # weight loss
        loss = loss + lw * closs

        # collect origin loss and loss weight
        if len(outputs) > 1:
            loss_info_dict['loss_' + lf_name] = closs.item()
            loss_info_dict[lf_name + '_weight'] = lw.item()

    # end for
    loss_info_dict['loss'] = loss.item()
    return loss, outputs, labels, loss_info_dict


def compute_single_metric(output, label, metric_func):
    """
    An auxiliary function for compute_metric
    :param output: tensor
    :param label: tensor
    :param metric_func: single_metric_func
    :return:
        :key metric_dict: Dict
            :key metric_name: metric_value
    """
    ndoutputs, ndlabels = output.cpu(), label.cpu()
    metric_value = metric_func(ndoutputs, ndlabels)
    metric_name = tool.get_func_name(metric_func)
    return {metric_name: metric_value}


def compute_metric(outputs, labels, metric_list):
    """
    cope with a situation which has more than one outputs/labels,
    and metric_list maybe have zero or one or many(more than len(outputs)) metric_func
    :param outputs: List[tensor] inherit from compute_loss
    :param labels: List[tensor] inherit from compute_loss
    :param metric_list: List[Union[metric_func,None]] or List[Union[List[metric_func], None]]
        shape:
            if it has two dimensions Or it has one dimension and len(metric_list) == len(outputs),
                suppose it corresponds to outputs/labels, len(metric_list) must be equal to len(outputs)
            else it must only have one dimension and it does not correspond to outputs/labels
        value:
            if element is None, it means that it does not need to calculate metric
    :return:
        metric_info_dict: Dict
            :key metric_metricname_1: value
            :key metric_metricname_2: value
    """
    metric_info_dict = {}
    if_two_dimension = False
    for el in metric_list:
        if isinstance(el, (tuple, list)):  # 只要有一个列表，就认为是二维
            if_two_dimension = True

    if_correspond = False
    if if_two_dimension or len(metric_list) == len(outputs):  # 认为是对应与outputs/labels
        assert len(metric_list) == len(outputs)
        if_correspond = True

    if if_correspond:  # 对应outputs/labels
        for output_ind, (mf_or_mf_list, output, label) in enumerate(zip(metric_list, outputs, labels)):
            if isinstance(mf_or_mf_list, (tuple, list)):  # list
                for mf in mf_or_mf_list:
                    if mf is not None:  # 是否计算loss
                        if len(outputs) == 1:  #
                            metric_info_dict.update(
                                tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                    'metric_'))
                        else:
                            metric_info_dict.update(
                                tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                    prefix='metric_',
                                                                    suffix='_' + str(output_ind + 1)))
            elif mf_or_mf_list is None:
                pass
            else:  # single func
                if len(outputs) == 1:
                    metric_info_dict.update(
                        tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf_or_mf_list),
                                                            'metric_'))
                else:
                    metric_info_dict.update(
                        tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf_or_mf_list),
                                                            prefix='metric_',
                                                            suffix='_' + str(output_ind + 1)))
    else:  # 一维 没有对应
        assert if_two_dimension == False
        # 可能只有一个metric_func,或者多于len(outputs)个metric_func
        for output_ind, (output, label) in enumerate(zip(outputs, labels)):
            for mf in metric_list:
                if mf is not None:  # 是否计算metric
                    if len(outputs) == 1:
                        metric_info_dict.update(
                            tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                'metric_'))
                    else:
                        metric_info_dict.update(
                            tool.add_pre_or_suf_on_key_for_dict(compute_single_metric(output, label, mf),
                                                                prefix='metric_',
                                                                suffix='_' + str(output_ind + 1)))
    return metric_info_dict


def fit(model, dataload, epochs, compile_kw,
        device=None, val_dataload=None, callbacks=[],
        quiet=False):
    """
    similar to tf.keras.Model.fit
    :param model: torch model
    :param dataload: torch.utils.data.DataLoader for train
    :param epochs: iteration
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
        :key loss_weights[optional]: List[numeric] or None calculate weight loss for every output
    :param device: torch device
    :param val_dataload: torch.utils.data.DataLoader for validation
    :param callbacks: List[Callback] or CallbackList similar to tf.keras.callbacks
    :return:
        history_epoch:
            :key lr: 1-D epoch lr if it has a scheduler
            :key time/val_time: 1-D epoch_time
            :key loss/val_loss: 1-D epoch_loss
            :key loss_funcname_1/val_loss_funcname_1: 1-D one_output_loss
            :key funcname_1_weight/val_funcname_1_weight: 1-D one_output_loss_weight
            :key metric_metricName_1/val_metric_metricName_1: 1-D epoch_metric
            ...
    """
    history_epoch = History()
    # process args
    steps_per_epoch = len(dataload)
    validation_steps = None if val_dataload is None else len(val_dataload)
    device = device if device is not None else get_device()
    callbacklist = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks,
                                                                                      model=model, params={
            'optimizer': compile_kw['optimizer']})
    callbacklist.append(PrintCallback(epochs, steps_per_epoch,quiet=quiet))

    # train begin
    callbacklist.on_train_begin(logs=None)
    for epoch_index in range(epochs):

        # epoch begin
        callbacklist.on_epoch_begin(epoch_index + 1, logs=None)

        # train one epoch
        epoch_logs = train_epoch(dataload, model, compile_kw, device, callbacks=callbacklist)

        # val one epoch
        if val_dataload is not None:
            val_epoch_logs = val_epoch(val_dataload, model, compile_kw, device)
        else:
            val_epoch_logs = {}

        # collect epoch logs
        history_epoch.update(epoch_logs)
        history_epoch.update(val_epoch_logs)

        # epoch end
        epoch_logs.update(val_epoch_logs)
        if callbacklist.on_epoch_end(epoch_index + 1, logs=epoch_logs): break

    # train end
    callbacklist.on_train_end(logs=None)

    return history_epoch


def train_epoch(dataload, model, compile_kw,
                device, callbacks=[]):
    """
    train dataload for one epoch
    :param dataload: torch.utils.data.DataLoader for train
    :param model: torch model
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
        :key loss_weights[optional]: List[numeric] or None calculate weight loss for every output
    :param device: torch device
    :param callbacks: List[Callback] or CallbackList similar to tf.keras.callbacks
    :return:
        epoch_logs:
            :key lr: float if it has a scheduler.
            :key time: float epoch time calculate sum batch time
            :key loss: float epoch loss calculate mean batch loss
            :key loss_funcname_1...: maybe
            :key funcname_1_weight...: maybe
            :key metric_metricName_1...: float epoch metric calculate mean batch metric
            ...
    """
    # epoch init
    callbacklist = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)

    # compile_kw
    loss_fn, optimizer, scheduler, metric_list, loss_weights = parser_compile_kw(compile_kw)

    # get current epoch lr
    cur_epoch_lr = get_optimizer_lr(optimizer)

    model.train()
    # collect batch logs
    history_batch = History()
    for i, data in enumerate(dataload):
        # batch begin
        callbacklist.on_train_batch_begin(i + 1, logs=None)

        # train step
        batch_logs = train_step(data, compile_kw, model, device)

        # collect batch logs
        history_batch.update(batch_logs)

        # batch end
        callbacklist.on_train_batch_end(i + 1, logs=batch_logs)

    # end for
    epoch_logs = history_batch.mean(but_sum_for_keys=['time'])
    epoch_logs['lr'] = cur_epoch_lr
    # 调整学习率
    if scheduler is not None: scheduler.step()

    return epoch_logs


def val_epoch(dataload, model, compile_kw, device):
    """
    val or test dataload for one epoch
    :param dataload: torch.utils.data.DataLoader for validation
    :param model: torch model
    :param compile_kw:
        :key loss: compute loss
        :key optimizer: torch.optim
        :key metric[optional]: func(outputs,labels) or List[func]
        :key scheduler[optional]: torch.optim.lr_scheduler
        :key loss_weights[optional]: List[numeric] or None calculate weight loss for every output
    :param device: torch.device
    :return:
        val_epoch_logs:
            :key val_time: float epoch time calculate sum batch time
            :key val_loss: float epoch loss calculate mean batch loss
            :key val_loss_funcname_1...:
            :key val_funcname_1_weight...:
            :key val_metric_metricName_1: float epoch metric calculate mean batch metric
    """
    model.eval()
    # collect batch logs
    history_batch = History()
    with torch.no_grad():
        for i, vdata in enumerate(dataload):
            # test step
            batch_logs = test_step(vdata, compile_kw, model, device)

            # collect batch logs
            history_batch.update(batch_logs)

    # end for
    val_epoch_logs = history_batch.mean(but_sum_for_keys=['val_time'])

    return val_epoch_logs


def train_step(data, compile_kw, model, device):
    """
    Perform a single training step.

    Args:
        data (tuple): A tuple containing the inputs and labels for the batch.
        compile_kw (dict): A dictionary containing the compilation parameters for the model.
        model: The model to be trained.
        device: The device on which the model and data should be loaded.

    Returns:
        dict: A dictionary containing the loss and metric information for the batch.
    """
    # batch init
    batch_logs = {}
    batch_start_time = time.time()

    # compile_kw
    loss_fn, optimizer, scheduler, metric_list, loss_weights = parser_compile_kw(compile_kw)

    # forward and back propagation
    optimizer.zero_grad()
    inputs, labels = data
    # compute loss
    loss, outputs, labels, loss_info_dict = compute_loss(inputs, labels, model, loss_fn, device, loss_weights)
    loss.backward()
    optimizer.step()

    # collect loss
    batch_logs.update(loss_info_dict)

    # compute metric and collect metric
    metric_info_dict = compute_metric(outputs, labels, metric_list)
    batch_logs.update(metric_info_dict)

    # collect batch time
    batch_logs['time'] = time.time() - batch_start_time

    return batch_logs


def test_step(data, compile_kw, model, device):
    """
    Perform a single testing step.

    Args:
        data (tuple): A tuple containing the input data and labels.
        compile_kw (dict): A dictionary containing the compilation keywords.
        model: The model to be tested.
        device: The device on which the testing will be performed.

    Returns:
        dict: A dictionary containing the loss and metric information for the testing step.
    """
    # batch init
    batch_logs = {}
    batch_start_time = time.time()

    # compile_kw
    loss_fn, optimizer, scheduler, metric_list, loss_weights = parser_compile_kw(compile_kw)

    # forward propagation
    vinputs, vlabels = data
    vloss, voutputs, vlabels, loss_info_dict = compute_loss(vinputs, vlabels, model, loss_fn, device,
                                                            loss_weights)

    # collect loss
    batch_logs.update(loss_info_dict)

    # compute metric and collect metric
    metric_info_dict = compute_metric(voutputs, vlabels, metric_list)
    batch_logs.update(metric_info_dict)

    # change name: add prefix 'val_' to key
    batch_logs = tool.add_pre_or_suf_on_key_for_dict(batch_logs, 'val_')

    # collect batch time
    batch_logs['val_time'] = time.time() - batch_start_time

    return batch_logs


class History:
    """
    It is convenient to collect key-value pairs.
    """
    def __init__(self):
        self.history = {}  # all values must be 1-D List

    def update(self, other_dict):
        """
        append a number which has the same key
        :param other_dict:
        :return:
        """
        for key in other_dict.keys():
            if key in self.history.keys():
                self.history[key].append(other_dict[key])
            else:
                self.history[key] = [other_dict[key]]

    def clear(self):
        """reset dict"""
        self.history = {}

    def mean(self, but_sum_for_keys=[]):
        """
        calculate mean value for every key of dictionary
        :return:
            mean_dict
        """
        mean_dict = {}
        if isinstance(but_sum_for_keys, str): but_sum_for_keys = [but_sum_for_keys]
        for key in self.history.keys():
            if key in but_sum_for_keys:
                mean_dict[key] = np.sum(self.history[key])
            else:
                mean_dict[key] = np.mean(self.history[key])
        return mean_dict


class ExtendModel(nn.Module):
    """
    ExtendModel is a subclass of torch.nn.Module
    It makes API more similar to tf.keras.Model
    """
    def compile(self, loss, optimizer, metric=None, scheduler=None, loss_weights=None):
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_weights = loss_weights
        if metric is None:
            self.metric_list = []
        elif isinstance(metric, (tuple, list)):
            self.metric_list = metric
        else:
            self.metric_list = [metric]

    def fit(self, dataload,
            epochs,
            device=None,
            val_dataload=None,
            callbacks=[],
            quiet=False):
        # init
        self.epochs = epochs
        self.device = device if device is not None else get_device()
        self.quiet = quiet
        steps_per_epoch = len(dataload)
        self.callbacklist = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks,
                                                                                               model=self, params={'optimizer': self.optimizer})
        self.callbacklist.append(PrintCallback(epochs, steps_per_epoch,quiet=quiet)) # default

        # train begin
        history_epoch = History()
        self.callbacklist.on_train_begin(logs=None)
        for epoch_index in range(epochs):

            # epoch begin
            self.callbacklist.on_epoch_begin(epoch_index + 1, logs=None)

            # train one epoch
            epoch_logs = self.train_epoch(dataload)

            # val one epoch
            if val_dataload is not None:
                val_epoch_logs = self.val_epoch(val_dataload)
            else:
                val_epoch_logs = {}

            # collect epoch logs
            history_epoch.update(epoch_logs)
            history_epoch.update(val_epoch_logs)

            # epoch end
            epoch_logs.update(val_epoch_logs)
            if self.callbacklist.on_epoch_end(epoch_index + 1, logs=epoch_logs): break

        # train end
        self.callbacklist.on_train_end(logs=None)

        return history_epoch

    def train_epoch(self, dataload):
        # get current epoch lr
        cur_epoch_lr = get_optimizer_lr(self.optimizer)

        self.train()
        history_batch = History()
        for i, data in enumerate(dataload):
            # batch begin
            self.callbacklist.on_train_batch_begin(i + 1, logs=None)

            # train step
            batch_logs = self.train_step(data)

            # collect batch logs
            history_batch.update(batch_logs)

            # batch end
            self.callbacklist.on_train_batch_end(i + 1, logs=batch_logs)

        # end for
        epoch_logs = history_batch.mean(but_sum_for_keys=['time'])
        epoch_logs['lr'] = cur_epoch_lr
        # 调整学习率
        if self.scheduler is not None: self.scheduler.step()

        return epoch_logs

    def val_epoch(self, val_dataload):
        self.eval()
        history_batch = History()
        with torch.no_grad():
            for i, vdata in enumerate(val_dataload):
                # test step
                batch_logs = self.test_step(vdata)

                # collect batch logs
                history_batch.update(batch_logs)

        # end for
        val_epoch_logs = history_batch.mean(but_sum_for_keys=['val_time'])

        return val_epoch_logs

    def train_step(self, data):
        # batch init
        batch_logs = {}
        batch_start_time = time.time()

        # forward and back propagation
        self.optimizer.zero_grad()
        inputs, labels = data
        # compute loss
        loss, outputs, labels, loss_info_dict = compute_loss(inputs, labels,
                                                             self, self.loss_fn,
                                                             self.device, self.loss_weights)
        loss.backward()
        self.optimizer.step()

        # collect loss
        batch_logs.update(loss_info_dict)

        # compute metric and collect metric
        metric_info_dict = compute_metric(outputs, labels, self.metric_list)
        batch_logs.update(metric_info_dict)

        # collect batch time
        batch_logs['time'] = time.time() - batch_start_time

        return batch_logs

    def test_step(self, data):
        # batch init
        batch_logs = {}
        batch_start_time = time.time()

        # forward propagation and compute loss
        vinputs, vlabels = data
        vloss, voutputs, vlabels, loss_info_dict = compute_loss(vinputs, vlabels,
                                                                self, self.loss_fn,
                                                                self.device, self.loss_weights)
        # collect loss
        batch_logs.update(loss_info_dict)

        # compute metric and collect metric
        metric_info_dict = compute_metric(voutputs, vlabels, self.metric_list)
        batch_logs.update(metric_info_dict)

        # collect batch time
        batch_logs['time'] = time.time() - batch_start_time

        # change name: add prefix 'val_' to key
        batch_logs = tool.add_pre_or_suf_on_key_for_dict(batch_logs, 'val_')

        return batch_logs


class WrapModel(ExtendModel):
    """
    WrapModel is a subclass of ExtendModel
    It is used to wrap a torch.nn.Module.
    """
    def __init__(self, model):
        super().__init__()
        module_list = nn.ModuleList([model])
        self.model = module_list[0]


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == '__main__':
    class mydataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return (torch.randn(15, 3), torch.randn(3, 3)), (torch.randn(30, 6), torch.randn(6, 3))


    dataset = mydataset()
    dataload = torch.utils.data.DataLoader(dataset, batch_size=32)
    for x, l in dataload:
        print(type(x))
        break