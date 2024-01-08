# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年11月24日
"""
import numpy as np
import copy
import time
import optuna
from typing import *
from . import callback
from . import tool

class MyHyperParameter:
    """
    set hyperparameters for config
    if bool_new is True, every time it will return a new value from Int/Float/Boolean/Choice
    """
    def __init__(self, convergence_factor=1.0001):
        super().__init__()
        self.convergence_factor = convergence_factor
        self.last_int_dict = {}
        self.last_float_dict = {}
        self.last_bool_dict = {}
        self.last_choice_dict = {}
        self.bool_new = True
        self.all_name_weights = {}
        self.all_name_candi_list = {}

    def Int(self, name, min, max, step=1):
        candi_list = list(range(min,max+1,step))
        #收集用于调整各个候选参数的权重，初始化为1
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new: #第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))#random.choice(candi_list)
            self.last_int_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_int_dict[name]

    def Float(self, name, min, max, step=1.):
        candi_list = list(np.arange(min,max,step))
        # 收集用于调整各个候选参数的权重
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new: #第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))
            self.last_float_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_float_dict[name]

    def Boolean(self, name):
        candi_list = [False, True]
        # 收集用于调整各个候选参数的权重
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new: #第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))
            self.last_bool_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_bool_dict[name]

    def Choice(self, name, candi_list):
        # 收集用于调整各个候选参数的权重
        if name not in self.all_name_weights.keys():
            self.all_name_candi_list[name] = np.array(candi_list)
            self.all_name_weights[name] = np.ones_like(candi_list)

        if self.bool_new:  # 第一次
            selected = np.random.choice(candi_list,
                                        p=self.normalize(self.all_name_weights[name]))
            self.last_choice_dict[name] = selected.tolist()
            return selected
        else:
            return self.last_choice_dict[name]

    def normalize(self, np_list):
        rowsum = np.sum(np_list)
        norm = np_list/rowsum
        return norm

    def update_configdict_weight(self, configdict):
        for name in configdict.keys():
            value = configdict[name]
            candi_list = self.all_name_candi_list[name]
            total_num = len(candi_list)
            #找到对应值的位置idx
            id = np.where(candi_list == value)
            id = int(id[0])
            #列出更新权重
            multi_w_leftlist =  np.linspace(1,self.convergence_factor,id,endpoint=False)
            multi_w_rightlist = np.linspace(self.convergence_factor,1,total_num-id)
            multi_w = np.concatenate([multi_w_leftlist,multi_w_rightlist],axis=0)
            self.all_name_weights[name] = self.all_name_weights[name] * multi_w

    def set_bool_new(self, bool_new):
        self.bool_new = bool_new

    def get_current_configdict(self):
        return_dict = {}
        return_dict.update(self.last_int_dict)
        return_dict.update(self.last_float_dict)
        return_dict.update(self.last_bool_dict)
        return_dict.update(self.last_choice_dict)
        return return_dict.copy()


class MyTuner:
    def __init__(self, max_trials, executions_per_trial=1, mode='min', quiet=False):
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.mode = mode
        self.hp = MyHyperParameter()
        self.configdict_list = []
        self.flag_list = []
        self.callbacklist = callback.CallbackList([callback.MyTunerPrintCallback(max_trials, executions_per_trial, quiet)])

    def run_trial(self, hp, **kwargs):
        #it will be rewritten.
        #运行一次实验，返回要检测的指标value
        pass

    def search(self, **kwargs):
        #运行所有trial，每个测验执行几次，统计平均值，收集指标和对应配置

        #tuner begin
        self.callbacklist.on_train_begin()

        #init
        tuner_start_time = time.time()

        for tri_idx in range(1, self.max_trials+1):

            #trial begin
            self.callbacklist.on_epoch_begin(tri_idx)

            #collect flag every trial
            per_trial_flags = []
            for ex_idx in range(1, self.executions_per_trial+1, 1):

                #execution begin
                self.callbacklist.on_train_batch_begin(ex_idx)

                #init execution
                batch_start_time = time.time()

                #same configdict for execution but new configdict for new trial.
                #for run_trial func
                if ex_idx == 1:
                    self.hp.set_bool_new(bool_new=True)
                else:
                    self.hp.set_bool_new(bool_new=False)

                #run once
                #it must be deepcopy, dict can not be modified in origin dict.
                copy_kwargs = copy.deepcopy(kwargs)
                flag = self.run_trial(self.hp, **copy_kwargs)

                # collect flag
                per_trial_flags.append(flag)

                #execution end
                self.callbacklist.on_train_batch_end(ex_idx, {'flag':flag,
                                                              'time':time.time()-batch_start_time})

            #collect configdict and mean flag
            current_trial_configdict = self.hp.get_current_configdict()
            current_trial_flag = float(np.mean(per_trial_flags))
            self.configdict_list.append(current_trial_configdict)
            self.flag_list.append(current_trial_flag)

            #sort for print
            self.sort_flag_and_configdict()

            #trial end 最佳配置、最佳flag、以及截止目前花费的tuner时间
            self.callbacklist.on_epoch_end(tri_idx, {'best_flag':self.flag_list[0],
                                                     'best_configdict':self.configdict_list[0],
                                                     'cur_flag':current_trial_flag,
                                                     'cur_configdict':current_trial_configdict,
                                                     'time': time.time()-tuner_start_time})

        #tuner end
        self.callbacklist.on_train_end()

    def sort_flag_and_configdict(self):
        # 按照mode排序，如果max，从大到小，如果min，从小到大
        self.flag_list = np.array(self.flag_list)
        if self.mode == 'min':
            argidx = np.argsort(self.flag_list)
        elif self.mode == 'max':
            argidx = np.argsort(-1 * self.flag_list)
        else:
            raise Exception('mode must be min or max')
        self.flag_list = list(self.flag_list[argidx])
        self.configdict_list = tool.config_dict_list_sort_with_argidx(self.configdict_list,argidx)
        #更新各个参数的权重
        self.hp.update_configdict_weight(self.configdict_list[0])

    def get_best_hyperparameters(self):
        #返回按最优到最差的配置列表
        return self.configdict_list

    def get_best_flags(self):
        return self.flag_list


class CombinePruner(optuna.pruners.BasePruner):
    """
    combine pruner
    if and_or is and, all pruners must be True, it will return True
    if and_or is or, one pruner is True, it will return True
    """
    def __init__(self, pruner_list: List[optuna.pruners.BasePruner], and_or: str='and'):
        super().__init__()
        self.and_or = and_or
        self.pruner_list = pruner_list

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        if self.and_or == 'and':
            for pruner in self.pruner_list:
                if not pruner.prune(study, trial):
                    return False
            return True
        elif self.and_or == 'or':
            for pruner in self.pruner_list:
                if pruner.prune(study, trial):
                    return True
            return False
        else:
            raise Exception('and_or must be and or or')