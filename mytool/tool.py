# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月22日
"""


import functools
import os
from typing import *
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import stat
import shutil
from tabulate import tabulate
from benedict import benedict
import optuna
import copy


###################################################################
###################################################################
#                         class
###################################################################
###################################################################


def import_class(module_path,class_name):
    """
    Dynamically imports one Class into namespace
    :param module_path: class module path
    :param class_name: class name must be the same as the origin
    :return:
        One Class definition
    """
    module = __import__(module_path,fromlist=[' '])
    return getattr(module,class_name)


###################################################################
###################################################################
#                         yaml
###################################################################
###################################################################


def load_yaml_args(yaml_filepath):
    """
    get args Dict
    :param yaml_filepath: str filepath
    :return:
        args: Dict
    """
    with open(yaml_filepath,encoding='utf-8') as f:
        args = yaml.load(f, yaml.Loader)
    return args


def save_yaml_args(yaml_filepath,data):
    with open(yaml_filepath,'w') as f:
        yaml.dump(data,f)


###################################################################
###################################################################
#                         func
###################################################################
###################################################################


def set_func_name(func,name):
    setattr(func,'__qualname__',name)
    return func


def get_func_name(func):
    """
    Gets the name of the function definition
    :param func:
    :return:
        func name
    """
    if hasattr(func,'__qualname__'):
        name_list = func.__qualname__.split('.')
        name = name_list[-1]
    else:
        name = func.__class__.__name__
    return name


###################################################################
###################################################################
#                         dict
###################################################################
###################################################################

def print_dicts_tablefmt(dicts:List[Dict], headers:Union[str,List[str]]='keys', tablefmt:str='fancy_grid'):
    """
    print the table of dicts.
    :param dicts: List[Dict] all dict must have same keys
    :param headers: first row of table
    :param tablefmt: args of tabulate. e.g. grid/pretty/presto/rst/orgtbl/github/simple
    :return:
        None
    """
    df = pd.DataFrame(dicts).T
    print(tabulate(df,headers=headers,tablefmt=tablefmt))


def has_hyperparameter(config_dict):
    """
    whether to have mytorch.MyHyperParameter
    :param config_dict: Dict
    :return:
        bool
    """
    for key in config_dict.keys():
        value = config_dict[key]
        if isinstance(value,dict):
            if 'type' in value.keys():
                return True
            else:
                if has_hyperparameter(value): return True
                else: pass
        else:
            pass
    return False


def modify_dict_with_hp(config_dict, hp, bool_tuner=True):
    """
    modify config_dict with hyperparameters for searching of keras_tuner.
    :param config_dict: config dictionary
    :param hp: hyperparameters
    :param bool_tuner: 是否正在tuner，如果是，调用hp方法，如果不是，hp为dict
    :return:
        modified config_dict
    """
    if not bool_tuner: #best hp
        best_hp_dict = hp if isinstance(hp,dict) else hp.values

    for key in config_dict.keys():
        value = config_dict[key]
        if isinstance(value,dict):
            if 'type' in value.keys(): #需要修改
                if value['type'] == 'int':
                    step = value['step']  if 'step' in value.keys() else None
                    config_dict[key] = hp.Int(key,value['min'],value['max'],step) if bool_tuner else best_hp_dict[key]
                elif value['type'] == 'float':
                    step = value['step'] if 'step' in value.keys() else None
                    config_dict[key] = hp.Float(key,value['min'],value['max'],step) if bool_tuner else best_hp_dict[key]
                elif value['type'] == 'list':
                    config_dict[key] = hp.Choice(key,value['list']) if bool_tuner else best_hp_dict[key]
                elif value['type'] == 'bool':
                    config_dict[key] = hp.Boolean(key) if bool_tuner else best_hp_dict[key]
                else:
                    raise Exception('unknown type!!!')
            else:
                config_dict[key] = modify_dict_with_hp(value,hp,bool_tuner)
        else: #一个数
            pass
    return config_dict


def modify_dict_with_trial(args, trial:Union[optuna.trial.Trial,optuna.trial.FrozenTrial]):
    """
    modify config_dict with hyperparameters for searching of optuna.
    Args:
        args: config dictionary
        trial: optuna.trial.Trial or optuna.trial.FrozenTrial

    Returns:
        modified config_dict
    """
    for key in args.keys():
        value = args[key]
        if isinstance(value, dict):
            if 'type' in value.keys(): # need modified
                cls = value['type']
                value.pop('type')
                if cls == 'int': # low high step log
                    args[key] = trial.suggest_int(key,**value)
                elif cls == 'float': # low high step log
                    args[key] = trial.suggest_float(key,**value)
                elif cls == 'discrete_uniform': # low high q
                    args[key] = trial.suggest_discrete_uniform(key,**value)
                elif cls == 'uniform': # low high
                    args[key] = trial.suggest_uniform(key,**value)
                elif cls == 'loguniform': # low high
                    args[key] = trial.suggest_loguniform(key,**value)
                elif cls == 'categorical': # choices
                    args[key] = trial.suggest_categorical(key,**value)
                else:
                    raise ValueError('cls must be in [int, float, discrete_uniform, uniform, loguniform, categorical]')
            else:
                args[key] = modify_dict_with_trial(value, trial)
        else:
            pass
    return args


def transform_dict_to_search_space(args):
    """
    transform config_dict to search_space for optuna.
    Args:
        args: config dictionary

    Returns:
        search_space: search_space for optuna. Dict
    """
    args = copy.deepcopy(args)
    search_space = {}
    for key in args.keys():
        value = args[key]
        if isinstance(value, dict):
            if 'type' in value.keys():
                cls = value['type']
                value.pop('type')
                if cls == 'int':
                    step = value['step'] if 'step' in value.keys() else 1
                    search_space[key] = np.arange(value['low'],value['high'],step).tolist()
                elif cls == 'float':
                    step = value['step'] if 'step' in value.keys() else 1.0
                    search_space[key] = np.arange(value['low'],value['high'],step).tolist()
                elif cls == 'discrete_uniform':
                    step = value['q'] if 'q' in value.keys() else 1.0
                    search_space[key] = np.arange(value['low'],value['high'],step).tolist()
                elif cls == 'uniform':
                    search_space[key] = np.linspace(value['low'],value['high'],10).tolist()
                elif cls == 'loguniform':
                    search_space[key] = np.logspace(value['low'],value['high'],10).tolist()
                elif cls == 'category':
                    search_space[key] = value['choices']
                else:
                    raise ValueError('cls must be in [int, float, discrete_uniform, uniform, loguniform, categorical]')
            else:
                deep_return = transform_dict_to_search_space(value)
                if len(deep_return.keys()) > 0:
                    search_space[key] = deep_return
        else:
            pass

    return_dict = {}
    search_space = benedict(search_space).flatten('*')
    for key in search_space.keys():
        new_key = key.split('*')[-1]
        value = search_space[key]
        return_dict[new_key] = value
    return return_dict


def config_dict_list_sort_with_argidx(config_dict_list, argidx):
    """
    sort config_dict_list with argidx
    Args:
        config_dict_list: List[Dict]
        argidx: List[int]

    Returns:
        sorted config_dict_list
    """
    return_list = []
    for arg in argidx:
        return_list.append(config_dict_list[int(arg)])

    return return_list


def add_pre_or_suf_on_key_for_dict(obj_dict, prefix=None, suffix=None, checkpoint_str=None):
    """
    add prefix or suffix for every key of dict.
    :param obj_dict: dict
    :param prefix: str
    :param suffix: str
    :param checkpoint_str: str if key contains checkpoint_str, it will be modified.
    :return:
        return_dict
    """
    if prefix is None and suffix is None:
        raise Exception('prefix or suffix must have str value')

    return_dict = {}

    if prefix is not None:
        for key in obj_dict.keys():
            if checkpoint_str is not None:
                if key.find(checkpoint_str) != -1:
                    if suffix is not None: #both add
                        return_dict[prefix + key + suffix] = obj_dict[key]
                    else: #add prefix
                        return_dict[prefix + key] = obj_dict[key]
                else:
                    return_dict[key] = obj_dict[key]
            else:
                if suffix is not None: #both add
                    return_dict[prefix + key + suffix] = obj_dict[key]
                else: #add prefix
                    return_dict[prefix + key] = obj_dict[key]
    else: #suffix is not None
        for key in obj_dict.keys():
            if checkpoint_str is not None:
                if key.find(checkpoint_str) != -1: #add suffix
                    return_dict[key + suffix] = obj_dict[key]
                else:
                    return_dict[key] = obj_dict[key]
            else:#add suffix
                return_dict[key + suffix] = obj_dict[key]

    return return_dict


def complete_dict_path(dict_args:Dict,work_dir:str,check_suffix:str='path'):
    """
    complete relative path in a config dictionary with work_dir
    :param dict_args: config dictionary
    :param work_dir: path which will be added to prefix of relative path
    :param check_suffix: key which ends with it to complete
    :return:
        completed config dictionary
    """
    return_dict = {}
    for key,value in dict_args.items():
        if key.endswith(check_suffix):
            return_dict[key] = os.path.join(work_dir,value)
        else:
            return_dict[key] = value
    return return_dict


def remove_dict_something_value(dict_args, something, flat_sep='*'):
    """
    remove something value in a config dictionary
    :param dict_args: config dictionary
    :param something: something value
    :param flat_sep: separator of flatten
    :return:
        completed config dictionary
    """
    return_benedict = False
    if isinstance(dict_args,benedict):
        return_benedict = True
    return_dict = benedict(dict_args)
    return_dict = return_dict.flatten(flat_sep).filter(lambda k,v: v != something).unflatten(flat_sep)
    if return_benedict:
        return return_dict
    else:
        return return_dict.dict()


def remove_dict_None_value(dict_args, flat_sep='*'):
    """
    remove None value in a config dictionary
    :param dict_args: config dictionary
    :param flat_sep: separator of flatten
    :return:
        completed config dictionary
    """
    return remove_dict_something_value(dict_args,None,flat_sep)



###################################################################
###################################################################
#                         os
###################################################################
###################################################################


def remove_prefile_in_dir(dir, num_check_threshold, remove_ratio):
    allfilenames = os.listdir(dir)
    allfilepath = [os.path.join(dir, filename) for filename in allfilenames]
    if len(allfilepath) > num_check_threshold:
        allfilepath.sort()  # 从小到大 从旧到新
        # 删除最旧的几个
        need_remove_num = int(remove_ratio * len(allfilepath))
        for i in range(need_remove_num):
            os.chmod(allfilepath[i], stat.S_IWRITE)
            if os.path.isfile(allfilepath[i]):
                os.remove(allfilepath[i])
            elif os.path.isdir(allfilepath[i]):
                shutil.rmtree(allfilepath[i])
            else:
                raise Exception('Unknown Error!')
    else:
        pass


def get_basename_split_ext(path):
    """
    get basename except ext.
    :param path: str
    :return:
        basename
    """
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]


###################################################################
###################################################################
#                         datetime
###################################################################
###################################################################


def get_datetime_name():
    """
    get datetime name for save configuration
    :return:
        timestamp: datetime name
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
    return timestamp


def get_date_range_start_end_pd(start_time_str,end_time_str,freq):
    """
    :param start_time_str: time_str
    :param end_time_str: time_str
    :param freq: pd.Timedelta e.g. 1D 1W 1min
        Y: 年(末)
        Q: 季度(末)
        M: 月(末)
        W: 星期
        D: 天
        B: 工作日
        H: 小时
        T(min): 分钟
        S: 秒
        L(ms): 毫秒
        U(us): 微秒
        N: 纳秒
    :return:
        date_list: 1-D [start_time_str:end_time_str:freq]
    other_applications:
        date_list.dayofyear .weekofyear 一年中第几天(从1算) 第几个星期(从1算，但可能有去年星期)
                .dayofweek .weekday 一星期中第几天(0星期一，6星期日)
        more information url:https://pandas.pydata.org/docs/reference/arrays.html#datetime-data
    """
    return pd.date_range(start_time_str,end_time_str,freq=freq)


def get_date_range_start_step_pd(start_time_str,freq,steps):
    """
    :param start_time_str: time_str
    :param freq: pd.Timedelta e.g. 1D 1W 1min
        Y: 年(末)
        Q: 季度(末)
        M: 月(末)
        W: 星期
        D: 天
        B: 工作日
        H: 小时
        T(min): 分钟
        S: 秒
        L(ms): 毫秒
        U(us): 微秒
        N: 纳秒
    :param steps: int
    :return:
        date_list: 1-D [start_time_str::freq] len=steps
    """
    return pd.date_range(start_time_str,periods=steps,freq=freq)


def get_latest_datetime_checkpoint(checkpoint_dir):
    """
    get latest checkpoint name from given checkpoint_dir
    :param checkpoint_dir: file to save
    :return:
        latest file name
    """
    filename_list = os.listdir(checkpoint_dir)
    filename_list.sort() #小 -> 大
    return os.path.join(checkpoint_dir,filename_list[-1])


###################################################################
###################################################################
#                         Text
###################################################################
###################################################################

def map_bytes_to_str(x_or_li):
    """
    将bytes或List[bytes]解码
    :param x_or_li: bytes or List[bytes]
    :return:
        x_or_li: decoded
    """
    if isinstance(x_or_li,bytes):
        return x_or_li.decode()
    elif isinstance(x_or_li,list):
        return list(map(map_bytes_to_str,x_or_li))
    else:
        raise Exception('must contain bytes!!')