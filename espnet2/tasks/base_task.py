import argparse
import logging
import random
import shutil
import sys
from abc import ABC, abstractmethod
from io import TextIOBase
from pathlib import Path
from typing import Union, Any, Dict, Type, Tuple, Optional, Sequence

import configargparse
import numpy as np
import torch
import torch.nn
import torch.optim
import yaml
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet.asr.asr_utils import add_gradient_noise
from espnet2.schedulers.abs_scheduler import (
    AbsEpochScheduler, AbsBatchScheduler, AbsValEpochScheduler, )
from espnet2.train.abs_espnet_model import AbsESPNetModel
from espnet2.train.batch_sampler import create_batch_sampler
from espnet2.train.dataset import ESPNetDataset, our_collate_fn
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils.device_funcs import to_device
from espnet2.utils.get_default_kwargs import get_defaut_kwargs
from espnet2.utils.types import (
    int_or_none, str2bool, str_or_none, NestedDictAction, str2triple_str)


class BaseTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @typechecked
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='base parser',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        #  to provide --print_config mode. Instead of it, do as
        parser.set_defaults(required=['output_dir'])

        group = parser.add_argument_group('Common configuration')

        group.add_argument('--config', is_config_file=True,
                           help='config file path')
        group.add_argument('--print_config', action='store_true',
                           help='Print the config file and exit')

        group.add_argument(
            '--log_level', type=lambda x: str(x).upper(), default='INFO',
            choices=('INFO', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'),
            help='The verbose level of logging')

        group.add_argument('--output_dir', type=str_or_none, default=None)
        group.add_argument('--ngpu', type=int, default=0,
                           help='The number of gpus. 0 indicates CPU mode')
        group.add_argument('--seed', type=int, default=0,
                           help='Random seed')

        group = parser.add_argument_group('Trainer related')
        group.add_argument('--max_epoch', type=int, default=40,
                           help='The maximum number epoch to train')
        group.add_argument(
            '--train_dtype', default="float32",
            choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
            help='Data type for training. '
                 'O0,O1,.. flags require apex. '
                 'See https://nvidia.github.io/apex/amp.html#opt-levels')
        group.add_argument(
            '--patience', type=int_or_none, default=None,
            help='Number of epochs to wait without improvement '
                 'before stopping the training')
        group.add_argument('--early_stopping_criterion', type=str2triple_str,
                           default=('eval', 'loss', 'min'),
                           help='The criterion used for judging of '
                                'early stopping. '
                                'Give a pair referring '
                                'the phase, "train" or "eval",'
                                'the criterion name and the mode, '
                                '"min" or "max", e.g. "acc,max".')
        group.add_argument('--best_model_criterion', type=str2triple_str,
                           action='append',
                           default=[('train', 'loss', 'min'),
                                    ('eval', 'loss', 'min'),
                                    ('train', 'acc', 'max'),
                                    ('eval', 'acc', 'max'),
                                    ],
                           help='The criterion used for judging of '
                                'the best model. '
                                'Give a pair referring '
                                'the phase, "train" or "eval",'
                                'the criterion name, and '
                                'the mode, '
                                '"min" or "max", e.g. "acc,max".')

        group.add_argument('--grad_clip', type=float, default=5.,
                           help='Gradient norm threshold to clip')
        group.add_argument('--grad_noise', type=str2bool, default=False,
                           help='The flag to switch to use noise injection to '
                                'gradients during training')
        group.add_argument('--accum_grad', type=int, default=1,
                           help='The number of gradient accumulation')
        group.add_argument('--log_interval', type=int_or_none, default=None,
                           help='Show the logs every the number iterations in'
                                'each epochs at the training phase. '
                                'If None is given, the value of 1% of '
                                'the number of training iteration '
                                'is selected.')
        group.add_argument('--n_latest_history', type=int, default=20,
                           help='The number of latest epochs to save '
                                'the training state.')

        group = parser.add_argument_group(
            'Resuming or transfer learning related')

        def epoch_type(value: str) -> Optional[Union[str, int]]:
            if value == 'latest':
                return value
            elif value.lower() in ('none', 'null', 'nil'):
                return None
            else:
                v = int(value)
                if v < 0:
                    raise TypeError('must be 0 or more integer')
                return v

        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument(
            '--resume_epoch', type=epoch_type, default=None,
            help='The training starts from the specified epoch. '
                 '"latest" indicates the latest-epoch file found '
                 'in output_path')
        egroup.add_argument('--resume_path', type=str_or_none, default=None)

        group.add_argument('--pretrain_path', type=str_or_none, default=None)
        group.add_argument('--pretrain_key', type=str_or_none, default=None)

        group = parser.add_argument_group('BatchSampler related')
        group.add_argument(
            '--batch_size', type=int, default=20,
            help='The mini-batch size used for training')
        group.add_argument(
            '--eval_batch_size', type=int_or_none, default=None,
            help='If not given, the value of --batch_size is used')
        group.add_argument('--batch_type', type=str, default='seq',
                           choices=['const', 'seq', 'bin', 'frame'])
        group.add_argument(
            '--eval_batch_type', type=str_or_none, default=None,
            choices=['const', 'seq', 'batch_bin', None],
            help='If not given, the value of --batch_type is used')

        group.add_argument(
            '--train_shape_file', type=str, action='append', default=[])
        group.add_argument(
            '--eval_shape_file', type=str, action='append', default=[])

        group.add_argument(
            '--max_length', type=int, action='append', default=[])

        group = parser.add_argument_group('Dataset related')
        group.add_argument('--num_workers', type=int, default=1,
                           help='The number of workers used for DataLoader')
        group.add_argument(
            '--train_data_path_and_name_and_type',
            type=str2triple_str, action='append', default=[])
        group.add_argument(
            '--eval_data_path_and_name_and_type',
            type=str2triple_str, action='append', default=[])
        group.add_argument(
            '--train_preprocess', action=NestedDictAction, default=dict())
        group.add_argument(
            '--eval_preprocess', action=NestedDictAction, default=dict())

        group = parser.add_argument_group('Optimizer related')
        group.add_argument(
            '--optim', type=str, default='adadelta',
            choices=cls.optimizer_choices(), help='The optimizer type')
        group.add_argument(
            '--optim_conf', action=NestedDictAction, default=dict())

        group.add_argument(
            '--escheduler', type=str_or_none,
            choices=cls.epoch_scheduler_choices(),
            help='The epoch-scheduler type')
        group.add_argument(
            '--escheduler_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for the epoch scheduler')

        group.add_argument(
            '--bscheduler', type=str_or_none, default=None,
            choices=cls.batch_scheduler_choices(),
            help='The batch-scheduler-type')
        group.add_argument(
            '--bscheduler_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for the batch scheduler')
        return parser

    @classmethod
    @typechecked
    def exclude_opts(cls) -> Tuple[str, ...]:
        return ('required', 'print_config', 'config', 'ngpu',
                'log_level', 'output_dir')

    @classmethod
    @typechecked
    def get_default_config(cls) -> Dict[str, Any]:
        parser = BaseTask.add_arguments()
        args, _ = parser.parse_known_args()
        config = vars(args)
        # Excludes the options not to be shown
        for k in BaseTask.exclude_opts():
            config.pop(k)

        # Get the default arguments from the specified class
        # e.g. --print_config --optim adadelta
        optim_class = cls.get_optimizer_class(args.optim)
        optim_conf = get_defaut_kwargs(optim_class)
        optim_conf.update(config['optim_conf'])
        config['optim_conf'] = optim_conf

        if args.escheduler is not None:
            escheduler_class = cls.get_epoch_scheduler_class(args.escheduler)
            escheduler_conf = get_defaut_kwargs(escheduler_class)
            escheduler_conf.update(config['escheduler_conf'])
            config['escheduler_conf'] = escheduler_conf

        if args.bscheduler is not None:
            bscheduler_class = cls.get_batch_scheduler_class(args.bscheduler)
            bscheduler_conf = get_defaut_kwargs(bscheduler_class)
            bscheduler_conf.update(config['bscheduler_conf'])
            config['bscheduler_conf'] = bscheduler_conf

        return config

    @classmethod
    @typechecked
    def check_required(cls, args: argparse.Namespace):
        for k in vars(args):
            if '-' in k:
                raise RuntimeError(
                    f'Use "_" instead of "-": parser.add_arguments("{k}")')

        required = ', '.join(
            f'--{a}' for a in args.required if getattr(args, a) is None)

        if len(required) != 0:
            parser = cls.add_arguments()
            parser.print_help(file=sys.stderr)
            p = Path(sys.argv[0]).name
            print(file=sys.stderr)
            print(f'{p}: error: the following arguments are required: '
                  f'{required}', file=sys.stderr)
            sys.exit(2)

    @classmethod
    @typechecked
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> torch.nn.Module:
        raise NotImplementedError

    @classmethod
    @typechecked
    def optimizer_choices(cls) -> Tuple[str, ...]:
        choices = ('Adam', 'SGD', 'Adadelta', 'Adagrad', 'AdamW',
                   'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop')
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        return choices

    @classmethod
    @typechecked
    def get_optimizer_class(cls, name: str) -> Type[torch.optim.Optimizer]:
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'adam':
            return torch.optim.Adam
        elif name.lower() == 'sgd':
            return torch.optim.SGD
        elif name.lower() == 'adadelta':
            return torch.optim.Adadelta
        elif name.lower() == 'adagrad':
            return torch.optim.Adagrad
        elif name.lower() == 'adaw':
            return torch.optim.AdamW
        elif name.lower() == 'adamax':
            return torch.optim.Adamax
        elif name.lower() == 'asgd':
            return torch.optim.ASGD
        elif name.lower() == 'lbfgs':
            return torch.optim.LBFGS
        elif name.lower() == 'rmsprop':
            return torch.optim.RMSprop
        elif name.lower() == 'rprop':
            return torch.optim.Rprop
        else:
            raise RuntimeError(
                f'--optim must be one of {cls.optimizer_choices()}: '
                f'--optim {name}')

    @classmethod
    @typechecked
    def epoch_scheduler_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('ReduceLROnPlateau', 'LambdaLR', 'StepLR', 'MultiStepLR',
                   'ExponentialLR', 'CosineAnnealingLR')
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        choices += (None,)
        return choices

    @classmethod
    @typechecked
    def get_epoch_scheduler_class(cls, name: str) -> Type[AbsEpochScheduler]:
        """Schedulers change optim-parameters at the end of each epochs

        FIXME(kamo): EpochScheduler is confusing name.

        EpochScheduler:
            >>> for epoch in range(10):
            >>>     train(...)
            >>>     scheduler.step()

        ValEpochScheduler:
            >>> for epoch in range(10):
            >>>     train(...)
            >>>     val = validate(...)
            >>>     scheduler.step(val)
        """
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'reducelronplateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau
        elif name.lower() == 'lambdalr':
            return torch.optim.lr_scheduler.LambdaLR
        elif name.lower() == 'steplr':
            return torch.optim.lr_scheduler.StepLR
        elif name.lower() == 'multisteplr':
            return torch.optim.lr_scheduler.MultiStepLR
        elif name.lower() == 'exponentiallr':
            return torch.optim.lr_scheduler.ExponentialLR
        elif name.lower() == 'cosineannealinglr':
            return torch.optim.lr_scheduler.CosineAnnealingLR
        else:
            raise RuntimeError(
                f'--escheduler must be one of '
                f'{cls.epoch_scheduler_choices()}: --escheduler {name}')

    @classmethod
    @typechecked
    def batch_scheduler_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ('CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts')
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        choices += (None,)
        return choices

    @classmethod
    @typechecked
    def get_batch_scheduler_class(cls, name: str) -> Type[AbsBatchScheduler]:
        """Schedulers change optim-parameters after every updating

        FIXME(kamo): BatchScheduler is confusing name.

        BatchScheduler:
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()
        """
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'cycliclr':
            return torch.optim.lr_scheduler.CyclicLR
        elif name.lower() == 'onecyclelr':
            return torch.optim.lr_scheduler.OneCycleLR
        elif name.lower() == 'cosineannealingwarmrestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        else:
            raise RuntimeError(
                f'--bscheduler must be one of '
                f'{cls.batch_scheduler_choices()}: --bscheduler {name}')

    @classmethod
    @typechecked
    def print_config(cls, file: TextIOBase = sys.stdout):
        # Shows the config: e.g. python train.py asr --print_config
        config = cls.get_default_config()
        file.write(yaml.safe_dump(config, indent=4, sort_keys=False))

    @classmethod
    @typechecked
    def main(cls, args: argparse.Namespace = None,
             cmd: Sequence[str] = None) -> None:
        if args is None:
            parser = cls.add_arguments()
            args = parser.parse_args(cmd)
        if args.print_config:
            cls.print_config()
            sys.exit(0)
        cls.check_required(args)

        logging.basicConfig(
            level=args.log_level,
            format=
            '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

        # 1. Set random-seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        # 2. Build train-data-iterator
        if args.train_dtype in ('float32', 'O0', 'O1', 'O2', 'O3'):
            dtype = 'float32'
        else:
            dtype = args.train_dtype
        train_dataset = ESPNetDataset(
            args.train_data_path_and_name_and_type,
            args.train_preprocess, float_dtype=dtype)
        train_batch_sampler = create_batch_sampler(
            type=args.batch_type, shape_files=args.train_shape_file,
            max_lengths=args.max_length,
            batch_size=args.batch_size, shuffle=True)
        train_iter = DataLoader(dataset=train_dataset,
                                batch_sampler=train_batch_sampler,
                                collate_fn=our_collate_fn,
                                num_workers = args.num_workers)

        # 3. Build eval-data-iterator
        eval_dataset = ESPNetDataset(
            args.eval_data_path_and_name_and_type,
            args.eval_preprocess, float_dtype=dtype)
        if args.eval_batch_type is None:
            args.eval_batch_type = args.batch_type
        if args.eval_batch_size is None:
            args.eval_batch_size = args.batch_size
        eval_batch_sampler = create_batch_sampler(
            type=args.eval_batch_type, shape_files=args.eval_shape_file,
            max_lengths=args.max_length,
            batch_size=args.eval_batch_size, shuffle=False)
        eval_iter = DataLoader(dataset=eval_dataset,
                               batch_sampler=eval_batch_sampler,
                               collate_fn=our_collate_fn,
                               num_workers=args.num_workers)

        # 4. Build model, optimizer, scheduler
        model = cls.build_model(args=args)
        if not isinstance(model, AbsESPNetModel):
            raise RuntimeError(
                f'model must inherit AbsESPNetModel, but got {type(model)}')

        optimizer_class = cls.get_optimizer_class(args.optim)
        optimizer = optimizer_class(model.parameters(), **args.optim_conf)

        # 5. Build epoch_scheduler: invoked at every epochs
        # e.g. torch.optim.lr_scheduler.StepLR
        if args.escheduler is not None:
            epoch_scheduler_class = \
                cls.get_epoch_scheduler_class(args.escheduler)
            epoch_scheduler = \
                epoch_scheduler_class(optimizer, **args.escheduler_conf)
        else:
            epoch_scheduler = None

        # 6. Build batch_scheduler: invoked after every updating
        # e.g. torch.optim.lr_scheduler.CyclicLR
        if args.bscheduler is not None:
            batch_scheduler_class = \
                cls.get_batch_scheduler_class(args.bscheduler)
            batch_scheduler = \
                batch_scheduler_class(optimizer, **args.bscheduler_conf)
        else:
            batch_scheduler = None

        # 7. Dump "args" to config.yaml
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / 'config.yaml').open('w') as f:
            logging.info(
                f'Saving the configuration in {output_path / "config.yaml"}')
            yaml.safe_dump(vars(args), f)

        logging.info(f'Model:\n{model}')
        logging.info(f'Optimizer:\n{optimizer}')
        logging.info(f'Epoch scheduler: {epoch_scheduler}')
        logging.info(f'Batch scheduler: {batch_scheduler}')
        logging.info(f'Train Dataset: {train_dataset}')
        logging.info(f'Train BatchSampler: {train_batch_sampler}')
        logging.info(f'Eval Dataset: {eval_dataset}')
        logging.info(f'Eval BatchSampler: {eval_batch_sampler}')

        # 8. Model to device
        # FIXME(kamo): I wanna move this block into train(),
        #  but model.to() seems to need to be used before
        #  optimizer.load_state_dict(). I don't know why.
        # For apex supporting
        if args.train_dtype in ('O0', 'O1', 'O2', 'O3'):
            try:
                from apex import amp
            except ImportError:
                logging.error(f'You need to install apex. '
                              f'See https://github.com/NVIDIA/apex#linux')
                raise
            model, optimizer = \
                amp.initialize(model, optimizer, opt_level=args.train_dtype)
        if args.train_dtype in ('float16', 'float32', 'float64'):
            dtype = getattr(torch, args.train_dtype)
        else:
            dtype = torch.float32
        model = \
            model.to(dtype=dtype, device='cuda' if args.ngpu > 0 else 'cpu')

        reporter = Reporter()

        # 9. Loads states from saved files
        cls.load(model=model, optimizer=optimizer,
                 reporter=reporter, output_path=output_path,
                 batch_scheduler=batch_scheduler,
                 epoch_scheduler=epoch_scheduler,
                 resume_epoch=args.resume_epoch,
                 resume_path=args.resume_path,
                 pretrain_path=args.pretrain_path,
                 pretrain_key=args.pretrain_key,
                 loc='cuda' if args.ngpu > 0 else 'cpu')

        # 10. Start training
        if args.log_interval is None:
            log_interval = max(len(train_batch_sampler) // 100, 30)
        cls.run(model=model,
                optimizer=optimizer,
                train_iter=train_iter,
                eval_iter=eval_iter,
                reporter=reporter,
                output_path=output_path,
                batch_scheduler=batch_scheduler,
                epoch_scheduler=epoch_scheduler,
                ngpu=args.ngpu,
                max_epoch=args.max_epoch,
                train_dtype=args.train_dtype,
                patience=args.patience,
                grad_noise=args.grad_noise,
                accum_grad=args.accum_grad,
                grad_clip=args.grad_clip,
                log_interval=log_interval,
                n_latest_history=args.n_latest_history,
                early_stopping_criterion=args.early_stopping_criterion,
                best_model_criterion=args.best_model_criterion,
                )

    @classmethod
    @typechecked
    def load(cls,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             reporter: Reporter,
             output_path: Union[str, Path],
             batch_scheduler: AbsBatchScheduler = None,
             epoch_scheduler: AbsEpochScheduler = None,
             resume_epoch: Union[int, str] = None,
             resume_path: Union[str, Path] = None,
             pretrain_path: Union[str, Path] = None,
             pretrain_key: str = None,
             loc: str = 'cpu') -> None:
        # For resuming: Specify either resume_epoch or resume_path.
        #     - resume_epoch: Load from outdir/{}epoch.pt.
        #     - resume_path: Load from the specified path.
        # Find the latest epoch snapshot
        if resume_epoch == 'latest':
            resume_epoch = 0
            for p in output_path.glob('*epoch'):
                try:
                    n = int(p.stem.replace('epoch', ''))
                except TypeError:
                    continue
                if n > resume_epoch:
                    resume_epoch = n
        # If not found any snapshots, then nothing is done
        if resume_epoch == 0:
            resume_epoch = None

        if resume_epoch is not None or resume_path is not None:
            if resume_path is None:
                resume_path = output_path / f'{resume_epoch}epoch'
                logging.info(f'--resume_epoch {resume_epoch}: '
                             f'Loading from {resume_path}')
            else:
                logging.info(f'--resume_path {resume_path}: '
                             f'Loading from {resume_path}')

            for key, obj in [('model', model),
                             ('optimizer', optimizer),
                             ('reporter', reporter),
                             ('epoch_scheduler', epoch_scheduler),
                             ('batch_scheduler', batch_scheduler)]:
                _st = torch.load(resume_path / f'{key}.pt', map_location=loc)
                if obj is not None:
                    obj.load_state_dict(_st)

        # FIXME(kamo): Should be done in Task.build_model()?
        # For distillation, fine-tuning, transfer learning, etc.
        if pretrain_path is not None:
            if pretrain_key is None:
                obj = model
            else:
                def get_attr(obj: Any, key: str):
                    """

                    >>> class A(torch.nn.Module):
                    ...     def __init__(self):
                    ...         super().__init__()
                    ...         self.linear = torch.nn.Linear(10, 10)
                    >>> a = A()
                    >>> assert A.linear.weight is get_attr(A, 'linear.weight')

                    """
                    if key.strip() == '':
                        return obj
                    for k in key.split('.'):
                        obj = getattr(obj, k)
                    return obj

                obj = get_attr(model, pretrain_key)

            state_dict = obj.state_dict()
            pretrained_dict = torch.load(pretrain_path, map_location=loc)
            # Ignores the parameters not existing in the train-model
            pretrained_dict = \
                {k: v for k, v in pretrained_dict.items() if k in state_dict}
            state_dict.update(pretrained_dict)
            obj.load_state_dict(state_dict)

    @classmethod
    @typechecked
    def run(cls,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_iter,
            eval_iter,
            reporter: Reporter,
            output_path: Union[str, Path],
            batch_scheduler: AbsBatchScheduler = None,
            epoch_scheduler: AbsEpochScheduler = None,
            max_epoch: int = 30,
            patience: int = None,
            ngpu: int = 1,
            train_dtype: str = 'float32',
            grad_noise: bool = False,
            accum_grad: int = 1,
            grad_clip: float = 5.,
            log_interval: int = 200,
            n_latest_history: int = 20,
            early_stopping_criterion: Tuple[str, str, str] =
            ('eval', 'loss', 'min'),
            best_model_criterion: Sequence[Tuple[str, str, str]] =
            (('eval', 'loss', 'min'), ('train', 'loss', 'min')),
            ) -> None:

        # Starting training process from here
        start_epoch = reporter.get_epoch() + 1
        if start_epoch == max_epoch + 1:
            logging.warning(f'The training has already reached at '
                            f'max_epoch: {start_epoch}')

        best_epoch_dict = {}
        for iepoch in range(start_epoch, max_epoch + 1):
            logging.info(f'{iepoch}epoch started')

            reporter.set_epoch(iepoch)
            # 1. Train and eval for one-epoch
            with reporter.start('train') as sub_reporter:
                all_steps_are_invalid = cls.train(
                    model=model,
                    optimizer=optimizer,
                    iterator=train_iter,
                    reporter=sub_reporter,
                    scheduler=batch_scheduler,
                    ngpu=ngpu,
                    use_apex=train_dtype in ('O0', 'O1', 'O2', 'O3'),
                    grad_noise=grad_noise,
                    accum_grad=accum_grad,
                    grad_clip=grad_clip,
                    log_interval=log_interval,
                    )
            with reporter.start('eval') as sub_reporter:
                cls.eval(model=model,
                         iterator=eval_iter,
                         reporter=sub_reporter,
                         ngpu=ngpu)

            # 2. Scheduler step
            if epoch_scheduler is not None:
                # Controls opt-params by scheduler e.g. learning rate decay
                if isinstance(epoch_scheduler, AbsValEpochScheduler):
                    val = reporter.get_value(
                        'eval',
                        'acc' if reporter.has_key('eval', 'acc') else 'loss')
                    epoch_scheduler.step(val)
                else:
                    epoch_scheduler.step()

            # 3. Report the results
            reporter.logging()
            reporter.save_stats_plot(output_path / 'images')

            # 4. Save the snapshot
            for key, obj in [('model', model),
                             ('optimizer', optimizer),
                             ('reporter', reporter),
                             ('epoch_scheduler', epoch_scheduler),
                             ('batch_scheduler', batch_scheduler),
                             ]:
                (output_path / f'{iepoch}epoch').mkdir(
                    parents=True, exist_ok=True)
                torch.save(obj.state_dict() if obj is not None else None,
                           output_path / f'{iepoch}epoch' / f'{key}.pt')

            # 5. Saves the best model
            _improved = []
            for _phase, k, _mode in best_model_criterion:
                if reporter.has_key(_phase, k):
                    best_epoch, _ = \
                        reporter.best_epoch_and_value(_phase, k, _mode)
                    best_epoch_dict[(_phase, k)] = best_epoch
                    # Creates sym links if it's the best result
                    if best_epoch == iepoch:
                        p = output_path / f'{k}.best.pt'
                        if p.exists():
                            p.unlink()
                        p.symlink_to(Path(f'{iepoch}epoch') / f'model.pt')
                        _improved.append(f'{_phase}/{k}')
            if len(_improved) == 0:
                logging.info(f'There are no improvements in this epoch')
            else:
                logging.info(f'The best model has been updated: ' +
                             ', '.join(_improved))

            # 6. Remove the snapshot files keeping n-latest model
            for e in range(1, iepoch - n_latest_history + 1):
                p = output_path / f'{e}epoch'
                if p.exists() and all(best_epoch_dict.get((ph, k)) != e
                                      for ph, k, _ in best_model_criterion):
                    shutil.rmtree(p)
                    logging.info(f'The snapshot was removed: {p}')

            # 7. If any updating has happened, stops the training
            if all_steps_are_invalid:
                logging.warning(f'All gradients for each steps are invalid '
                                f'in this epoch. Something seems wrong. '
                                f'This training was stopped at {iepoch}epoch')
                break

            # 8. Check early stopping
            if patience is not None:
                _phase, _criterion, _mode = early_stopping_criterion
                if not reporter.has_key(_phase, _criterion):
                    raise RuntimeError(
                        f'{_phase}/{_criterion} is not found in stats')
                best_epoch, _ = \
                    reporter.best_epoch_and_value(_phase, _criterion, _mode)
                if iepoch - best_epoch > patience:
                    logging.info(
                        f'[Early stopping] The value has not been improved '
                        f'{iepoch - best_epoch} epochs continuously. '
                        f'The training was stopped at {iepoch}epoch')
                    break

        else:
            logging.info(f'The training was finished at {max_epoch} epochs ')

    @classmethod
    @typechecked
    def train(cls,
              model: torch.nn.Module,
              iterator,
              optimizer: torch.optim.Optimizer,
              reporter: SubReporter,
              scheduler: AbsBatchScheduler = None,
              ngpu: int = 1,
              use_apex: bool = False,
              grad_noise: bool = False,
              accum_grad: int = 1,
              grad_clip: float = 5.,
              log_interval: int = 200,
              ) -> bool:
        model.train()
        all_steps_are_invalid = True
        for iiter, batch in enumerate(iterator, 1):
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, 'cuda' if ngpu > 0 else 'cpu')
            if ngpu <= 1:
                # NOTE(kamo): data_parallel also should work with ngpu=1,
                # but for debuggability it's better to keep this block.
                loss, stats, weight = model(**batch)
            else:
                loss, stats, weight = \
                    data_parallel(model, (), range(ngpu), module_kwargs=batch)
                # Weighted averaging of loss from torch-data-parallel
                loss = (loss * weight.to(loss.dtype) / weight.sum()).mean(0)
                stats = {k: (v * weight.to(v.dtype) / weight.sum()).mean(0)
                         if v is not None else None
                         for k, v in stats.items()}
                weight = weight.sum()
            if weight.dim() not in (0, 1):
                raise RuntimeError(
                    f'weight must be 0 or 1 dimension: {weight.dim()}')

            reporter.register(stats, weight)

            if use_apex:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # gradient noise injection
            if grad_noise:
                add_gradient_noise(model, reporter.get_total_count(),
                                   duration=100, eta=1.0, scale_factor=0.55)

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip)
            if iiter % accum_grad == 0:
                if not np.isfinite(grad_norm):
                    logging.warning(f'The grad norm is {grad_norm}. '
                                    f'Skipping updating the model.')
                else:
                    all_steps_are_invalid = False
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if iiter % log_interval == 0:
                reporter.logging(nlatest=log_interval)
        return all_steps_are_invalid

    @classmethod
    @typechecked
    @torch.no_grad()
    def eval(cls, model: torch.nn.Module, iterator, reporter: SubReporter,
             ngpu: int = 1) -> None:
        model.eval()
        for batch in iterator:
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, 'cuda' if ngpu > 0 else 'cpu')
            if ngpu <= 1:
                _, stats, weight = model(**batch)
            else:
                _, stats, weight = \
                    data_parallel(model, (), range(ngpu), module_kwargs=batch)
                stats = {k: (v * weight.to(v.dtype) / weight.sum()).mean(0)
                         if v is not None else None
                         for k, v in stats.items()}
                weight = weight.sum()

            reporter.register(stats, weight)
