import contextlib
from typing import Callable, Optional, Dict, Any, Tuple, Union
from micro_config import ConfigScript, MetaConfig, ConfigScriptDict
from dataclasses import dataclass, asdict
import tree
from data import Seq2SeqDataset, Seq2SeqIterableDataset, Dataset, dataloader
from collections import deque
import jax
import os
import pickle as pkl
from utils.logs import reduce_logs, label_logs, pool_logs, log
from tqdm.auto import tqdm
import wandb
from jax.experimental.maps import Mesh
from jax.random import KeyArray
from core import TKInference, TKInferenceConfig, TKTrain, TKTrainConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from subprocess import call
from gcloud import gcloud_save, gcloud_save_str, init_gcloud

@dataclass
class EvaluateLossConfig(ConfigScript):
    eval_dataset: ConfigScript
    inference: Union[TKInferenceConfig, TKTrainConfig]
    rng: int
    bsize: int
    prefetch_batches: Optional[int]
    eval_batches: Optional[int]
    verbose: bool

    def unroll(self, metaconfig: MetaConfig):
        if isinstance(self.inference, TKTrainConfig):
            _, inference, _, mesh = self.inference.unroll(metaconfig)
        else:
            inference, _, mesh = self.inference.unroll(metaconfig)
        return {
            'eval_dataset': self.eval_dataset.unroll(metaconfig), 
            'inference': inference, 
            'mesh': mesh, 
            'rng': jax.random.PRNGKey(self.rng), 
            'bsize': self.bsize, 
            'prefetch_batches': self.prefetch_batches, 
            'eval_batches': self.eval_batches, 
            'verbose': self.verbose, 
        }

def evaluate_loss(*, eval_dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
                  inference: TKInference, mesh: Optional[Mesh], rng: KeyArray, bsize: int, 
                  prefetch_batches: Optional[int], eval_batches: Optional[int], verbose: bool):

        # load model
        if mesh is None:
            mesh = contextlib.nullcontext
        
        # setup evaluator loop state
        eval_logs = []

        # eval on batches
        with mesh:
            rng, new_rng = jax.random.split(rng)
            d = dataloader(new_rng, eval_dataset, bsize, prefetch_batches=prefetch_batches, trunc=True)
            for i, (items, _) in enumerate(d):
                
                # conditionally terminate early
                if eval_batches is not None and i >= eval_batches:
                    break

                # get eval logs
                loss, _, _ = inference.eval_log_probs_from_tokens(items['input_ids'], items['decoder_input_ids'])
                eval_logs.append({'loss': loss})
        
        # gather and postproc eval logs
        eval_logs = pool_logs(reduce_logs(eval_logs))

        if verbose:
            print(eval_logs)

        return eval_logs

@dataclass
class TrainLoopConfig(ConfigScript):
    train_dataset: ConfigScript
    trainer: TKTrainConfig
    rng: int
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    prefetch_batches: Optional[int]
    log_every: int
    eval_every: int
    save_every: Optional[int]
    save_only_at_end: bool
    use_wandb: bool
    wandb_project: str
    wandb_run_name: Optional[str]
    verbose: bool
    shuffle: bool
    push_script: Optional[str]
    use_bucket: bool = False

    def unroll(self, metaconfig: MetaConfig):
        trainer, inference, model, mesh = self.trainer.unroll(metaconfig)
        return {
            'train_dataset': self.train_dataset.unroll(metaconfig), 
            'trainer': trainer, 
            'inference': inference, 
            'model': model, 
            'mesh': mesh, 
            'evaluator': None, 
            'rng': jax.random.PRNGKey(self.rng), 
            'save_dir': metaconfig.convert_path(self.save_dir), 
            'max_checkpoints': self.max_checkpoints, 
            'epochs': self.epochs, 
            'max_steps': self.max_steps, 
            'bsize': self.bsize, 
            'prefetch_batches': self.prefetch_batches, 
            'log_every': self.log_every, 
            'eval_every': self.eval_every, 
            'save_every': self.save_every, 
            'save_only_at_end': self.save_only_at_end, 
            'use_wandb': self.use_wandb, 
            'wandb_project': self.wandb_project, 
            'wandb_run_name': self.wandb_run_name, 
            'wandb_config': asdict(self), 
            'verbose': self.verbose,
            'shuffle': self.shuffle,
            'push_script': self.push_script,
            'use_bucket': self.use_bucket
        }

def get_checkpoint_path(exp_dir, step):
    if jax.device_count() == jax.local_device_count():
        save_dir_path = os.path.join(exp_dir, 'outputs/model_%d' % (step+1))
    else:
        # separate folder for each shard
        save_dir_path = os.path.join(exp_dir, 'outputs/model_%d_h%d' % (step+1, int(jax.process_index())))

    return save_dir_path

def train_model(*, train_dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
                trainer: TKTrain, inference: TKInference, model: FlaxPreTrainedModel, 
                mesh: Optional[Mesh], evaluator: Optional[Callable[[TKInference], Tuple[float, Dict[str, Any]]]], 
                rng: KeyArray, save_dir: Optional[str], max_checkpoints: Optional[int], 
                epochs: int, max_steps: Optional[int], bsize: int, prefetch_batches: Optional[int], 
                log_every: int, eval_every: Optional[int], save_every: Optional[int], save_only_at_end: bool, 
                use_wandb: bool, wandb_project: str, wandb_run_name: Optional[str], 
                wandb_config: Optional[Any], verbose: bool, shuffle: bool, push_script: Optional[str],
                use_bucket: bool):
        
        # initalize wandb
        if use_wandb and jax.process_index() == 0:
            wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config, reinit=True)

        # save mesh info
        # When loading parameters sharded across multiple hosts, assume that the mesh has the same shape,  
        # and that the parameters for each process id are on the same process id as they were at parameter saving time.
        # We save the mesh info so that this information can be recovered.
        # This info can also be used to re-combine the params with `utils.mp_utils.combine_host_param_shards`.
        if save_dir is not None and mesh is not None:
            with open(os.path.join(save_dir, 'system_mesh.pkl'), 'wb') as f:
                pkl.dump({'mesh': tree.map_structure(lambda x: {'id': int(x.id), 'process_index': int(x.process_index)}, mesh.devices.tolist()), 
                          'process_index': int(jax.process_index()), 'process_count': int(jax.process_count())}, f)
        elif mesh is None:
            mesh = contextlib.nullcontext

        # initalize training loop state
        train_logs = []
        best_perf = float('inf')
        saved_checkpoints = deque([])
        step = 0
        steps_per_epoch = len(train_dataset) // bsize if isinstance(train_dataset, Dataset) else None

        # train loop
        with mesh:
            for epoch in tqdm(range(epochs), disable=jax.process_index() > 0):
                rng, new_rng = jax.random.split(rng)
                if not shuffle:
                    new_rng = None
                d = dataloader(new_rng, train_dataset, bsize, prefetch_batches=prefetch_batches, trunc=True)
                for items, _ in tqdm(d, total=steps_per_epoch, disable=jax.process_index() > 0):
                    
                    # step model and get training logs
                    rng, new_rng = jax.random.split(rng)
                    loss = trainer.train_step_from_tokens(items['input_ids'], items['decoder_input_ids'], new_rng)
                    train_logs.append({'loss': loss})
                    
                    # publish training logs
                    if (step + 1) % log_every == 0:
                        logs = reduce_logs(train_logs)
                        logs = pool_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
                        if jax.process_index() == 0:
                            log(logs, use_wandb)
                        train_logs = []
                    
                    # periodically save checkpoint
                    if save_dir is not None and save_every is not None and (step + 1) % save_every == 0 and (not save_only_at_end):
                        if verbose:
                            print('saving checkpoint...')

                        if use_bucket:
                            init_gcloud()

                            exp_dir = os.path.normpath(save_dir).split(os.sep)
                            exp_dir = [x for x in exp_dir if len(x) > 0][-2]

                            save_dir_path = get_checkpoint_path(exp_dir, step)

                            gcloud_save(jax.device_get(trainer.params), save_dir_path, 'flax_model.msgpack')
                            gcloud_save_str(model.config.to_json_string(use_diff=False), save_dir_path, 'config.json')
                        else:
                            # conditionally delete old checkpoints
                            if (max_steps is not None) and (len(saved_checkpoints) >= max_checkpoints):
                                os.system('rm -rf %s' % (saved_checkpoints.popleft()))

                            model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
                            model.save_pretrained(
                                model_dir, 
                                params=jax.device_get(trainer.params), 
                            )
                            saved_checkpoints.append(model_dir)
                            if verbose:
                                print('saved.')

                    # conditionally terminate
                    if max_steps is not None and (step + 1) >= max_steps:
                        break

                    step += 1
                
                # conditionally terminate
                if max_steps is not None and (step + 1) >= max_steps:
                    break
        
        # save final checkpoint
        if save_dir is not None and save_only_at_end:
            if use_bucket:
                init_gcloud()
                
                exp_dir = os.path.normpath(save_dir).split(os.sep)
                exp_dir = [x for x in exp_dir if len(x) > 0][-2]

                save_dir_path = get_checkpoint_path(exp_dir, step)

                gcloud_save(jax.device_get(trainer.params), save_dir_path, 'flax_model.msgpack')
                gcloud_save_str(model.config.to_json_string(use_diff=False), save_dir_path, 'config.json')
            else:
                # conditionally delete old checkpoints
                if (max_steps is not None) and (len(saved_checkpoints) >= max_checkpoints):
                    os.system('rm -rf %s' % (saved_checkpoints.popleft()))

                model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
                model.save_pretrained(
                    model_dir, 
                    params=jax.device_get(trainer.params), 
                )
                saved_checkpoints.append(model_dir)
                if verbose:
                    print('saved.')

        # stop wandb
        if use_wandb and jax.process_index() == 0:
            wandb_run.finish()
