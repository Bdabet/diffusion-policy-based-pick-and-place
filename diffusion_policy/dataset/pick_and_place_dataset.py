from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
import threading
import torchvision.transforms.functional as TF
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class PickPlaceDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
            use_flags=False,
            enable_augmentation=False, 
            aug_brightness_range=(0.7, 1.3),
            aug_contrast_range=(0.7, 1.3),
            aug_gamma_range=(0.7, 1.5),
            aug_probability=1.0
        ):
        assert os.path.isdir(dataset_path)

        # Initialize augmenter if enabled
        self.augmenter = None
        if enable_augmentation:
            self.augmenter = TorchvisionAugmenter(
                brightness_range=aug_brightness_range,
                contrast_range=aug_contrast_range,
                gamma_range=aug_gamma_range,
                probability=aug_probability
            )
            print(f"Image augmentation ENABLED (brightness={aug_brightness_range}, "
                  f"contrast={aug_contrast_range}, gamma={aug_gamma_range})")
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        print("dataset path used", dataset_path)
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        print("exception raised")
                        
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                use_flags=use_flags
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()


        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            
        if use_flags:
                    
            flags_keys = list()
            flags_shape_meta = shape_meta["flags"]
            for key, attr in flags_shape_meta.items():
                flags_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
            

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.use_flags = use_flags
        if use_flags:
            self.flags_keys = flags_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        val_set.augmenter = None  # disable augmentation for val set
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()

        for key in self.rgb_keys:
            # Load images: shape (T, H, W, C) uint8
            images = data[key][T_slice]
            
            # ============================================================
            # AUGMENTATION HAPPENS HERE - Per frame, thread-safe
            # ============================================================
            if self.augmenter is not None:
                # Augment each frame in the sequence
                augmented_images = []
                for t in range(len(images)):
                    img = images[t]  # (H, W, C) uint8
                    img_aug = self.augmenter.augment_image(img)
                    augmented_images.append(img_aug)
                images = np.stack(augmented_images, axis=0)  # (T, H, W, C)
            # ============================================================

            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        if self.use_flags:
            flag_dict = dict()
            for key in self.flags_keys:
                flag_dict[key] = data[key].astype(np.float32)
                del data[key]
        
        action = data['action'].astype(np.float32)

        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        if self.use_flags:
            torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'flags': dict_apply(flag_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
            }
        else:
            torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
            }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, shape_meta, store, use_flags=False):
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            out_resolutions[key] = (w,h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,),(4,),(6,),(7,)]

    if use_flags:
        flags_keys = list()
        flags_shape_meta = shape_meta["flags"]
        for key, attr in flags_shape_meta.items():
            flags_keys.append(key)

    action_shape = tuple(shape_meta['action']['shape'])
    assert action_shape in [(5,),(7,),(8,)]

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        # Only include flags_keys if use_flags is True
        lowdim_and_action_keys = lowdim_keys + ['action']
        if use_flags:
            lowdim_and_action_keys += flags_keys
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_and_action_keys,
            image_keys=rgb_keys
        )

    # transform lowdim dimensions for x,y motion form
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
    
    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])



    return replay_buffer

class TorchvisionAugmenter:
    """
    Thread-safe augmentation using torchvision transforms.
    """
    def __init__(
        self,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        gamma_range=(0.7, 1.5),
        probability=1.0
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.probability = probability
        self._thread_local = threading.local()
    
    def _get_rng(self):
        """Get thread-local random number generator."""
        if not hasattr(self._thread_local, 'rng'):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                seed = worker_info.id + torch.initial_seed() % 2**32
            else:
                seed = threading.get_ident()
            self._thread_local.rng = np.random.RandomState(seed)
        return self._thread_local.rng
    
    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply augmentations using torchvision (100% thread-safe).
        
        Args:
            img: (H, W, C) uint8 numpy array
        Returns:
            Augmented (H, W, C) uint8 numpy array
        """
        rng = self._get_rng()
        
        # Random skip
        if rng.random() > self.probability:
            return img
        
        # Convert to torch tensor (C, H, W) float32 [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply augmentations using torchvision
        if self.brightness_range is not None:
            brightness_factor = rng.uniform(*self.brightness_range)
            img_tensor = TF.adjust_brightness(img_tensor, brightness_factor)
        
        if self.contrast_range is not None:
            contrast_factor = rng.uniform(*self.contrast_range)
            img_tensor = TF.adjust_contrast(img_tensor, contrast_factor)
        
        if self.gamma_range is not None:
            gamma = rng.uniform(*self.gamma_range)
            img_tensor = TF.adjust_gamma(img_tensor, gamma)
        
        # Convert back to numpy uint8
        img_tensor = torch.clamp(img_tensor * 255.0, 0, 255)
        img_aug = img_tensor.permute(1, 2, 0).byte().numpy()
        
        return img_aug


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_robomimic_real_image_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')
