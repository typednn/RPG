# common hooks; like saving the trajectories and so on
import abc
from typing import List
from tools.config import Configurable
from tools.utils import logger


class HookBase:
    # one can config the hooks later .
    def __init__(self) -> None:
        pass

    def init(self, trainer):
        pass

    def on_epoch(self, trainer, **locals_):
        pass

HOOKS = {}
def as_hook(f):
    global HOOKS
    HOOKS[f.__name__] = f
    return f


class RLAlgo(abc.ABC):
    def __init__(self, obs_rms, hooks) -> None:
        super().__init__()
        self.epoch_id = 0
        self.total = 0

        self.obs_rms = obs_rms

        self.mode='training'


        self.hooks = hooks
        for i in hooks:
            i.init(self)

    def start(self, env, **kwargs):
        obs, timestep = env.start(**kwargs)
        obs = self.norm_obs(obs, True)
        return obs, timestep

    def step(self, env, action):
        data = env.step(action)
        # if 'success' in data['info']:
        #     data['success'] = data['info'].pop('success')
        obs = self.norm_obs(data.pop('obs'), True)
        data['next_obs'] = self.norm_obs(data['next_obs'], False)
        return data, obs

    def norm_obs(self, x, update=True):
        if self.obs_rms:
            if update and self.mode == 'training':
                self.obs_rms.update(x) # always update during training...
            x = self.obs_rms.normalize(x)

        if isinstance(x[0], dict):
            x = list(x)  # don't know what happend ..
        return x

    def sample(self, p):
        if self.mode == 'training':
            return p.sample() #TODO: not differentiable now.
        elif self.mode == 'sample':
            return p.sample()
        raise NotImplementedError


    # @abc.abstractmethod
    # def modules(self, **kwargs):
    #     pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs):
        pass

    def call_hooks(self, locals_):
        self.mode = 'sample'

        traj = locals_['traj']
        self.total += traj.n
        #print(self.total, )
        logger.logkvs_mean(traj.summarize_epsidoe_info())
        logger.logkv_mean('total_steps', self.total)

        self.epoch_id += 1
        locals_.pop('self')

        for i in self.hooks:
            i.on_epoch(self, **locals_)
        
        self.mode = 'training'

        
def build_hooks(hook_cfg: dict) -> List[HookBase]:
    hooks = []
    if hook_cfg is not None:
        for k, v in hook_cfg.items():
            hooks.append(HOOKS[k](**v))
    return hooks


@as_hook
class save_model(HookBase):
    def __init__(self, n_epoch=10, model_name='model'):
        self.n_epoch = n_epoch
        self.model_name = model_name

    def init(self, trainer):
        self.modules = trainer.modules()

    def on_epoch(self, trainer, **locals_):
        #return super().on_epoch(trainer, **locals_)
        if trainer.epoch_id % self.n_epoch == 0:
            logger.torch_save(self.modules, self.model_name)

    
@as_hook
class log_info(HookBase):
    def __init__(self, n_epoch=10) -> None:
        super().__init__()
        self.n_epoch = n_epoch

    def on_epoch(self, trainer, **locals_):
        if trainer.epoch_id % self.n_epoch == 0:
            logger.dumpkvs()

    
    
@as_hook
class save_traj(HookBase):
    def __init__(self, n_epoch=10, traj_name='traj', save_gif_epochs=0, **kwargs) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.traj_name = traj_name
        self.kwargs = kwargs
        self.save_gif_epochs = save_gif_epochs

        self.imgs = []

    def on_epoch(self, trainer: RLAlgo, env, steps, **locals_):
        """

        traj is a Trajectory
        traj.traj[0] is a dict of transition. 
        traj.traj[i]['_attrs']  records dict of values for rendering for state; must be numpy or float


        render_traj calls envs._render_traj_rgb function to generate a dict of form

        {  
            'state': np.array, T, B, 2,
            'background': {
                'image': background image (that can be plt.imshow),
                'xlim': [0, 256] or None,
                'ylim': [0, 256] or None,
            },
            'actions': np.array, T, B, 2,
        }
        """
        if trainer.epoch_id % self.n_epoch == 0:
            assert trainer.mode == 'sample'
            from .traj import Trajectory
            traj: Trajectory = trainer.evaluate(env, steps)

            logger.logkvs_mean(
                {'eval_' + k: v 
                 for k, v in traj.summarize_epsidoe_info().items()}
            )

            old_obs = traj.get_tensor('obs')
            if trainer.obs_rms:
                old_obs = trainer.obs_rms.unormalize(old_obs)
                raise NotImplementedError
            traj.old_obs = old_obs

            data = env.render_traj(traj) 

            from tools.utils import plt_save_fig_array
            from solver.draw_utils import plot_colored_embedding, plot_point_values
            import matplotlib.pyplot as plt

            images = {}
            import numpy as np
            background = data.get('background', {})

            def clear(use_bg=True):
                plt.clf()
                if use_bg:
                    if 'image' in background:
                        plt.imshow(np.uint8(background['image']*255))
                    if 'xlim' in background:
                        plt.xlim(background['xlim'])
                    if 'ylim' in background:
                        plt.ylim(background['ylim'])

            def get(name):
                plt.title(name)
                plt.tight_layout()
                images[name] = plt_save_fig_array()[:, :, :3]

            # plot z.
            clear()
            z = traj.get_tensor('z', device='cpu')
            import torch
            if z.dtype == torch.int64:
                print('zbin', torch.bincount(z.long().flatten()))
                print('z.shape', z.shape, 'data state shape', data['state'].shape)
            plot_colored_embedding(z, data['state'], s=2)
            get('latent')

            if 'actions' in data:
                clear(use_bg=False)
                plot_colored_embedding(z, data['actions'])
                get('action')

                
            # traj _attrs, record 
            if '_attrs' in traj.traj[0]:
                for k in traj.traj[0]['_attrs']:
                    v = [i['_attrs'][k] for i in traj.traj]
                    v = np.array(v)
                    print(k, v.shape)

                    clear()
                    v = v.reshape(-1)
                    plot_point_values(v.reshape(-1), data['state'], s=2)
                    get(k)

            for k, v in images.items():
                logger.savefig(k + '.png', v)

            #logger.savefig(self.traj_name + '.png', img)
            if self.save_gif_epochs > 0:
                img = np.concatenate([v for k, v in images.items()], axis=1)
                self.imgs.append(img)
                if len(self.imgs) % self.save_gif_epochs == 0:
                    logger.animate(self.imgs, self.traj_name + '.gif')


@as_hook
class evaluate_pi(HookBase):
    def __init__(self, n_epoch=10, **kwargs) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.kwargs = kwargs

    def on_epoch(self, trainer: RLAlgo, env, steps, **locals_):
        if trainer.epoch_id % self.n_epoch == 0:
            assert trainer.mode == 'sample'
            traj = trainer.evaluate(env, steps)
            from tools.utils import logger
            logger.logkvs_mean(
                {'eval_' + k: v 
                 for k, v in traj.summarize_epsidoe_info().items()}
            )
                    


@as_hook
class monitor_action_std(HookBase):
    def __init__(self, n_epoch=1, std_decay=None) -> None:
        super().__init__()
        self.n_epoch = n_epoch
        self.std_decay = std_decay

    def init(self, trainer):
        self.head = trainer.pi.actor_optim.actor.head
        assert self.head.std_mode.startswith('fix')

        if self.std_decay is not None:
            from tools.utils import scheduler
            self.scheduler = scheduler.Scheduler.build(
                cfg=self.std_decay, init_value=self.head.std_scale)

    def on_epoch(self, trainer, **locals_):
        if trainer.epoch_id % self.n_epoch == 0:
            import torch
            print('action_std', (torch.exp(self.head.log_std) * self.head.std_scale).detach().cpu().numpy().reshape(-1))

            if self.std_decay is not None:
                self.head.std_scale = self.scheduler.step(epoch=trainer.total)
