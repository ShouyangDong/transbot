import tvm
from tvm import meta_schedule as ms
import numpy as np
from tvm.meta_schedule.testing.space_generation import generate_design_space

import jax as jx
import jax.numpy as jnp
from jax import jit, vmap,lax

from functools import partial
# import operator
from tqdm import tqdm

from softmax import Softmax
from tvm.target import Target
import random

ActionSpace = [
    ms.schedule_rule.AutoBind(),
    ms.schedule_rule.AutoInline(
        into_producer=True,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=False,
        require_injective=False,
        require_ordered=False,
    ),
    ms.schedule_rule.CrossThreadReduction(
        thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]
    ),
    ms.schedule_rule.MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
    ),
    ms.schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=-1,  # disable parallelize
        max_vectorize_extent=-1,  # disable vectorize
        unroll_max_steps=[0, 16, 64, 512, 1024],
        unroll_explicit=True,
    ),
    ms.schedule_rule.RandomComputeLocation(),
    ms.schedule_rule.InlineConstantScalars(),
]
GFLOPS = 64 * 1280 * 2 / 1e9
A_Length = len(ActionSpace)




def objective(mod, target, name, inputs):
    """We design an objective function. If compile and runtime error happens,
    then the score is quiet large.
    """

    try:
        myfunc = tvm.build(mod, target=target, name=name)
    except:
        return 0.0

    try:
        myfunc(*inputs)
    except:
        return 0.0
    evaluator = myfunc.time_evaluator(
        myfunc.entry_name, tvm.device("cuda", 0), number=10
    )
    time_ms = evaluator(*inputs).mean * 1e3
    return GFLOPS / (time_ms / 1e3)



# 使用一个辅助函数选择对应的 Action
def get_action_from_space(action_id):
    return ActionSpace[action_id]  # 通过索引获取相应的 Action

# 使用 vmap 或 lax.map 实现映射
def dynamic_action_selection(cur_action_ids):
    return jx.vmap(get_action_from_space)(cur_action_ids)

class TvmGo:
    def __init__(self,mod,mod_name,tvm_tgt,inputs,action_len=A_Length,optimizer_len = 7, goal_reward=False, timeout=None):
        self.timeout = timeout
        self.mod = mod
        self.tvm_tgt = tvm_tgt
        self.action_len = action_len
        self.optimizer_len = optimizer_len
        self.inputs = inputs
        self.mod_name = mod_name
        self.best_reward = 0.01
        self.best_optimizer_ids = None

    def perform_action(self, actions):
        """Generates a design space for a given `action`. It calls `generate_design_space()`
        with specific parameters to apply the given scheduling rule (`action`) to the module.
        The function returns a new `ProgramState` object, which represents the new program
        state after applying the action."""
        # TODO(dongshouyang):change the spaces
        spaces = generate_design_space(
            kind="cuda",
            mod = self.mod,
            target = self.tvm_tgt,
            types = None,
            sch_rules = actions,
        )
        
        score =  objective(spaces[0].mod, self.tvm_tgt, self.mod_name, self.inputs)
        return spaces[0].mod,score

    # @partial(jit, static_argnums=(0,))
    @jit
    def step(self, action_id, env_state):
        # env_state 我需要得到如下信息。
        # -- optimize_path: 父节点所有位置。个数为访问深度
        # -- optimize_grid: 访问矩阵。矩阵是 action length * optimizer length
        # -- depth: 访问深度
        optimize_grid,trajectory, depth = env_state
        trajectory = trajectory.at[depth].set(action_id) 
        cur_action_ids  = lax.dynamic_slice(trajectory, (0,), (depth.val[0] + 1,)) #trajectory[:depth + 1]
        # cur_actions = dynamic_action_selection(cur_action_ids.val)
        cur_actions = [ActionSpace[_i] for _i in jx.device_get(cur_action_ids.val[0]).tolist()]

        # tvm_space,reward = self.perform_action(cur_actions)
        try:
            _tvm_space,_reward = self.perform_action(cur_actions)

            rewards = []
            for _i in range(20):
                _tvm_space,_reward = self.perform_action(cur_actions)
                rewards.append(_reward)
            
            reward = np.stack(rewards).mean()
                
        except:
            reward = 0.
        
        # reward = max(0.8,reward)
        
        # if depth.val < 2:
        #     reward = reward + random.uniform(-0.2, 0.2)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_optimizer_ids = cur_action_ids.val[0].tolist()

        print(f'action:',cur_action_ids.val[0].tolist(),'reward:',reward,'best_reward:',self.best_reward,'best_reward_ids:',self.best_optimizer_ids)
        optimize_grid.at[depth,action_id].set(True)
        # Treminated if we reach the goal
        terminal = depth > self.optimizer_len 

        next_env_state = optimize_grid,trajectory, depth + 1
        return next_env_state, self.get_observation(next_env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        optimize_grid = jnp.zeros([self.action_len,self.optimizer_len],dtype=bool)
        trajectory = jnp.zeros(self.optimizer_len,dtype=int)
        
        depth = 0
        env_state = optimize_grid,trajectory, depth
        return env_state 
    
    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        optimize_grid,trajectory, depth = env_state
    
        return optimize_grid
    
    def num_actions(self):
        return self.action_len
    
    
def ref_program(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
    

def build_env():

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(64, 1280)).astype("float32")
    buff_a = tvm.nd.array(a_np, dev)
    buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
    inputs = [buff_a, buff_c]
    tvm_tgt = Target("nvidia/nvidia-a100", host="llvm")
    name = "softmax"
    
    inputs = [buff_a, buff_c]
    action_len = len(ActionSpace)
    optimizer_len = 8
    tvm_env = TvmGo(Softmax,name,tvm_tgt,inputs,action_len=action_len,optimizer_len = optimizer_len)
    return  tvm_env

def _test():

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(64, 1280)).astype("float32")
    buff_a = tvm.nd.array(a_np, dev)
    buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
    inputs = [buff_a, buff_c]
    tvm_tgt = Target("nvidia/nvidia-a100", host="llvm")
    name = "softmax"
    
    inputs = [buff_a, buff_c]
    action_len = len(ActionSpace)
    optimizer_len = 8
    tvm_env = TvmGo(Softmax,name,tvm_tgt,inputs,action_len=action_len,optimizer_len = optimizer_len)
    optimize_path, optimize_grid,trajectory, depth = tvm_env.reset(None)
    optimize_path = [0,]
    env_state = optimize_path,optimize_grid,trajectory,depth

    # scores = np.zeros([7,7],dtype = np.float32)
    # for _i1 in range(7):
    #     for _j2  in tqdm(range(7)):
    #         _actions = [ActionSpace[_i1],ActionSpace[_j2]]
    #         mod,_scores = tvm_env.perform_action(_actions)
    #         scores[_i1,_j2] = _scores
    
    # print(scores)
    tvm_env.step(1,env_state)


            
if __name__ == "__main__":
    _test()