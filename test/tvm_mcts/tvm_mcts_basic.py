# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A demo of Graphviz visualization of a search tree."""

<<<<<<< HEAD
from typing import Optional, Sequence
=======
>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455

from absl import app
from absl import flags
import chex
import jax
import jax.numpy as jnp
import mctx
from tvm_environments import build_env
<<<<<<< HEAD
import pygraphviz
from jax import jit, vmap, grad
from utils import convert_tree_to_graph

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
=======
from jax import vmap
from utils import convert_tree_to_graph
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_simulations", 512, "Number of simulations.")
<<<<<<< HEAD
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("max_depth", 5, "The maximum search depth.")
flags.DEFINE_string("output_file", "./tvm_search_tree.png",
                    "The output file for the visualization.")

jax.config.update('jax_disable_jit', True) 
jax.config.update('jax_enable_x64', True)
=======
flags.DEFINE_integer(
    "max_num_considered_actions",
    16,
    "The maximum number of actions expanded at the root.",
)
flags.DEFINE_integer("max_depth", 5, "The maximum search depth.")
flags.DEFINE_string(
    "output_file", "./tvm_search_tree.png", "The output file for the visualization."
)

jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)
>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455
jax.disable_jit()
BS = 1


# this assumes the agent has access to the exact environment dynamics
def get_recurrent_fn(env):
<<<<<<< HEAD
    batch_step = vmap(env.step, in_axes=(0,0))
=======
    batch_step = vmap(env.step, in_axes=(0, 0))

>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455
    def recurrent_fn(params, key, actions, env_states):
        key, subkey = jax.random.split(key)
        env_states, obs, rewards, terminals, _ = batch_step(actions, env_states)

        discount = jnp.ones_like(rewards)
<<<<<<< HEAD
        prior_logits = jax.random.uniform(subkey, shape=[BS,env.action_len])

        recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=rewards,
        discount=discount,
        prior_logits=prior_logits,
        value=jnp.ones([BS], dtype=jnp.float32))
        return recurrent_fn_output, env_states
=======
        prior_logits = jax.random.uniform(subkey, shape=[BS, env.action_len])

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discount,
            prior_logits=prior_logits,
            value=jnp.ones([BS], dtype=jnp.float32),
        )
        return recurrent_fn_output, env_states

>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455
    return recurrent_fn


def _run_demo(rng_key: chex.PRNGKey):
<<<<<<< HEAD
  """Runs a search algorithm on a toy environment."""
  env  = build_env()
  batch_reset = vmap(env.reset)
  
  key, subkey = jax.random.split(rng_key)
  subkeys = jax.random.split(subkey, num=BS)
  
  states_init = batch_reset(subkeys)
  num_actions = env.action_len
  batch_size = BS
  recurrent_fn = get_recurrent_fn(env)

  # Using optimistic initial values to encourage exploration.
  values = jnp.full([env.optimizer_len], 0.1)
  # The prior policies for each state.
  root_state = 0
  
  key, logits_rng = jax.random.split(key)
  rng_key, logits_rng, q_rng, search_rng = jax.random.split(key, 4)
  prior_logits = jax.random.normal(
      logits_rng, shape=[batch_size, num_actions]) *0.01 + 0.5

  qvalues = jax.random.uniform(q_rng, shape=prior_logits.shape)
  raw_value = jnp.sum(jax.nn.softmax(prior_logits) * qvalues, axis=-1)


  root = mctx.RootFnOutput(
      prior_logits = prior_logits,#jnp.full([batch_size, num_actions],
      value=jnp.ones([batch_size], dtype=jnp.float32),
      # The embedding will hold the state index.
      embedding=states_init,
  )

  # Running the search.
  policy_output = mctx.gumbel_muzero_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=FLAGS.num_simulations,
      max_depth=env.optimizer_len,
      max_num_considered_actions=FLAGS.max_num_considered_actions,
  )
  return policy_output




def main(_):
  rng_key = jax.random.PRNGKey(FLAGS.seed)
#   jitted_run_demo =  jax.jit(_run_demo)
  print("Starting search.")
  policy_output = _run_demo(rng_key)
  batch_index = 0
  selected_action = policy_output.action[batch_index]
  q_value = policy_output.search_tree.summary().qvalues[
      batch_index, selected_action]
  print("Selected action:", selected_action)
  # To estimate the value of the root state, use the Q-value of the selected
  # action. The Q-value is not affected by the exploration at the root node.
  print("Selected action Q-value:", q_value)
  graph = convert_tree_to_graph(policy_output.search_tree)
  print("Saving tree diagram to:", FLAGS.output_file)
  graph.draw(FLAGS.output_file, prog="dot")


if __name__ == "__main__":
  app.run(main)
=======
    """Runs a search algorithm on a toy environment."""
    env = build_env()
    batch_reset = vmap(env.reset)

    key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=BS)

    states_init = batch_reset(subkeys)
    num_actions = env.action_len
    batch_size = BS
    recurrent_fn = get_recurrent_fn(env)

    # Using optimistic initial values to encourage exploration.
    values = jnp.full([env.optimizer_len], 0.1)
    # The prior policies for each state.
    root_state = 0

    key, logits_rng = jax.random.split(key)
    rng_key, logits_rng, q_rng, search_rng = jax.random.split(key, 4)
    prior_logits = (
        jax.random.normal(logits_rng, shape=[batch_size, num_actions]) * 0.01 + 0.5
    )

    qvalues = jax.random.uniform(q_rng, shape=prior_logits.shape)
    raw_value = jnp.sum(jax.nn.softmax(prior_logits) * qvalues, axis=-1)

    root = mctx.RootFnOutput(
        prior_logits=prior_logits,  # jnp.full([batch_size, num_actions],
        value=jnp.ones([batch_size], dtype=jnp.float32),
        # The embedding will hold the state index.
        embedding=states_init,
    )

    # Running the search.
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=FLAGS.num_simulations,
        max_depth=env.optimizer_len,
        max_num_considered_actions=FLAGS.max_num_considered_actions,
    )
    return policy_output


def main(_):
    rng_key = jax.random.PRNGKey(FLAGS.seed)
    #   jitted_run_demo =  jax.jit(_run_demo)
    print("Starting search.")
    policy_output = _run_demo(rng_key)
    batch_index = 0
    selected_action = policy_output.action[batch_index]
    q_value = policy_output.search_tree.summary().qvalues[batch_index, selected_action]
    print("Selected action:", selected_action)
    # To estimate the value of the root state, use the Q-value of the selected
    # action. The Q-value is not affected by the exploration at the root node.
    print("Selected action Q-value:", q_value)
    graph = convert_tree_to_graph(policy_output.search_tree)
    print("Saving tree diagram to:", FLAGS.output_file)
    graph.draw(FLAGS.output_file, prog="dot")


if __name__ == "__main__":
    app.run(main)
>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455
