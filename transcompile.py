import os
import random

import numpy as np

from reward import reward_function
from transformation import get_LLM_answers


class SophisticatedQLearningAgent:
    def __init__(
        self,
        states,
        actions,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.995,
        max_episodes=10000,
        max_steps_per_episode=200,
    ):
        # Initialize the Q-learning agent with specified parameters.
        self.states = states  # Number of states in the environment
        self.actions = actions  # Number of possible actions
        self.learning_rate = learning_rate  # Rate at which the agent learns
        self.discount_factor = discount_factor  # Factor for discounting future rewards
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.min_exploration_rate = min_exploration_rate  # Minimum exploration rate
        self.exploration_decay_rate = (
            exploration_decay_rate  # Rate of decay for exploration
        )
        self.max_episodes = max_episodes  # Maximum number of training episodes
        self.max_steps_per_episode = max_steps_per_episode  # Maximum steps per episode
        self.q_table = np.zeros((states, actions))  # Initialize Q-table with zeros

    def choose_action(self, state):
        # Choose an action based on the current state and exploration-exploitation strategy.
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(
                0, self.actions - 1
            )  # Explore: choose a random action
        else:
            action = np.argmax(
                self.q_table[state, :]
            )  # Exploit: choose the best known action
        return action

    def learn(self, state, action, reward, next_state):
        # Update the Q-table based on the action taken and the resulting state.
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)

    def update_exploration_rate(self):
        # Decrease exploration rate over time.
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay_rate,
        )

    def has_converged(self, threshold=0.005):
        # Check if the Q-values have converged.
        return np.all(
            np.abs(self.q_table - np.max(self.q_table, axis=1, keepdims=True))
            < threshold
        )

    def train(self):
        # Train the agent over a series of episodes.
        for episode in range(self.max_episodes):
            state = random.randint(0, self.states - 1)
            for step in range(self.max_steps_per_episode):
                action = self.choose_action(state)
                next_state = random.randint(0, self.states - 1)
                reward = 1 if next_state == self.states - 1 else 0
                self.learn(state, action, reward, next_state)
                state = next_state
            self.update_exploration_rate()
            if self.has_converged():
                return True, episode
        return False, self.max_episodes

    def resize_q_table(self, new_states, new_actions):
        # Resize the Q-table if the number of states or actions increases.
        if new_states > self.states or new_actions > self.actions:
            new_q_table = np.zeros((new_states, new_actions))
            new_q_table[: self.states, : self.actions] = self.q_table
            self.q_table = new_q_table
            self.states = new_states
            self.actions = new_actions


# Initialize and refine the sophisticated agent
sophisticated_agent = SophisticatedQLearningAgent(states=10, actions=5)
max_refinements = 5
refinement_count = 0
converged = False

# Refine the agent until it converges or reaches the maximum number of refinements
while not converged and refinement_count < max_refinements:
    new_states = sophisticated_agent.states + 10  # Increasing the number of states
    new_actions = sophisticated_agent.actions  # Keeping the number of actions constant
    sophisticated_agent.resize_q_table(new_states, new_actions)  # Resize the Q-table
    sophisticated_agent.max_episodes += 5000  # Increase the number of episodes
    sophisticated_agent.max_steps_per_episode += 50  # Increase the steps per episode
    converged, episodes = sophisticated_agent.train()  # Train the agent
    refinement_count += 1

# Result of training after refinements
print(f"Converged: {converged}, Episodes: {episodes}, Refinements: {refinement_count}")
print(f"Q-Table:\n{sophisticated_agent.q_table}")
