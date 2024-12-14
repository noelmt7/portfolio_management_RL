import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the Q-learning agent class
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.action_size))
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        q_update = reward
        if not done:
            q_update += self.discount_factor * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.learning_rate * (q_update - self.q_table[state_key][action])

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

# Helper function for plotting with Seaborn color palette
def plot_rewards(rewards, title, xlabel, ylabel):
    sns.set_palette("viridis")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rewards, label="Total Reward")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

def plot_histogram(rewards, title, xlabel, ylabel):
    sns.set_palette("plasma")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(rewards, bins=20, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def plot_buy_sell_actions(stock_prices, buy_indices, sell_indices, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(stock_prices, label="Stock Price", color="blue")
    ax.scatter(buy_indices, stock_prices[buy_indices], marker="^", color="green", label="Buy", s=100)
    ax.scatter(sell_indices, stock_prices[sell_indices], marker="v", color="red", label="Sell", s=100)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# Streamlit UI
st.title("Reinforcement Learning for Portfolio Management")
st.sidebar.header("Upload Stock Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

# Load the model
@st.cache_resource
def load_model():
    with open('q_learning_agent.pkl', 'rb') as file:
        q_table = pickle.load(file)
    # Create agent instance and load the Q-table
    agent = QLearningAgent(state_size=2, action_size=3)  # [Buy, Hold, Sell]
    agent.q_table = q_table
    return agent

# Load the agent
agent = load_model()

# Display the uploaded file
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data")
    st.dataframe(data.head())

    # Preprocessing the dataset
    data['returns'] = data['Close'].pct_change().fillna(0)

    # Initialize environment
    class PortfolioManagementEnv:
        def __init__(self, data, initial_balance=1000):
            self.data = data
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.position = 0
            self.current_step = 0
            self.done = False

        def reset(self):
            self.balance = self.initial_balance
            self.position = 0
            self.current_step = 0
            self.done = False
            return self._get_state()

        def _get_state(self):
            return [self.data.iloc[self.current_step]['Close'], self.position]

        def step(self, action):
            price = self.data.iloc[self.current_step]['Close']
            reward = 0

            if action == 1:  # Buy
                self.position += 1
                self.balance -= price
            elif action == 2:  # Sell
                if self.position > 0:
                    self.position -= 1
                    self.balance += price

            self.current_step += 1
            if self.current_step >= len(self.data) - 1:
                self.done = True

            portfolio_value = self.balance + (self.position * price)
            reward = portfolio_value - self.initial_balance
            return self._get_state(), reward, self.done

    env = PortfolioManagementEnv(data)

    # Run a single iteration
    state = env.reset()
    total_reward = 0
    rewards = []
    buy_indices = []
    sell_indices = []

    while True:
        action = agent.get_action(state)  # Use the loaded agent to get the action
        state, reward, done = env.step(action)
        total_reward += reward
        rewards.append(total_reward)

        # Track buy and sell actions
        if action == 1:  # Buy action
            buy_indices.append(env.current_step)
        elif action == 2:  # Sell action
            sell_indices.append(env.current_step)

        if done:
            break

    st.write(f"**Test Total Reward:** {total_reward:.2f}")

    # Plot total reward over time
    plot_rewards(rewards, title="Portfolio Value Over Time", xlabel="Step", ylabel="Portfolio Value")

    # Plot histogram of rewards across all steps
    plot_histogram(rewards, title="Distribution of Rewards", xlabel="Portfolio Value", ylabel="Frequency")

    # Plot Buy/Sell actions with Stock Price
    plot_buy_sell_actions(data['Close'], buy_indices, sell_indices, title="Buy and Sell Actions", xlabel="Step", ylabel="Price")

else:
    st.info("Please upload a stock data CSV file.")
