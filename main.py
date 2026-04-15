import numpy as np
import matplotlib.pyplot as plt
import os

ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def step(state, action_idx):
    ay, ax = ACTIONS[action_idx]
    y, x = state
    ny, nx = y + ay, x + ax
    
    if ny < 0 or ny >= ROWS or nx < 0 or nx >= COLS:
        ny, nx = y, x
        
    if ny == 3 and 1 <= nx <= 10:
        return START, -100, False
        
    if (ny, nx) == GOAL:
        return (ny, nx), -1, True
        
    return (ny, nx), -1, False

def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        values = q_table[state[0], state[1], :]
        max_val = np.max(values)
        return np.random.choice([a for a, v in enumerate(values) if v == max_val])

def q_learning(episodes=500, alpha=0.9, gamma=1.0, epsilon=0.1):
    q_table = np.zeros((ROWS, COLS, 4))
    rewards = np.zeros(episodes)
    
    for ep in range(episodes):
        state = START
        ep_reward = 0
        done = False
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = step(state, action)
            ep_reward += reward
            
            best_next_action = np.argmax(q_table[next_state[0], next_state[1], :])
            td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
            td_error = td_target - q_table[state[0], state[1], action]
            q_table[state[0], state[1], action] += alpha * td_error
            
            state = next_state
        rewards[ep] = ep_reward
        
    return q_table, rewards

def sarsa(episodes=500, alpha=0.9, gamma=1.0, epsilon=0.1):
    q_table = np.zeros((ROWS, COLS, 4))
    rewards = np.zeros(episodes)
    
    for ep in range(episodes):
        state = START
        ep_reward = 0
        done = False
        action = choose_action(state, q_table, epsilon)
        
        while not done:
            next_state, reward, done = step(state, action)
            ep_reward += reward
            next_action = choose_action(next_state, q_table, epsilon)
            
            td_target = reward + gamma * q_table[next_state[0], next_state[1], next_action]
            td_error = td_target - q_table[state[0], state[1], action]
            q_table[state[0], state[1], action] += alpha * td_error
            
            state = next_state
            action = next_action
            
        rewards[ep] = ep_reward
        
    return q_table, rewards

runs = 50
episodes = 500

q_rewards_avg = np.zeros(episodes)
sarsa_rewards_avg = np.zeros(episodes)

q_q_table = np.zeros((ROWS, COLS, 4))
sarsa_q_table = np.zeros((ROWS, COLS, 4))

print("Running Q-learning...")
for _ in range(runs):
    q_tbl, rewards = q_learning(episodes=episodes, alpha=0.9, gamma=1.0, epsilon=0.1)
    q_rewards_avg += rewards
    q_q_table += q_tbl
q_rewards_avg /= runs
q_q_table /= runs

print("Running Sarsa...")
for _ in range(runs):
    s_tbl, rewards = sarsa(episodes=episodes, alpha=0.9, gamma=1.0, epsilon=0.1)
    sarsa_rewards_avg += rewards
    sarsa_q_table += s_tbl
sarsa_rewards_avg /= runs
sarsa_q_table /= runs

script_dir = os.path.dirname(os.path.abspath(__file__))
artifact_dir = r"C:\Users\User\.gemini\antigravity\brain\87f1ae9f-6583-4094-8122-392ea2a645a6"

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

plt.figure(figsize=(10, 6))
plt.plot(sarsa_rewards_avg, label='Sarsa (raw)', color='tab:blue', alpha=0.3)
plt.plot(smooth(sarsa_rewards_avg, 5), label='Sarsa (smoothed)', color='tab:blue', linewidth=2)
plt.plot(q_rewards_avg, label='Q-learning (raw)', color='tab:red', alpha=0.3)
plt.plot(smooth(q_rewards_avg, 5), label='Q-learning (smoothed)', color='tab:red', linewidth=2)
plt.ylim([-100, 0])
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.title('Sarsa vs Q-learning on Cliff Walking\nEpsilon=0.1, Alpha=0.9, Gamma=1.0 (Averaged over 50 runs)')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(script_dir, 'cliff_walking_rewards.png'))
plt.savefig(os.path.join(artifact_dir, 'cliff_walking_rewards.png'))
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

for ax, q_table, title in zip(axes, [q_q_table, sarsa_q_table], ['Q-learning Policy', 'Sarsa Policy']):
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks(np.arange(0, COLS+1, 1))
    ax.set_yticks(np.arange(0, ROWS+1, 1))
    ax.grid(color='k', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14)

    for i in range(ROWS):
        for j in range(COLS):
            if i == 3 and j == 0:
                ax.text(j+0.5, ROWS-i-0.5, 'Start', ha='center', va='center', fontsize=12, fontweight='bold')
            elif i == 3 and 1 <= j <= 10:
                ax.add_patch(plt.Rectangle((j, ROWS-i-1), 1, 1, facecolor='lightblue'))
                ax.text(j+0.5, ROWS-i-0.5, 'Cliff', ha='center', va='center', fontsize=12)
            elif i == 3 and j == 11:
                ax.text(j+0.5, ROWS-i-0.5, 'Goal', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                best_action = np.argmax(q_table[i, j, :])
                ay, ax_move = ACTIONS[best_action]
                dy = -ay * 0.35
                dx = ax_move * 0.35
                ax.annotate('', xy=(j+0.5+dx, ROWS-i-0.5+dy), xytext=(j+0.5, ROWS-i-0.5),
                            arrowprops=dict(arrowstyle='->', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'cliff_walking_policies.png'))
plt.savefig(os.path.join(artifact_dir, 'cliff_walking_policies.png'))
plt.close()

print("Experiment completed. Outputs saved.")
