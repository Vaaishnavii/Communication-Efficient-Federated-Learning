# 🧠 Communication-Efficient Federated Learning via Deep Reinforcement Learning-Guided Compression

This project presents a novel framework that integrates **Federated Learning (FL)** with **Deep Reinforcement Learning (DRL)** to significantly reduce communication overhead during distributed training across heterogeneous clients. The approach leverages an intelligent RL agent to dynamically decide when to synchronize, compress, or skip updates based on client resource availability and training progress. By incorporating techniques like **Top-K sparsification**, **8-bit quantization**, **asynchronous aggregation**, and **meta-gradient-based clustering**, the system achieves high accuracy on non-IID data while minimizing communication costs.

---

## 🎯 Motivation

In traditional Federated Learning setups, frequent model synchronization leads to massive communication overhead, especially in bandwidth-limited or resource-constrained environments. This project addresses that challenge by:
- Using **Deep Q-Learning (DQN)** to learn an optimal policy for each client that balances local computation and global communication.
- Applying **semantic-aware compression techniques** to reduce the size of updates sent over the network.
- Handling **stale updates** using **asynchronous aggregation**, which ensures model convergence despite communication irregularities.
- Grouping similar clients using **meta-gradient clustering**, leading to personalized and efficient learning.

---

## 🛠️ System Overview

### 🔁 Federated Learning Core
- CNN model trained locally on non-IID MNIST data
- Clients partitioned using Dirichlet distribution (α = 0.3)
- Periodic global aggregation of local models

### 🤖 Reinforcement Learning Agent
- **DQN** learns a communication control policy per client
- State includes: local accuracy, gradient divergence, staleness, round number
- Actions include: full sync, compressed sync, skip update, adjust compression rate

### 📉 Compression & Optimization
- **Top-K sparsification** retains only high-importance gradients
- **8-bit quantization** minimizes transmission size
- **Semantic pruning** disables updates for low-impact layers
- Clients asynchronously sync with the server based on learned policy

### 🧠 Client Clustering
- **Meta-gradient based similarity** metric
- Dynamic grouping using silhouette-optimized clustering
- Improves personalization and accelerates convergence

---

## 📊 Evaluation & Results

- Dataset: **MNIST** (non-IID, Dirichlet α = 0.3)
- Model: **2 Conv + 2 FC layers (CNN)**
- Baselines: FedAvg, FedProx, SCAFFOLD
- Metrics:
  - ✅ Final accuracy (up to 89% with DRL policy)
  - 📉 16× reduction in communication frequency
  - 📉 Reduced staleness-induced degradation via async updates
  - 📈 Faster convergence compared to fixed-frequency strategies
    
---

## 📌 Highlights

- 🧠 **Smart RL-driven communication** per client
- 🧹 **Compression-aware update selection** for gradient transfer
- 🔁 **Async update handling** for partial client participation
- 🧩 **Client clustering** enables partial personalization
- 📈 **Per-round visualizations** for accuracy, loss, reward, and sync metrics

---

