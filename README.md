#  SHM_MDOF: Structural Health Monitoring using AI

Welcome to the **SHM_MDOF** project! 

##  What does this project do?
**SHM_MDOF** stands for **S**tructural **H**ealth **M**onitoring for **M**ulti-**D**egree **o**f **F**reedom systems. 

Imagine a complex structure like a building or a bridge. When exposed to forces (like wind or earthquakes), it vibrates. This project uses advanced AI called **Physics-Informed Neural Networks (PINNs)** to analyze those vibrations. By doing so, the AI can figure out the internal properties of the structure and detect if there is any damage. 

This project is divided into two parts:
* **Stage 1:** Training the base AI models, evaluating them, and seeing how well they learn the structure's properties.
* **Stage 2:** Advanced testing (like blind tests) and automatically generating beautiful plots and tables specifically designed for a thesis or research paper.

---

##  Table of Contents
1. [Prerequisites](#-prerequisites)
2. [Project Structure](#-project-structure)
3. [Setup & Installation](#-setup--installation)
4. [Running Stage 1 (Base Pipeline)](#-running-stage-1-base-pipeline)
5. [Running Stage 2 ( Advanced Analysis)](#-running-stage-2-thesis--advanced-analysis)
6. [Where to find your Results](#-where-to-find-your-results)

---

##  Prerequisites
To run this code on your machine, you need a few tools installed:
* A Windows computer running **WSL (Windows Subsystem for Linux)** with an Ubuntu terminal.
* **Anaconda** or **Miniconda** installed inside your Ubuntu terminal.
* An editor like **PyCharm** or VS Code to view the files (optional, but helpful).

---

##  Project Structure
Here is a quick map of where everything lives in this folder:
* `Data/` - Contains the raw data/vibration numbers the AI will learn from.
* `mdof_2dof_pinn/` - The main brain of the project. Contains all the Python scripts.
  * `stage2/` - Scripts specifically for the advanced Stage 2 analysis.
* `results/` - The folder where all your generated graphs, tables, and AI models are saved after running the code.
* `run commands` - Text files containing the terminal commands (which are also listed fully below).

---

##  Setup & Installation
Before running any code, you need to open your terminal and activate the project.

1. Open your **WSL (Ubuntu) terminal**.
2. Go to the project folder by typing:
   ```bash
   cd /mnt/c/Users/pavan/PycharmProjects/SHM_MDOF
   
---

##  Advanced Usage & Specific Commands
For advanced users, here are additional command variations supported by the scripts:

**Stage 1 Advanced Commands:**
* **Train a specific range forcefully:** `python -m mdof_2dof_pinn.run_all_runs --start 1 --end 5 --force`
* **Evaluate a single run:** `python -m mdof_2dof_pinn.evaluate --config mdof_2dof_pinn/config_mdof.yaml --run-id 1`
* **Evaluate a specific range:** `python -m mdof_2dof_pinn.evaluate --config mdof_2dof_pinn/config_mdof.yaml --start 1 --end 5`
* **Plot a single run:** `python -m mdof_2dof_pinn.plot_results --config mdof_2dof_pinn/config_mdof.yaml --run-id 1`
* **Plot a specific range:** `python -m mdof_2dof_pinn.plot_results --config mdof_2dof_pinn/config_mdof.yaml --start 1 --end 5`
* **Generate plots summary only:** `python -m mdof_2dof_pinn.plot_results --config mdof_2dof_pinn/config_mdof.yaml --summary-only`

**Stage 2 Advanced Commands:**
* **Train a single Stage 2 run:** `python -m mdof_2dof_pinn.stage2.trainer_stage2 --start 1 --end 1`
* **Train a different Stage 2 range:** `python -m mdof_2dof_pinn.stage2.trainer_stage2 --start 10 --end 20`
* **Evaluate without success threshold:** `python -m mdof_2dof_pinn.stage2.evaluate_stage2`
* **Evaluate from a specific CSV:** `python -m mdof_2dof_pinn.stage2.evaluate_stage2 --csv results/stage2_blind/stage2_summary.csv`
* **Summarize Stage 2 (Default):** `python -m mdof_2dof_pinn.stage2.summarize_stage2_results`
* **Plot Stage 2 (Default):** `python -m mdof_2dof_pinn.stage2.plot_stage2_results`

## **Where to find your Results**

Once you finish running the commands above, open the results/ folder in this project. Inside, you will find:

Images/Plots: Beautiful graphs showing the structural vibrations and AI predictions.

CSV Files: Spreadsheets containing the exact numbers and parameters the AI identified.

Tables: Formatted text ready .