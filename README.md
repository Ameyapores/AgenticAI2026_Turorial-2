# Self-Improving Agents: Memory vs RL

This repository contains the self-improving agent tutorial notebook and its Python export.

## Setup on macOS

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/Ameyapores/AgenticAI2026_Turorial-2.git
   cd AgenticAI2026_Turorial-2
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

3. Install the dependencies:

   ```bash
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```

4. Configure the required LLM API. Copy the example file, then edit `.env` with the provider, model, and API key supplied by the course staff:

   ```bash
   cp .env.example .env
   open -e .env
   ```

   Enable exactly one provider configuration in `.env`. Never commit `.env` or put an API key in a notebook cell.

5. Start JupyterLab:

   ```bash
   .venv/bin/python -m jupyter lab
   ```

6. Open `Self_Improving_Agents_Memory_vs_RL_v3.ipynb`. Select the Python kernel from `.venv`, then run the cells from top to bottom. The notebook makes live model requests, so valid API settings and an internet connection are required.

## Python export

The notebook is also exported as `Self_Improving_Agents_Memory_vs_RL_v3.py`. The shared API client is in `llm_client.py`.

## Current limitation

The self-improving notebook imports `simulator.py`, but that module was not included in the supplied ZIP or the current repository. The environment setup above is complete; the notebook will need that module added before all cells can run successfully.