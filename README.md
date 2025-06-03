# 🤖 Strategic Decision Advisor: Make, Buy, or Joint Venture (TCE-Based AI System)

This project implements an AI-powered decision-support system designed to help executives choose the optimal strategy for sourcing or developing a specific business capability. It evaluates options like **MAKE** (Internal Expansion/Build), **BUY** (Outsourcing/Market Purchase), or **JV** (Joint Venture/Strategic Alliance) by applying **Transaction Cost Economics (TCE)** principles (Coase, Williamson) — both of whom received the Nobel Prize in Economics and considering **scenario-based uncertainty**.

The system allows users to define their strategic business question via a Streamlit interface. It then leverages a pipeline of specialized Python agents, many of which utilize Large Language Models (LLMs) internally (defaulting to `gpt-4o-mini`), to perform analysis and generate a comprehensive recommendation report.

Demo link: https://www.loom.com/share/8ec7755ef7534d4bbce9d26d4493faa6
---

## 🎯 Purpose & Key Questions Answered

The Strategic Transaction Advisor is specifically designed to address complex **organizational boundary decisions**, often framed as "make-versus-buy-versus-ally" choices. It helps executives determine the most economically efficient and strategically sound way to acquire, develop, or access a specific business capability, asset, or service.

**Typical questions this system can help analyze:**

*   "Should our company **build** a new AI recommendation engine internally, **buy** an existing solution, or form a **joint venture** to co-develop it?"
*   "What is the most economically sound approach for our new Customer Relationship Management (CRM) system – internal development or a SaaS subscription?"
*   "How should we structure the development and operation of our upcoming digital transformation platform?"

---

## 🚀 Core Features

-   **User-Defined Business Task**: Users specify the strategic question via a Streamlit UI.
-   **Transaction Cost Economics Engine**: LLM-enhanced agents analyze asset specificity, transaction frequency, uncertainty, and partner trust.
-   **Scenario Modeling**: Evaluates strategies across 8 predefined future scenarios (combinations of market demand, market volatility, and partner reliability).
-   **Interactive Probability Assignment**: Streamlit UI allows users to assign subjective probabilities to each of the 8 scenarios.
-   **Multi-Agent Workflow**: A sequence of specialized Python agents process information:
    1.  `ContextExtractorAgent`: Extracts key TCE factors from the user's business task using an LLM.
    2.  `ScenarioGeneratorAgent`: Generates detailed, task-specific narratives and implications for the 8 scenarios using an LLM.
    3.  `ProbabilityCollectorAgent`: Programmatically validates and uses an LLM to qualitatively review scenario probabilities.
    4.  `TransactionLogicAgent`: Programmatically recommends a strategy (MAKE/BUY/JV) for *each* scenario and uses an LLM for enriched reasoning and risk assessment.
    5.  `AggregationAgent`: Programmatically calculates probability-weighted scores and uses an LLM to synthesize a final overall recommendation considering qualitative factors.
    6.  `ExplanationAgent`: Generates a comprehensive executive summary of the final recommendation using an LLM.
-   **LLM Integration**: Leverages LLMs (configurable, defaults to `gpt-4o-mini`) for qualitative analysis, content generation, and complex reasoning tasks within agents.
-   **Reporting**:
    -   Generates a detailed Markdown report (`report.md`) suitable for executives.
    -   Outputs a JSON summary (`output_summary.json`) of all data processed and generated.
    -   Logs detailed execution steps to `run.log`.
-   **Dockerized**: Can be easily set up and run using Docker and Docker Compose.

---

## 🏗️ System Architecture & Workflow

The system operates through a user interface provided by Streamlit, which then triggers a backend pipeline orchestrated by `main.py`.

1.  **User Interaction (`streamlit_app.py`):**
    *   User inputs their strategic **Business Task**.
    *   User assigns **Scenario Probabilities** to 8 archetypes.
    *   Streamlit saves the task to `user_business_task.txt` and probabilities to `scenario_weights.json`.
    *   Streamlit executes `main.py`.

2.  **Backend Orchestration (`main.py`):**
    *   Loads environment variables (API keys, model configs).
    *   Reads the user-defined business task and scenario probabilities.
    *   Initializes a `shared_context` dictionary.
    *   Sequentially executes the six specialized agents, each taking `shared_context` as input and returning an updated version.

3.  **Agent Pipeline (in `agents/` modules, executed by `main.py`):**
    *   Each agent performs its specific analysis or generation task, many using LLMs.
    *   Data like extracted TCE factors, detailed scenario narratives, per-scenario recommendations, aggregated scores, and the final recommendation are progressively added to `shared_context`.

4.  **Output Generation (`main.py`):**
    *   Saves the final `shared_context` to `output_summary.json`.
    *   Saves the LLM-generated `executive_summary` (from `ExplanationAgent`) to `report.md`.

5.  **Results Display (`streamlit_app.py`):**
    *   After `main.py` completes, Streamlit reads `output_summary.json` to display generated scenario details.
    *   Streamlit reads `report.md` and renders it as a formatted report in the UI.

*(A more detailed step-by-step agent workflow is provided in a subsequent section).*

---

## 📄 Files and Directory Structure

```
.
├── agents/                     # Python modules for each specialized agent
│   ├── __init__.py
│   ├── context_extractor.py
│   ├── scenario_generator.py
│   ├── probability_collector.py
│   ├── transaction_logic.py
│   ├── aggregation_agent.py
│   └── explanation_agent.py
├── outputs/                    # (Created by user for Docker) Stores generated reports and logs
├── .env                        # (User created) Environment variables (OpenAI API key, etc.) - GITIGNORED
├── .gitignore                  # Specifies intentionally untracked files for Git
├── .dockerignore               # Specifies files to exclude from Docker image
├── Dockerfile                  # Instructions to build the Docker image
├── docker-compose.yml          # Defines and runs multi-container Docker applications
├── main.py                     # Main script to orchestrate the agent workflow
├── streamlit_app.py            # Streamlit frontend application
├── requirements.txt            # Python dependencies
├── scenario_weights.json       # Stores user-defined scenario probabilities
├── user_business_task.txt      # (Temporary) Stores user-defined business task from Streamlit
├── crew_config.yaml            # (Design document) Defines agent roles & goals
└── README.md                   # This file
```

---
## 🤖 Detailed Agent Roles & Workflow

The `main.py` script orchestrates the following sequence:

1.  **`ContextExtractorAgent` (Business Analyst Role)**
    *   **Input:** User-defined `task` (business question).
    *   **Method:** Uses an LLM to analyze the `task` and extract key TCE factors.
    *   **Output:** Adds `asset_specificity`, `frequency`, `uncertainty`, `partner_trust` (each with value and reasoning) to `shared_context`.

2.  **`ScenarioGeneratorAgent` (Scenario Planner Role)**
    *   **Input:** `task` from `shared_context`.
    *   **Method:** Uses an LLM to generate detailed, task-specific content (narrative, characteristics, TCE implications, risks, success factors) for each of the 8 scenario archetypes (HD-S-R, etc.). Output expected as JSON.
    *   **Output:** Adds `scenario_descriptions` (a dictionary of these rich details for all 8 scenarios) to `shared_context`.

3.  **`ProbabilityCollectorAgent` (Risk Analyst Role)**
    *   **Input:** `scenario_probabilities` (user-defined), `scenario_descriptions` (for LLM context).
    *   **Method:** Performs programmatic checks (completeness, sum to 1.0). Then, uses an LLM to qualitatively review probabilities for coherence, providing feedback and potentially suggested adjustments in JSON format.
    *   **Output:** Updates `scenario_probabilities` in `shared_context`; adds `programmatic_probability_validation` and `llm_probability_assessment` reports.

4.  **`TransactionLogicAgent` (Strategy Evaluator Role)**
    *   **Input:** Extracted TCE factors, (validated) `scenario_probabilities`, `scenario_descriptions`.
    *   **Method (for each scenario):**
        1.  Programmatically applies a TCE scoring model to recommend MAKE, BUY, or JV with a numeric confidence.
        2.  Uses an LLM to provide enriched qualitative reasoning, confidence, and risk assessment for this recommendation within the specific scenario's context. Output expected as JSON.
    *   **Output:** Adds `scenario_results` to `shared_context` (containing programmatic and LLM analysis for each of the 8 scenarios).

5.  **`AggregationAgent` (Decision Synthesizer Role)**
    *   **Input:** `scenario_results`, (validated) `scenario_probabilities`.
    *   **Method:**
        1.  Programmatically calculates probability-weighted scores for MAKE, BUY, JV.
        2.  Uses an LLM to synthesize these scores with qualitative strategic factors (risk tolerance, control, resources, time horizon – guided by its prompt) to provide a final overall recommendation and justification in JSON.
    *   **Output:** Adds `strategy_scores_programmatic`, `llm_aggregation_synthesis` (LLM's full analysis), and updates `final_recommendation` in `shared_context`.

6.  **`ExplanationAgent` (Narrative Generator Role)**
    *   **Input:** The entire `shared_context`.
    *   **Method:** Uses an LLM, prompted with all accumulated data and a detailed structure, to write a comprehensive executive summary in Markdown.
    *   **Output:** Adds `executive_summary` (Markdown string) to `shared_context`.

---

## 📊 Scenario Framework Archetypes

The system analyzes strategies against 8 future scenario archetypes:

| Scenario | Demand | Market | Partners | General Description                                          |
|----------|--------|--------|----------|--------------------------------------------------------------|
| HD-S-R   | High   | Stable | Reliable | Strong growth, predictable conditions, trustworthy partners    |
| HD-S-O   | High   | Stable | Opportunistic | Strong growth but untrustworthy partners                     |
| HD-V-R   | High   | Volatile | Reliable | Strong growth with market turbulence, reliable partners        |
| HD-V-O   | High   | Volatile | Opportunistic | Strong growth, volatile market, opportunistic partners       |
| LD-S-R   | Low    | Stable | Reliable | Low demand, stable conditions, trustworthy partners           |
| LD-S-O   | Low    | Stable | Opportunistic | Low demand, stable market, opportunistic partners           |
| LD-V-R   | Low    | Volatile | Reliable | Low demand, volatile market, reliable partners              |
| LD-V-O   | Low    | Volatile | Opportunistic | Low demand, volatile market, opportunistic partners         |
*Note: The `ScenarioGeneratorAgent` develops rich, task-specific narratives for each of these archetypes.*

---

## 🚀 Setup and Installation

### Prerequisites

-   Python 3.10+ (Python 3.11+ recommended for latest features)
-   Docker & Docker Compose (Strongly Recommended for ease of use)
-   An active OpenAI API Key with sufficient credits/quota.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Configure Environment Variables

Create a `.env` file in the project root directory (e.g., alongside `main.py`):

```env
OPENAI_API_KEY="your_actual_openai_api_key_here"

# Optional: Set specific LLM models for agents. If not set, agents use their coded defaults (e.g., "gpt-4o-mini").
# CONTEXT_EXTRACTOR_MODEL="gpt-4o-mini"
# SCENARIO_GENERATOR_MODEL="gpt-4o-mini" # Or a more powerful model if needed for complex generation
# PROBABILITY_COLLECTOR_MODEL="gpt-4o-mini"
# TRANSACTION_LOGIC_MODEL="gpt-4o-mini"
# AGGREGATION_AGENT_MODEL="gpt-4o-mini"
# EXPLANATION_AGENT_MODEL="gpt-4o-mini" # Or a more powerful model for detailed report writing

# For Docker: Specifies where main.py should write its output files inside the container.
# This path will be mapped to a host directory by docker-compose.yml.
OUTPUT_DIR_PATH="/app/outputs" 
```
**Important:** Ensure `.env` is listed in your `.gitignore` file to prevent committing secrets.

### 3. (Optional) Local Python Environment Setup

If you choose not to use Docker for local development/running:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*(Ensure `PYTHONUTF8=1` is set in your system environment on Windows if you encounter Unicode errors when running locally).*

---

## 🏃 Running the System

### Option 1: Using Docker (Recommended)

1.  Ensure Docker Desktop (or Docker Engine + Docker Compose) is installed and running.
2.  **Create an `outputs` directory** in your project root. This directory will be mapped into the Docker container for persistent storage of generated reports and logs.
    ```bash
    mkdir outputs
    ```
3.  From the project root directory (containing `docker-compose.yml`), build and run the service:
    ```bash
    docker-compose up --build
    ```
    The `--build` flag ensures the image is rebuilt if you've made changes to the `Dockerfile` or application code.
4.  **Access the application:** Open your web browser and navigate to `http://localhost:8501`.

    When you run the pipeline via the Streamlit UI:
    *   `user_business_task.txt` (temporary) and `scenario_weights.json` are created/updated in the project root (inside the container, but `scenario_weights.json` might be volume-mapped).
    *   `run.log`, `output_summary.json`, and `report.md` will be saved to the `outputs/` directory on your host machine (due to the volume mapping defined in `docker-compose.yml`).

### Option 2: Running Locally (Without Docker)

1.  Ensure you have completed **Step 2 (Environment Variables)** and **Step 3 (Local Python Environment Setup)** from the Setup section.
2.  If you are on Windows and encounter encoding errors, try setting the `PYTHONUTF8` environment variable:
    *   In PowerShell: `$env:PYTHONUTF8 = "1"`
    *   In CMD: `set PYTHONUTF8=1`
    (Or add `PYTHONUTF8=1` to your system environment variables).
3.  Start the Streamlit application from your project root:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  Streamlit will provide a URL (usually `http://localhost:8501`) to access in your browser.

    When you run the pipeline:
    *   `user_business_task.txt` and `scenario_weights.json` are created/updated in the project root.
    *   `run.log`, `output_summary.json`, and `report.md` are created/updated in the project root (or wherever `OUTPUT_DIR_PATH` defaults to if not set, which is `.`).

---

## 🔧 Configuration & Customization

-   **Business Question:** Defined by the user in the Streamlit UI (Step 1).
-   **Scenario Probabilities:** Adjusted via sliders in the Streamlit UI (Step 2); saved to `scenario_weights.json`.
-   **LLM Models:** Can be customized per agent or globally using environment variables in the `.env` file (e.g., `CONTEXT_EXTRACTOR_MODEL="gpt-3.5-turbo"`). Agents default to `gpt-4o-mini` if specific variables aren't set.
-   **Agent Logic & Prompts:** The core reasoning, programmatic rules (e.g., in `TransactionLogicAgent`), and LLM prompts for each agent are defined in their respective Python files in the `agents/` directory. These can be modified to alter behavior, depth of analysis, or output style.
-   **Output Paths (for Docker):** The `OUTPUT_DIR_PATH` environment variable in `.env` tells `main.py` where to write files inside the container. `docker-compose.yml` maps a host directory (e.g., `./outputs`) to this container path.

---

## 📈 Understanding the Output

The system produces three main artifacts after a successful pipeline run:

1.  **`report.md` (Displayed in Streamlit):**
    *   This is the primary human-readable output – a comprehensive executive summary generated by the `ExplanationAgent`.
    *   It includes:
        *   The final strategic recommendation (MAKE, BUY, or JV).
        *   Detailed strategic rationale based on TCE principles and context.
        *   A summary of how the recommendation performs under various scenarios.
        *   Implementation considerations, key success factors, and risk mitigation strategies.
        *   Discussion of alternative strategies.

2.  **`output_summary.json`:**
    *   A complete JSON dump of the final `shared_context` dictionary.
    *   Contains all data inputs, intermediate results from each agent (e.g., extracted TCE factors, detailed `scenario_descriptions`, per-scenario `scenario_results`, LLM assessments, programmatic scores), and the final `executive_summary` string.
    *   Extremely useful for debugging, detailed review, or feeding into other analytical tools.

3.  **`run.log`:**
    *   A detailed log file capturing the execution flow of `main.py` and informational/debug/error messages from all agents. Essential for troubleshooting and understanding the step-by-step processing.

---

## 🧠 Theoretical Foundation (Brief)

The system's core analytical approach is rooted in **Transaction Cost Economics (TCE)**, primarily drawing from the work of Ronald Coase and Oliver Williamson — both of whom received the Nobel Prize in Economics. Key TCE concepts used include:

-   **Asset Specificity**: Degree of specialization of assets required for the capability.
-   **Transaction Frequency**: How often transactions related to the capability occur.
-   **Uncertainty**: Level of market, technological, or environmental unpredictability.
-   **Partner Trust / Opportunism Risk**: Likelihood of opportunistic behavior from external partners.

The system aims to recommend a governance structure (MAKE, BUY, or JV) that minimizes the sum of production and transaction costs under these conditions, considering various future scenarios.

---

## 📝 Dependencies

Key Python dependencies include:
- `crewai` (and its dependencies like `openai`, `litellm`)
- `streamlit`
- `python-dotenv`
- `pyyaml` (if `crew_config.yaml` were actively used for loading)
- `markdown2` (if used for any MD-to-HTML conversion beyond Streamlit's built-in)

Refer to `requirements.txt` for the complete list.

---

## 💡 Future Enhancements (Potential)

-   More sophisticated UI for scenario probability elicitation (e.g., pairwise comparisons).
-   Allowing users to define custom scenario dimensions or modify existing ones.
-   Integration with a database for storing/comparing multiple analyses.
-   Advanced sensitivity analysis on probabilities and TCE factor assessments.
-   Directly using CrewAI's `Crew` and `Process` orchestration in `main.py` for more complex agent interactions (e.g., hierarchical or parallel tasks).
-   More robust error handling and user feedback loops within the Streamlit UI for backend failures.

---

*This system aims to provide a structured, AI-augmented approach to complex strategic decision-making.*
```
