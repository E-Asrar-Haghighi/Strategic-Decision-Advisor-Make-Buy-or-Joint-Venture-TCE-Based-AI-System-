import streamlit as st
import json
import subprocess
import os
import sys # For sys.executable

st.set_page_config(layout="wide")

st.title("🤖 Strategic Decision Advisor: Make, Buy, or Joint Venture (TCE-Based AI System)")

# --- Configuration ---
SCENARIO_WEIGHTS_FILE = "scenario_weights.json"
OUTPUT_DIR = os.getenv("OUTPUT_DIR_PATH", ".")
OUTPUT_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "output_summary.json")
REPORT_MD_FILE = os.path.join(OUTPUT_DIR, "report.md")
MAIN_SCRIPT_PATH = "main.py"

# --- What Kind of Questions Does This System Answer? ---
st.markdown("---")
st.subheader("🎯 Purpose of this Advisor")

st.info(
    """
    This system helps you make strategic decisions about **how to best source or develop
    a specific business capability or asset**. It analyzes your situation using
    **Transaction Cost Economics (TCE)** and recommends whether to:

    - **MAKE (Build/Internal Expansion):** Develop the capability in-house.
    - **BUY (Outsource/Market Purchase):** Acquire the capability or service externally.
    - **JV (Joint Venture/Ally):** Partner with another entity to co-develop or access the capability.

    **Examples of questions this system can help with:**
    - "Should our company build a new AI recommendation engine internally, buy a solution, or form a joint venture?"
    - "What is the most economically sound approach for our new customer relationship management (CRM) system?"
    - "How should we structure the development of our upcoming digital platform?"

    **Pro Tip:** For better results, include details like:
    - Asset Specificity
    - Transaction Frequency
    - Uncertainty
    - Partner Trust

    **Example Business Question & Context**
    - **Business Question:** Should our company develop the AI customer insights platform internally, outsource development to a third-party vendor, or form a strategic partnership/joint venture with a specialized AI firm?
    - **Asset Specificity:** The platform requires customization for our proprietary datasets and internal business logic.
    - **Transaction Frequency:** Frequent updates and model retraining are necessary.
    - **Uncertainty:** High technical and market uncertainty exists (e.g., evolving AI and regulations).
    - **Partner Trust:** We trust one partner strongly but have little experience with others.
    """
)

st.markdown("---")
# --- End Purpose Explanation ---


# --- Step 1: Define Your Strategic Business Question ---
st.header("Step 1: Define Your Strategic Business Question")
BUSINESS_TASK_TEMP_FILE = "user_business_task.txt"  # Temp file to pass task
if 'business_task' not in st.session_state:
    st.session_state.business_task = "Should we outsource or build an AI recommendation engine?" 
business_task_description = st.text_area(
    "Describe the capability or decision you are analyzing:",
    value=st.session_state.business_task,
    height=100,
    help="Example: 'Should we develop a new CRM internally, buy a SaaS solution, or partner to create one?' The system will analyze whether to MAKE, BUY, or JV for this."
)
st.session_state.business_task = business_task_description # Keep session state updated

if not business_task_description.strip():
    st.warning("Please enter a business question to analyze.")
    st.stop() # Stop further rendering if no task is provided

st.markdown("---")
# --- End Step 1 ---


# --- Scenario Probability Assignment ---
st.header("Scenario Probability Assignment")
st.markdown(
    """
    Based on your understanding of the potential future, adjust the sliders below to assign a probability (0-100)
    to each of the 8 scenario archetypes relevant to your strategic question.
    The total of all probabilities **must sum to 100%**.

    The agent pipeline will use these probabilities to weigh the outcomes of different strategies (MAKE, BUY, JV)
    under each scenario. A `ScenarioGeneratorAgent` will also develop detailed, task-specific narratives
    for these archetypes based on the business question being analyzed.

    **Scenario Archetypes Codes & General Meanings:**
    - **HD/LD**: High Demand / Low Demand (for the capability/project in question)
    - **S/V**: Stable Market / Volatile Market (relevant to the capability/project)
    - **R/O**: Reliable Partner (High Trust) / Opportunistic Partner (Low Trust) (if external partners are considered)
    """
)

# Define scenario archetypes
scenarios = [
    "HD-S-R", "HD-S-O", "HD-V-R", "HD-V-O",
    "LD-S-R", "LD-S-O", "LD-V-R", "LD-V-O"
]

# Function to load initial weights or set defaults
def load_initial_weights():
    even_dist_val = 100 // len(scenarios)
    remainder = 100 % len(scenarios)
    default_initial_weights = {}
    for i, s_key in enumerate(scenarios):
        default_initial_weights[s_key] = even_dist_val + (1 if i < remainder else 0)

    if os.path.exists(SCENARIO_WEIGHTS_FILE):
        try:
            with open(SCENARIO_WEIGHTS_FILE, "r") as f:
                loaded_weights = json.load(f)
            if isinstance(loaded_weights, dict) and \
               all(s_key in loaded_weights and isinstance(loaded_weights[s_key], (int, float)) for s_key in scenarios) and \
               sum(int(v) for v in loaded_weights.values()) == 100:
                for s_key in scenarios:
                    loaded_weights[s_key] = max(0, min(100, int(loaded_weights[s_key])))
                return loaded_weights
            else:
                st.warning(f"'{SCENARIO_WEIGHTS_FILE}' found but content is invalid or doesn't sum to 100. Using default even distribution.")
                return default_initial_weights
        except (json.JSONDecodeError, Exception) as e:
            st.warning(f"Error reading '{SCENARIO_WEIGHTS_FILE}': {e}. Using default even distribution.")
            return default_initial_weights
    return default_initial_weights

if 'scenario_input_weights' not in st.session_state:
    st.session_state.scenario_input_weights = load_initial_weights()

current_weights = {}
total_probability = 0

cols = st.columns(4)
for i, s_name in enumerate(scenarios):
    with cols[i % 4]:
        initial_value = st.session_state.scenario_input_weights.get(s_name, 100 // len(scenarios))
        current_weights[s_name] = st.slider(
            f"{s_name}",
            min_value=0,
            max_value=100,
            value=initial_value,
            step=1,
            key=f"slider_{s_name}"
        )
    total_probability += current_weights[s_name]

st.metric(label="Total Assigned Probability", value=f"{total_probability}%", delta=f"{total_probability - 100}% from target")

if total_probability != 100:
    st.error("Total probabilities must sum to 100%. Please adjust the sliders.")
    run_button_disabled = True
else:
    st.success("Total probability is 100%. Ready to save and run.")
    run_button_disabled = False
# --- End Scenario Probability Assignment ---


# --- Run Agent Pipeline Button ---
can_run_pipeline = bool(business_task_description.strip()) and (total_probability == 100)
run_button_label = "Save Probabilities and Run Analysis" if can_run_pipeline else "Complete Task & Probabilities to Run"

if st.button(run_button_label, disabled=not can_run_pipeline, type="primary"):
    # 1. Save the business task to a temporary file
    try:
        with open(BUSINESS_TASK_TEMP_FILE, "w", encoding="utf-8") as f:
            f.write(st.session_state.business_task)
        st.success(f"Business task saved for pipeline.")
    except Exception as e:
        st.error(f"Error saving business task: {e}")
        st.stop()

    # 2. Save weights (as before)
    with open(SCENARIO_WEIGHTS_FILE, "w") as f:
        json.dump(current_weights, f, indent=2)
    st.session_state.scenario_input_weights = current_weights.copy()
    st.success(f"Probabilities saved to '{SCENARIO_WEIGHTS_FILE}'.")

    # 3. Run the pipeline (as before)
    st.info("The agent pipeline can take some time to run. Please be patient.")
    with st.spinner("🚀 Running agent pipeline... This may take a minute or two..."):
        try:
            command = [sys.executable, MAIN_SCRIPT_PATH] # main.py will now read the temp file
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            st.success("✅ Agent pipeline run complete!")

            if process.stdout:
                with st.expander("Agent Pipeline Standard Output (from main.py)", expanded=False):
                    st.text_area("Log", process.stdout, height=100, key="agent_stdout_log")

            st.markdown("---")
            st.subheader("📊 Final Strategic Recommendation Report")
            if os.path.exists(REPORT_MD_FILE):
                with open(REPORT_MD_FILE, "r", encoding="utf-8") as r:
                    st.markdown(r.read())
            else:
                st.error(f"'{REPORT_MD_FILE}' not found. The agent pipeline might have encountered an issue generating the final report.")

        except subprocess.CalledProcessError as e:
            st.error("❌ Agent pipeline run FAILED.")
            st.subheader("Error Details (from `main.py` execution):")
            error_message = "Error output from agent pipeline:\n"
            if e.stdout: error_message += "--- STDOUT ---\n" + e.stdout + "\n"
            if e.stderr: error_message += "--- STDERR ---\n" + e.stderr + "\n"
            if not e.stdout and not e.stderr: error_message = f"The agent pipeline script ({MAIN_SCRIPT_PATH}) failed with exit code {e.returncode} but produced no stdout or stderr."
            st.text_area("Error Log", error_message, height=250, key="agent_pipeline_error_log")
            st.info(f"Check 'run.log' (expected in '{os.path.join(OUTPUT_DIR, 'run.log')}') for more detailed internal logs.")
        except FileNotFoundError as e:
            st.error(f"🚨 Execution Error: {e}. Is '{MAIN_SCRIPT_PATH}' in the correct location?")
        except Exception as e:
            st.error(f"😱 An unexpected error occurred: {e}")
            st.info(f"Check 'run.log' (expected in '{os.path.join(OUTPUT_DIR, 'run.log')}') for logs.")
# --- End Run Agent Pipeline Button ---

st.markdown("---")
st.markdown("### Notes:")
st.markdown(f"- Probabilities are saved to `{SCENARIO_WEIGHTS_FILE}` before running the pipeline.")
st.markdown(f"- Output files (`{os.path.basename(REPORT_MD_FILE)}`, `{os.path.basename(OUTPUT_SUMMARY_FILE)}`, `run.log`) are expected in `{OUTPUT_DIR if OUTPUT_DIR != '.' else 'the project root (or outputs/ if configured for Docker)'}`.")