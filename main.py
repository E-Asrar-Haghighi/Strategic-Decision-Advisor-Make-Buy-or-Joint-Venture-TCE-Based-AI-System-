import json
import logging
import os
import sys
from dotenv import load_dotenv

# from markdown2 import markdown # Commented out as it seems unused for MD->HTML conversion
# from crewai import Crew, Task # Commented out as not used for orchestration in this script's current form

# Import custom agents
from agents.context_extractor import ContextExtractorAgent
from agents.scenario_generator import ScenarioGeneratorAgent
from agents.probability_collector import ProbabilityCollectorAgent
from agents.transaction_logic import TransactionLogicAgent
from agents.aggregation_agent import AggregationAgent
from agents.explanation_agent import ExplanationAgent

# Load environment variables from .env file
# This is still useful for other potential configurations like OPENAI_API_KEY
# that your agents might use.
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    filename="run.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_business_task(task_description: str) -> tuple[bool, str]:
    """
    Validates if the business task is relevant and well-formed.
    
    Args:
        task_description: The business task description to validate
        
    Returns:
        tuple[bool, str]: (is_valid, validation_message)
    """
    if not task_description or len(task_description.strip()) < 10:
        return False, "Business task description is too short or empty. Please provide a clear strategic question."
    
    # Basic check for keywords often found in make-vs-buy questions
    keywords = ["build", "buy", "make", "outsource", "partner", "develop", "acquire", "internal", "external", "jv", "alliance"]
    if not any(keyword in task_description.lower() for keyword in keywords):
        return False, "The business task does not seem to be a 'make-vs-buy-vs-ally' type question. Please rephrase to focus on how to source or develop a capability."
    
    return True, "Task appears relevant."

# --- Load Scenario Probabilities ---
try:
    with open("scenario_weights.json") as f:
        scenario_probs_raw = json.load(f)
        # Basic validation:
        if not isinstance(scenario_probs_raw, dict) or not all(isinstance(v, (int, float)) for v in scenario_probs_raw.values()):
            raise ValueError("scenario_weights.json must be a dictionary with numeric values.")
        
        scenario_probs = {k: float(v) / 100.0 for k, v in scenario_probs_raw.items()}
        
        # Check if probabilities sum approximately to 1.0, only if there are probabilities
        if scenario_probs and not (0.99 < sum(scenario_probs.values()) < 1.01):
             logger.warning(
                 f"Sum of scenario probabilities is {sum(scenario_probs.values()):.2f}, not 1.0. "
                 "Please check scenario_weights.json."
            )

except FileNotFoundError:
    logger.error("CRITICAL: scenario_weights.json not found.")
    sys.exit("Error: scenario_weights.json not found. Please ensure the file exists in the root directory.")
except json.JSONDecodeError:
    logger.error("CRITICAL: scenario_weights.json is not valid JSON.")
    sys.exit("Error: scenario_weights.json is corrupted. Please ensure it's valid JSON.")
except ValueError as ve:
    logger.error(f"CRITICAL: Invalid data in scenario_weights.json: {ve}")
    sys.exit(f"Error: Invalid data in scenario_weights.json: {ve}")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to load scenario weights due to an unexpected error: {e}", exc_info=True)
    sys.exit(f"An unexpected error occurred while loading scenario_weights.json: {e}")
# --- End Load Scenario Probabilities ---

# --- Load Business Task from Temp File (if exists) ---
BUSINESS_TASK_TEMP_FILE = "user_business_task.txt"
business_task = "Should we outsource or build an AI recommendation engine?" # Default
if os.path.exists(BUSINESS_TASK_TEMP_FILE):
    try:
        with open(BUSINESS_TASK_TEMP_FILE, "r", encoding="utf-8") as f:
            business_task = f.read().strip()
            if not business_task:
                business_task = "Should we outsource or build an AI recommendation engine?"
                logger.warning("Empty business task read from file, using default task.")
            else:
                # Validate the business task
                is_valid, validation_message = validate_business_task(business_task)
                if not is_valid:
                    logger.warning(f"Business task validation failed: {validation_message}")
                    # We'll continue with the task but log the warning
                    # The ExplanationAgent can take this into account
                else:
                    logger.info("Business task validation passed.")
    except Exception as e:
        logger.warning(f"Could not read business task from {BUSINESS_TASK_TEMP_FILE}: {e}")

shared_context = {
    "task": business_task,
    "scenario_probabilities": scenario_probs,
    "asset_specificity": None,
    "frequency": None,
    "uncertainty": None,
    "partner_trust": None,
    "scenario_descriptions": {},  # Ensure this is initialized
    "scenario_results": {},
    "final_recommendation": "N/A",
    "strategy_scores_programmatic": {},
    "llm_aggregation_synthesis": {},
    "executive_summary": "No summary generated yet."
}

# Initialize agents
# NOTE: The README mentions "Multi-agent architecture using CrewAI".
# The current implementation is a direct sequential execution of custom agent objects.
# If CrewAI is intended for orchestration, these agents would need to be refactored
# into CrewAI Agent classes, and Tasks/Crew would be defined and kicked off here.
logger.info("Initializing agents...")
try:
    context_agent = ContextExtractorAgent()
    scenario_generator_agent = ScenarioGeneratorAgent()
    probability_collector_agent = ProbabilityCollectorAgent()  # Optional, but included for completeness
    transaction_agent = TransactionLogicAgent()
    aggregation_agent = AggregationAgent()
    explanation_agent = ExplanationAgent()
    logger.info("All agents initialized successfully.")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize one or more agents: {e}", exc_info=True)
    sys.exit(f"Error during agent initialization: {e}")


try:
    # Execute agent workflow
    logger.info("Starting context extraction...")
    shared_context = context_agent.execute(shared_context)
    logger.info(
        f"Context after extraction: Asset Specificity='{shared_context.get('asset_specificity')}', "
        f"Frequency='{shared_context.get('frequency')}', Uncertainty='{shared_context.get('uncertainty')}', "
        f"Partner Trust='{shared_context.get('partner_trust')}'"
    )

    logger.info("Generating scenario descriptions...")
    shared_context = scenario_generator_agent.execute(shared_context)
    logger.info(f"Context after scenario generation. Scenario descriptions count: {len(shared_context.get('scenario_descriptions', {}))}")

    # Optional: Validate scenario probabilities
    logger.info("Validating scenario probabilities...")
    shared_context = probability_collector_agent.execute(shared_context)
    logger.info(f"Probabilities after validation. LLM Assessment: {shared_context.get('llm_probability_assessment',{}).get('llm_assessment_status')}")

    logger.info("Applying transaction cost logic...")
    shared_context = transaction_agent.execute(shared_context)
    logger.info(f"Context after transaction logic. Scenario results count: {len(shared_context.get('scenario_results', {}))}")

    logger.info("Aggregating results...")
    shared_context = aggregation_agent.execute(shared_context)
    logger.info(
        f"Context after aggregation: Final Recommendation='{shared_context.get('final_recommendation')}', "
        f"Strategy Scores count: {len(shared_context.get('strategy_scores', {}))}"
    )

    logger.info("Generating explanation...")
    shared_context = explanation_agent.execute(shared_context)
    summary_length = len(shared_context.get('executive_summary', '')) if shared_context.get('executive_summary') else 0
    logger.info(f"Context after explanation: Executive summary length: {summary_length} characters.")

    logger.info("Agent workflow completed successfully.")

except Exception as e:
    logger.critical("CRITICAL: Agent execution failed during workflow.", exc_info=True)
    # Continue to report generation with potentially incomplete data

# --- Report Generation ---
logger.info("Generating final report...")
explanation_agent = ExplanationAgent()
shared_context = explanation_agent.execute(shared_context)

# Write the markdown report
with open("report.md", "w", encoding="utf-8") as f:
    f.write(shared_context.get("executive_summary", "No executive summary generated."))

# Write the full context to output_summary.json
with open("output_summary.json", "w", encoding="utf-8") as f:
    json.dump(shared_context, f, indent=2, ensure_ascii=False)

logger.info("Final context saved to output_summary.json")
logger.info("Markdown report generated as report.md")