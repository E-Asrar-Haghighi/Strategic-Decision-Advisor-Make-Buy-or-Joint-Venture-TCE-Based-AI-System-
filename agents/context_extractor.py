import os
import re # For parsing the output
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI # Updated import
from typing import Dict, Any
import logging # For logging within the agent

logger = logging.getLogger(__name__)

class ContextExtractorAgent:
    """
    Business Analyst agent that extracts and understands business context
    for transaction cost analysis decisions. Internally uses a CrewAI setup.
    """

    def __init__(self):
        # Ensure OPENAI_API_KEY is loaded by main.py's load_dotenv()
        # and is available in the environment.
        # You can also explicitly pass os.getenv("OPENAI_API_KEY") to OpenAI()
        # if preferred, but CrewAI's OpenAI class usually picks it up.
        try:
            self.llm = ChatOpenAI(
                model=os.getenv("CONTEXT_EXTRACTOR_MODEL", "gpt-4"),
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for ContextExtractorAgent: {e}")
            # Depending on desired behavior, you might raise the error or set self.llm to None
            # and handle it in execute. For now, let's assume it must initialize.
            raise RuntimeError(f"ContextExtractorAgent: LLM initialization failed. Is OPENAI_API_KEY set? Error: {e}")


        self.agent = Agent(
            role="Business Analyst",
            goal="Understand the business capability under consideration and extract key transaction cost factors (Asset Specificity, Transaction Frequency, Uncertainty, Partner Trust).",
            backstory="""You are an experienced business analyst specializing in strategic decision-making.
            You excel at gathering context for make-vs-buy decisions by identifying key factors like
            asset specificity, transaction frequency, uncertainty levels, and partner relationships.
            You must provide an assessment (High/Medium/Low) for each factor along with your reasoning.""",
            llm=self.llm, # Pass the initialized LLM
            verbose=True,
            allow_delegation=False,
            max_iter=3 # Limit iterations to prevent runaway agent
        )

    def _parse_llm_output(self, llm_result: str) -> Dict[str, str]:
        """
        Parses the LLM's string output to extract structured factor assessments.
        Expected format from LLM:
        Asset Specificity: [High/Medium/Low] - [reasoning]
        Transaction Frequency: [High/Medium/Low] - [reasoning]
        Uncertainty: [High/Medium/Low] - [reasoning]
        Partner Trust: [High/Medium/Low] - [reasoning]
        """
        parsed_values = {
            "asset_specificity": "Unknown",
            "frequency": "Unknown",
            "uncertainty": "Unknown",
            "partner_trust": "Unknown"
        }
        # More robust regex to capture High/Medium/Low, case-insensitive, and handle variations
        patterns = {
            "asset_specificity": r"Asset Specificity:\s*(High|Medium|Low)\b.*",
            "frequency": r"Transaction Frequency:\s*(High|Medium|Low)\b.*",
            "uncertainty": r"Uncertainty:\s*(High|Medium|Low)\b.*",
            "partner_trust": r"Partner Trust:\s*(High|Medium|Low)\b.*"
        }

        if not llm_result or not isinstance(llm_result, str):
            logger.warning("LLM result for context extraction was empty or not a string.")
            return parsed_values

        for key, pattern in patterns.items():
            match = re.search(pattern, llm_result, re.IGNORECASE | re.MULTILINE)
            if match:
                # Capitalize the first letter of the extracted value (e.g., "high" -> "High")
                value = match.group(1).strip().capitalize()
                if value in ["High", "Medium", "Low"]:
                    parsed_values[key] = value
                else:
                    logger.warning(f"Parsed value '{value}' for '{key}' is not High/Medium/Low. LLM Output: {llm_result[:500]}")
            else:
                logger.warning(f"Could not find pattern for '{key}' in LLM output. LLM Output: {llm_result[:500]}")
        
        logger.info(f"Parsed context factors: {parsed_values}")
        return parsed_values

    def create_task(self, context: Dict[str, Any]) -> Task:
        """Create a task for extracting business context."""
        initial_task_description = context.get('task', 'No specific business question provided. Please assume a general business scenario.')
        
        return Task(
            description=f"""
            Analyze the business question: "{initial_task_description}"

            You MUST extract and assess the following four key transaction cost factors.
            For each factor, state whether it is High, Medium, or Low, and provide brief reasoning.
            1. **Asset Specificity**: How specialized are the required assets/capabilities for this task?
               (High: Highly specialized, difficult to redeploy; Medium: Some specialization; Low: General purpose)

            2. **Transaction Frequency**: How often will transactions related to this capability occur or this capability be used?
               (High: Regular, ongoing; Medium: Periodic; Low: One-time or rare)

            3. **Uncertainty**: What is the level of market, technological, or environmental uncertainty surrounding this capability?
               (High: Rapidly changing, unpredictable; Medium: Some volatility; Low: Stable, predictable)

            4. **Partner Trust / Opportunism Risk**: If considering external partners, what is the general level of trust or risk of opportunism with potential partners in this domain?
               (High Trust/Low Opportunism Risk, Medium Trust/Medium Opportunism Risk, Low Trust/High Opportunism Risk). Frame your answer as High, Medium, or Low for "Partner Trust".

            Your final answer MUST be ONLY in the following format, with each factor on a new line:
            Asset Specificity: [High/Medium/Low] - [Your concise reasoning here]
            Transaction Frequency: [High/Medium/Low] - [Your concise reasoning here]
            Uncertainty: [High/Medium/Low] - [Your concise reasoning here]
            Partner Trust: [High/Medium/Low] - [Your concise reasoning here]

            Do not include any other conversational text, greetings, or summaries. Only provide the four lines as specified.
            Example for Partner Trust: Partner Trust: Medium - While some established vendors exist, the rapid evolution of AI means new, unproven partners might emerge, posing moderate risk.
            """,
            agent=self.agent,
            expected_output="""
            A four-line response strictly adhering to the specified format:
            Asset Specificity: [High/Medium/Low] - [reasoning]
            Transaction Frequency: [High/Medium/Low] - [reasoning]
            Uncertainty: [High/Medium/Low] - [reasoning]
            Partner Trust: [High/Medium/Low] - [reasoning]
            """
        )

    def test_llm(self):
        """Test LLM connectivity and response."""
        try:
            response = self.llm.invoke("Say hello")
            logger.info(f"LLM is working. Response: {response}")
            return True
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            return False

    def execute(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the context extraction task using the internal CrewAI setup."""
        logger.info(f"ContextExtractorAgent: Starting execution for task: {shared_context.get('task')}")
        
        # Test LLM connectivity first
        if not self.test_llm():
            logger.error("LLM test failed. Cannot proceed with context extraction.")
            return shared_context

        task_to_run = self.create_task(shared_context)
        
        # We create a new Crew for each execution to ensure tasks are fresh,
        # though for a single-agent, single-task crew, this is straightforward.
        crew = Crew(agents=[self.agent], tasks=[task_to_run], verbose=True) # verbose=True for more detailed logs

        llm_result = None
        try:
            logger.info("Kicking off internal crew for context extraction...")
            llm_result = crew.kickoff(inputs=shared_context) # Pass context if needed by crew/tasks setup
            logger.info(f"ContextExtractorAgent: Raw LLM Result:\n{llm_result}")

            # Debug: log type and repr of llm_result
            logger.info(f"Type of llm_result: {type(llm_result)}; Value: {repr(llm_result)}")

            # Force string conversion if not already a string
            if not isinstance(llm_result, str):
                llm_result = str(llm_result)

            # Improved empty/whitespace check
            if not llm_result or not llm_result.strip():
                logger.error("ContextExtractorAgent: LLM result was empty or only whitespace. Cannot parse.")
                extracted_factors = self._parse_llm_output("")
            else:
                extracted_factors = self._parse_llm_output(llm_result)

            # Update the shared_context with parsed values
            shared_context.update(extracted_factors)
            logger.info(f"ContextExtractorAgent: Updated shared_context: {shared_context}")

        except Exception as e:
            logger.exception(f"ContextExtractorAgent: Error during crew kickoff or parsing: {e}. Raw LLM output (if any): {llm_result}")
            # Fallback or re-raise: If critical, might re-raise.
            # For now, we'll allow it to proceed with potentially "Unknown" values if parsing failed
            # or if the LLM call failed before producing a result.
            # The _parse_llm_output already defaults to "Unknown" for unparsed items.
            shared_context.update(self._parse_llm_output(llm_result if isinstance(llm_result, str) else ""))

        return shared_context