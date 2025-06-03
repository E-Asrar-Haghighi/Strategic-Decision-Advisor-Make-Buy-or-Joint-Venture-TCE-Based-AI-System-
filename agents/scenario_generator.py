import os
import re
import json # For attempting to parse JSON output from LLM
from crewai import Agent, Task, Crew
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ScenarioGeneratorAgent:
    """
    Scenario Planner agent that generates realistic future scenarios
    by combining demand, market stability, and partner behavior dimensions.
    Internally uses a CrewAI setup.
    """

    def __init__(self):
        logger.info("Initializing ScenarioGeneratorAgent...")
        self.agent = Agent(
            role="Expert Scenario Planner",
            goal="Generate detailed descriptions for 8 distinct future scenarios by combining demand level, market stability, and partner behavior. Each description must be comprehensive and tailored to the given business context.",
            backstory="""You are a highly experienced strategic scenario planner with deep expertise in environmental scanning,
            future studies, and transaction cost economics. You excel at creating plausible, diverse, and insightful
            scenarios that capture key uncertainties facing organizations. Your scenarios are not just descriptions;
            they provide actionable insights on transaction cost implications, success factors, and risks,
            helping decision-makers prepare for multiple possible futures.""",
            config={
                "llm": {
                    "provider": "openai",
                    "model": os.getenv("SCENARIO_GENERATOR_MODEL", "gpt-4o-mini")
                }
            },
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        logger.info(f"ScenarioGeneratorAgent initialized with model: {os.getenv('SCENARIO_GENERATOR_MODEL', 'gpt-4o-mini')}")
        self.scenario_codes = [
            "HD-S-R", "HD-S-O", "HD-V-R", "HD-V-O",
            "LD-S-R", "LD-S-O", "LD-V-R", "LD-V-O"
        ]

    def _parse_llm_scenario_output(self, llm_result: str) -> Dict[str, Dict[str, str]]:
        """
        Parses the LLM's string output to extract structured scenario descriptions.
        This is a complex parsing task. A more robust approach might involve
        instructing the LLM to return JSON.
        """
        logger.info("Starting to parse LLM scenario output...")
        scenarios_data = {}
        if not llm_result or not isinstance(llm_result, str):
            logger.warning("ScenarioGeneratorAgent: LLM result was empty or not a string.")
            return {code: {"error": "No LLM output received"} for code in self.scenario_codes}

        # Attempt 1: Try to parse as JSON (if LLM was instructed to produce JSON)
        try:
            logger.info("Attempting to parse LLM output as JSON...")
            # Pre-process to find the JSON block if it's embedded
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", llm_result)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                logger.info(f"Found JSON block in LLM output: {json_str[:200]}...")  # Log first 200 chars
                parsed_json = json.loads(json_str)
                # Basic validation: check if it's a dict and has scenario codes as keys
                if isinstance(parsed_json, dict) and all(code in parsed_json for code in self.scenario_codes):
                    logger.info("Successfully parsed LLM output as JSON with all required scenario codes.")
                    return parsed_json
                else:
                    missing_codes = [code for code in self.scenario_codes if code not in parsed_json]
                    logger.warning(f"JSON parsing succeeded but missing scenario codes: {missing_codes}")
            else:
                logger.info("No clear JSON block found in LLM output for scenarios. Will attempt regex parsing.")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM output as JSON: {e}. Will attempt regex parsing. Output preview: {llm_result[:500]}")

        # Attempt 2: Regex-based parsing (more brittle)
        logger.info("Attempting regex-based parsing of LLM output...")
        for code in self.scenario_codes:
            scenarios_data[code] = {
                "name": f"Scenario {code}", # Default name
                "narrative": "Not found",
                "characteristics": "Not found",
                "transaction_implications": "Not found",
                "success_factors": "Not found",
                "risks": "Not found"
            }
            # Example: Looking for a block for each scenario code
            pattern = re.compile(
                rf"Scenario {re.escape(code)}.*?Narrative:\s*(.*?)\s*Key Characteristics:\s*(.*?)\s*Implications for Transaction Costs:\s*(.*?)\s*Critical Success Factors:\s*(.*?)\s*Risk Factors:\s*(.*?)(?=\s*Scenario|$)",
                re.IGNORECASE | re.DOTALL
            )
            match = pattern.search(llm_result)
            if match:
                logger.info(f"Found regex match for scenario {code}")
                scenarios_data[code]["narrative"] = match.group(1).strip()
                scenarios_data[code]["characteristics"] = match.group(2).strip()
                scenarios_data[code]["transaction_implications"] = match.group(3).strip()
                scenarios_data[code]["success_factors"] = match.group(4).strip()
                scenarios_data[code]["risks"] = match.group(5).strip()
            else:
                logger.warning(f"Could not parse details for scenario {code} using regex. LLM Output preview: {llm_result[:500]}")
        
        # Log final parsing results
        logger.info(f"Final parsing results - Number of scenarios with data: {sum(1 for data in scenarios_data.values() if data['narrative'] != 'Not found')}")
        return scenarios_data

    def create_task(self, context: Dict[str, Any]) -> Task:
        """Create a task for generating scenarios."""
        logger.info("Creating scenario generation task...")
        initial_task_description = context.get('task', 'a strategic business decision')
        industry_context_guess = f"The business operates in an industry related to {initial_task_description}." # Simple guess

        # Instruction for JSON output
        json_output_instruction = """
        You MUST provide your output as a single JSON object.
        The JSON object should have keys corresponding to each scenario code (e.g., "HD-S-R", "HD-S-O", etc.).
        Each scenario code key should map to an object containing the following string fields:
        "name": "Full Scenario Name (e.g., High Demand, Stable Market, Reliable Partners)"
        "narrative": "Your 2-3 sentence vivid picture of this future."
        "characteristics": "Specific market conditions, competitive dynamics."
        "transaction_implications": "How this environment affects make/buy decisions and transaction costs."
        "success_factors": "What capabilities matter most in this scenario."
        "risks": "Primary threats and uncertainties for the decision regarding '{initial_task_description}'."

        Example for one scenario code "HD-S-R":
        "HD-S-R": {
            "name": "High Demand, Stable Market, Reliable Partners",
            "narrative": "The market for AI recommendation engines is booming, with consistent high demand. Competitive dynamics are established and predictable. External AI solution providers are trustworthy and deliver on their promises.",
            "characteristics": "Sustained growth in AI solutions, predictable competition, strong and reliable AI vendor ecosystem.",
            "transaction_implications": "Lowered transaction costs due to market stability and partner trust make outsourcing AI development attractive. However, high demand might also justify internal investment to capture full value and ensure strategic control.",
            "success_factors": "Ability to rapidly scale AI capabilities, forming strong long-term partnerships with key AI vendors, maintaining operational excellence in AI deployment.",
            "risks": "Potential capacity constraints if demand outstrips supply, over-dependency on a few key partners, risk of market saturation in the long term if innovation stalls."
        }
        Ensure the entire output is a valid JSON object, starting with { and ending with }.
        Do not include any text before or after the JSON object, including "```json" markers.
        """

        task = Task(
            description=f"""
            The company is considering a strategic decision: "{initial_task_description}".
            Assume this is for a time horizon of 3-5 years.
            {industry_context_guess}

            Generate detailed descriptions for 8 future scenarios based on three key dimensions:
            1. **Demand Level** for the outcome of "{initial_task_description}": High (HD) vs Low (LD)
            2. **Market Stability** relevant to "{initial_task_description}": Stable (S) vs Volatile (V)
            3. **Partner Behavior** of potential external partners for "{initial_task_description}": Reliable (R) vs Opportunistic (O)

            The 8 required scenario codes are:
            HD-S-R, HD-S-O, HD-V-R, HD-V-O, LD-S-R, LD-S-O, LD-V-R, LD-V-O.

            {json_output_instruction}

            Guidelines for content:
            - Make scenarios realistic and plausible for the context of "{initial_task_description}".
            - Ensure clear differentiation between scenarios.
            - Focus on factors relevant to transaction cost decisions (make, buy, ally).
            - Consider potential industry-specific dynamics if inferable from "{initial_task_description}".
            - Include both opportunities and challenges presented by each scenario.
            - Each narrative and description should be concise but impactful.
            """,
            agent=self.agent,
            expected_output="A single, valid JSON object string containing detailed descriptions for all 8 scenarios as specified. The JSON object should have scenario codes as top-level keys."
        )
        logger.info("Scenario generation task created successfully.")
        return task

    def execute(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the scenario generation task."""
        logger.info(f"ScenarioGeneratorAgent: Starting execution for task: {shared_context.get('task')}")
        task_to_run = self.create_task(shared_context)
        crew = Crew(agents=[self.agent], tasks=[task_to_run], verbose=True) # verbose=True for more detail

        llm_result = None
        scenario_descriptions = {code: {"error": "Processing not completed"} for code in self.scenario_codes} # Default

        try:
            logger.info("Kicking off internal crew for scenario generation...")
            llm_result = crew.kickoff(inputs=shared_context) # Pass context if needed
            logger.info(f"ScenarioGeneratorAgent: Raw LLM Result:\n{llm_result}")

            if llm_result and isinstance(llm_result, str):
                scenario_descriptions = self._parse_llm_scenario_output(llm_result)
                # Check if parsing was successful for all scenarios
                for code in self.scenario_codes:
                    if "error" in scenario_descriptions.get(code, {}) or scenario_descriptions.get(code,{}).get("narrative") == "Not found":
                        logger.warning(f"Scenario {code} might not have been parsed correctly.")
            else:
                logger.error("ScenarioGeneratorAgent: LLM result was empty or not a string.")
                # scenario_descriptions will retain the default error state

        except Exception as e:
            logger.exception(f"ScenarioGeneratorAgent: Error during crew kickoff or parsing: {e}. Raw LLM output (if any): {llm_result}")
            # Fallback: scenario_descriptions will retain the default error state or partially parsed data

        shared_context["scenario_descriptions"] = scenario_descriptions
        logger.info(f"ScenarioGeneratorAgent: Updated shared_context with scenario_descriptions. Number of scenarios processed: {len(scenario_descriptions)}")
        logger.info(f"DEBUG: scenario_descriptions content: {json.dumps(scenario_descriptions, indent=2)}")
        return shared_context