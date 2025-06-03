import os
import json
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

class TransactionLogicAgent:
    """
    Strategy Evaluator agent that applies Coase and Williamson transaction cost theory
    programmatically and then uses an LLM to enrich the explanation and risk assessment.
    """
    scenario_codes = [
        "HD-S-R", "HD-S-O", "HD-V-R", "HD-V-O",
        "LD-S-R", "LD-S-O", "LD-V-R", "LD-V-O"
    ]

    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model=os.getenv("TRANSACTION_LOGIC_MODEL", "gpt-4o-mini"),
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for TransactionLogicAgent: {e}")
            raise RuntimeError(f"TransactionLogicAgent: LLM initialization failed. Is OPENAI_API_KEY set? Error: {e}")

        self.agent = Agent(
            role="Strategic Economist and Risk Analyst",
            goal="Elaborate on transaction cost-based strategic recommendations, providing nuanced reasoning, confidence assessment, and risk analysis for each scenario.",
            backstory="""You are a distinguished strategic economist with profound expertise in transaction cost economics (Coase, Williamson)
            and its practical application to corporate strategy (make-vs-buy, joint ventures, outsourcing).
            You are adept at not just applying theoretical frameworks but also at articulating the 'why' behind strategic choices,
            assessing confidence, and identifying key risks and mitigations in diverse business scenarios.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    def _williamson_logic_programmatic(self, asset_specificity: str, frequency: str,
                             uncertainty: str, partner_trust: str,
                             demand: str, stability: str, reliability: str) -> Dict[str, Any]:
        """Programmatically apply Williamson's framework for a specific scenario."""
        specificity_score = {"Low": 1, "Medium": 2, "High": 3}.get(asset_specificity.capitalize(), 2)
        frequency_score = {"Low": 1, "Medium": 2, "High": 3}.get(frequency.capitalize(), 2)
        uncertainty_score = {"Low": 1, "Medium": 2, "High": 3}.get(uncertainty.capitalize(), 2)
        trust_score = {"Low": 1, "Medium": 2, "High": 3}.get(partner_trust.capitalize(), 2)

        demand_modifier = 1.15 if demand == "HD" else 0.85 # Slightly adjusted modifiers
        stability_modifier = 0.9 if stability == "S" else 1.1
        # For reliability_modifier, high reliability (R) should make MAKE less necessary relative to BUY/JV if TCE costs are low
        # Opportunistic (O) partners increase TCE for BUY/JV, making MAKE relatively more attractive.
        # So, reliability_modifier should affect MAKE negatively if partners are reliable, or positively if opportunistic.
        # Let's adjust how reliability_modifier affects MAKE_SCORE.
        # If reliable, TCE for external options are lower, so MAKE is less preferred relatively.
        # If opportunistic, TCE for external options are higher, MAKE is more preferred.
        make_reliability_factor = 1.1 if reliability == "O" else 0.9 # If Opp, MAKE is more attractive

        make_score = (specificity_score * frequency_score * make_reliability_factor) * demand_modifier
        # For BUY, high trust & stability are good. Low specificity helps.
        buy_score = ((4 - specificity_score) * trust_score) * stability_modifier
        if reliability == "O": # Opportunistic partners make BUY much less attractive
            buy_score *= 0.5
        # For JV, it's a hybrid.
        jv_score = (specificity_score * 0.5 + frequency_score * 0.5 + trust_score) * (1 if stability == "S" else 0.8)
        if reliability == "O":
            jv_score *= 0.7


        scores = {"MAKE": make_score, "BUY": buy_score, "JV": jv_score}
        recommended_strategy = max(scores, key=scores.get)
        
        # Confidence based on how much higher the top score is than the next best
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence_raw = 10
        if len(sorted_scores) > 1:
            diff_ratio = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] if sorted_scores[0] > 0 else 1)
            confidence_raw = 5 + (diff_ratio * 10) # Scale difference to 0-5, add to base 5
        confidence = min(10, max(1, int(round(confidence_raw))))


        reasoning = self._generate_programmatic_reasoning(
            recommended_strategy, demand, stability, reliability,
            asset_specificity, frequency, uncertainty, partner_trust, scores
        )

        return {
            "strategy": recommended_strategy,
            "confidence_programmatic": confidence,
            "reasoning_programmatic": reasoning,
            "scores_programmatic": {k: round(v, 2) for k,v in scores.items()}
        }

    def _generate_programmatic_reasoning(self, strategy: str, demand: str, stability: str, reliability: str,
                               asset_specificity: str, frequency: str, uncertainty: str, partner_trust: str, scores: Dict) -> str:
        """Generate basic reasoning for the programmatic recommendation."""
        as_val = asset_specificity.capitalize()
        fq_val = frequency.capitalize()
        uc_val = uncertainty.capitalize()
        pt_val = partner_trust.capitalize()

        reason = f"Strategy {strategy} recommended with score {scores[strategy]:.2f} (Buy: {scores['BUY']:.2f}, JV: {scores['JV']:.2f}, Make: {scores['MAKE']:.2f}). "
        reason += f"Context: Asset Specificity={as_val}, Frequency={fq_val}, Uncertainty={uc_val}, Partner Trust={pt_val}. "
        reason += f"Scenario: Demand={demand}, Stability={stability}, Partner Reliability={reliability}. "

        if strategy == "MAKE":
            reason += f"MAKE is favored due to factors like high asset specificity ({as_val}) and/or high transaction frequency ({fq_val}), especially if partners are opportunistic ({reliability}). High demand ({demand}) also supports internal control."
        elif strategy == "BUY":
            reason += f"BUY is favored when asset specificity ({as_val}) is low, partners are reliable ({reliability}), and market conditions are stable ({stability}). Low frequency ({fq_val}) transactions also suit market procurement."
        elif strategy == "JV":
            reason += f"JV (Hybrid) is a balanced approach suitable for medium asset specificity ({as_val}) or frequency ({fq_val}), or when needing to share risks/rewards, especially with reliable partners ({reliability}) in moderately uncertain ({uc_val}) environments."
        return reason.strip()

    def _parse_llm_transaction_logic_output(self, llm_result_str: str, num_scenarios: int) -> Dict[str, Any]:
        """ Parses LLM output, expecting JSON for all scenarios. """
        default_output = {code: {"error": "LLM parsing failed"} for code in self.scenario_codes}
        if not llm_result_str or not isinstance(llm_result_str, str):
            return default_output
        try:
            # Try to find a JSON block
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", llm_result_str)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                parsed_data = json.loads(json_str)
                # Basic validation: is it a dict and has keys for our scenarios?
                if isinstance(parsed_data, dict) and all(s_code in parsed_data for s_code in self.scenario_codes):
                    # Further validate structure of each scenario's data
                    for s_code in self.scenario_codes:
                        if not all(k in parsed_data[s_code] for k in ["llm_reasoning", "llm_confidence", "llm_risk_assessment"]):
                            parsed_data[s_code]["error"] = "Missing required fields from LLM output."
                            logger.warning(f"LLM output for scenario {s_code} is missing required fields.")
                    return parsed_data
                else: # Does not have all scenario codes or is not a dict
                    logger.warning(f"LLM JSON output does not contain all required scenario codes or is not a dictionary. Parsed: {parsed_data}")
                    return default_output # Or try to salvage what's there
            else: # No JSON block found
                logger.warning(f"No JSON block found in LLM output for transaction logic. Output: {llm_result_str[:500]}")
                return default_output
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError parsing LLM output for transaction logic: {e}. Output: {llm_result_str[:500]}")
            return default_output
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM transaction logic output: {e}. Output: {llm_result_str[:500]}")
            return default_output


    def create_llm_task(self, context: Dict[str, Any], programmatic_results: Dict[str, Any]) -> Task:
        """Create a task for the LLM to elaborate on programmatic transaction cost logic."""
        
        # Prepare programmatic results string for the LLM
        programmatic_results_str = json.dumps(programmatic_results, indent=2)

        json_output_instruction = """
        You MUST provide your output as a single JSON object.
        The JSON object should have keys corresponding to each scenario code (e.g., "HD-S-R", "HD-S-O", etc.).
        Each scenario code key should map to an object containing the following string fields:
        - "llm_reasoning": "Your detailed qualitative explanation (2-3 sentences) for the recommended strategy in this scenario, considering all input factors and TCE principles. Elaborate on the programmatic reasoning provided."
        - "llm_confidence": "Your qualitative assessment of confidence in this recommendation (e.g., 'High', 'Medium-High', 'Medium', 'Medium-Low', 'Low'), along with a brief justification for this confidence level."
        - "llm_risk_assessment": "Identify 1-2 key risks associated with the recommended strategy in this specific scenario and suggest a brief mitigation for each."

        Example for one scenario "HD-S-R" where programmatic recommended "MAKE":
        "HD-S-R": {
            "llm_reasoning": "Internalizing ('MAKE') this capability is strongly advised under High Demand, Stable Market, and Reliable Partner conditions. While reliable partners reduce outsourcing risk, the high asset specificity and frequency, coupled with strong demand, suggest that capturing full value and ensuring strategic control via hierarchy is optimal. This aligns with TCE principles where specific, frequent transactions are best managed internally to mitigate potential hold-up, even with trustworthy external parties.",
            "llm_confidence": "High - The convergence of high asset specificity, high frequency, and strong demand provides a clear case for internal development based on Williamson's framework.",
            "llm_risk_assessment": "Risk: High upfront investment and potential for internal inefficiencies. Mitigation: Phased rollout and continuous benchmarking against market alternatives. Risk: Underutilization if demand forecasts are overly optimistic. Mitigation: Develop flexible internal capacity that can be partially repurposed."
        }
        Ensure the entire output is a valid JSON object, starting with { and ending with }. Do not include any markdown like ```json.
        """

        return Task(
            description=f"""
            The business is considering: "{context.get('task', 'a strategic initiative')}".
            Key overarching factors are:
            - Asset Specificity: {context.get('asset_specificity', 'Not assessed')}
            - Transaction Frequency: {context.get('frequency', 'Not assessed')}
            - Overall Uncertainty Level: {context.get('uncertainty', 'Not assessed')}
            - General Partner Trust: {context.get('partner_trust', 'Not assessed')}

            A programmatic analysis based on Transaction Cost Economics (TCE) has been performed for 8 scenarios, yielding the following initial recommendations and scores:
            {programmatic_results_str}

            Your task is to review these programmatic results for each of the 8 scenarios. For each scenario:
            1.  Elaborate on the `reasoning_programmatic` to provide a richer, more nuanced `llm_reasoning` (2-3 sentences). Explain *why* the recommended strategy (MAKE, BUY, or JV) is suitable given the combination of asset specificity, frequency, overall uncertainty, partner trust, and the specific scenario's characteristics (Demand, Stability, Partner Reliability).
            2.  Provide an `llm_confidence` assessment (qualitative: High, Medium-High, Medium, Medium-Low, Low) for the recommended strategy, justifying your assessment. This can differ from `confidence_programmatic`.
            3.  Identify 1-2 key `llm_risk_assessment` points associated with the recommended strategy *in that specific scenario*, and suggest brief mitigations.

            {json_output_instruction}
            """,
            agent=self.agent,
            expected_output="A single, valid JSON object string. The object's keys are scenario codes (e.g., 'HD-S-R'). Each key maps to an object with 'llm_reasoning', 'llm_confidence', and 'llm_risk_assessment'."
        )

    def execute(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes programmatic TCE logic and then uses LLM to enrich explanations.
        """
        logger.info("TransactionLogicAgent: Starting execution.")
        asset_specificity = shared_context.get('asset_specificity', 'Medium')
        frequency = shared_context.get('frequency', 'Medium')
        uncertainty = shared_context.get('uncertainty', 'Medium')
        partner_trust = shared_context.get('partner_trust', 'Medium')

        # 1. Apply programmatic transaction cost logic first
        programmatic_scenario_results = {}
        for code in self.scenario_codes:
            parts = code.split('-') # HD-S-R -> [HD, S, R]
            demand, stability, reliability = parts[0], parts[1], parts[2]
            programmatic_scenario_results[code] = self._williamson_logic_programmatic(
                asset_specificity, frequency, uncertainty, partner_trust,
                demand, stability, reliability
            )
        logger.info(f"Programmatic TCE results: {json.dumps(programmatic_scenario_results, indent=2)}")

        # 2. Use LLM to elaborate and add qualitative insights
        llm_enriched_results = {}
        try:
            llm_task = self.create_llm_task(shared_context, programmatic_scenario_results)
            crew = Crew(agents=[self.agent], tasks=[llm_task], verbose=True)
            logger.info("Kicking off internal crew for LLM elaboration on TCE logic...")
            llm_output_str = crew.kickoff(inputs=shared_context) # Context for the overall task
            logger.info(f"TransactionLogicAgent: Raw LLM output for TCE elaboration:\n{llm_output_str}")

            if llm_output_str:
                llm_parsed_data = self._parse_llm_transaction_logic_output(llm_output_str, len(self.scenario_codes))
                # Merge LLM insights with programmatic results
                for code in self.scenario_codes:
                    llm_enriched_results[code] = programmatic_scenario_results[code].copy() # Start with programmatic
                    if code in llm_parsed_data and "error" not in llm_parsed_data[code]:
                        llm_enriched_results[code].update(llm_parsed_data[code]) # Add/overwrite with LLM fields
                    else:
                        logger.warning(f"LLM parsing failed or data missing for scenario {code}. Using only programmatic results for this scenario.")
                        llm_enriched_results[code]["llm_reasoning"] = "LLM elaboration not available."
                        llm_enriched_results[code]["llm_confidence"] = "Not assessed by LLM."
                        llm_enriched_results[code]["llm_risk_assessment"] = "Not assessed by LLM."
            else:
                logger.warning("LLM returned no output for TCE elaboration. Using only programmatic results.")
                llm_enriched_results = programmatic_scenario_results # Fallback
                for code in llm_enriched_results: # Add placeholder LLM fields
                    llm_enriched_results[code]["llm_reasoning"] = "LLM elaboration not available due to no output."
                    llm_enriched_results[code]["llm_confidence"] = "Not assessed by LLM."
                    llm_enriched_results[code]["llm_risk_assessment"] = "Not assessed by LLM."


        except Exception as e:
            logger.exception(f"TransactionLogicAgent: Error during LLM crew kickoff or parsing for TCE: {e}")
            llm_enriched_results = programmatic_scenario_results # Fallback to programmatic on error
            for code in llm_enriched_results: # Add placeholder LLM fields
                 llm_enriched_results[code]["llm_reasoning"] = f"LLM elaboration failed: {e}"
                 llm_enriched_results[code]["llm_confidence"] = "Not assessed by LLM."
                 llm_enriched_results[code]["llm_risk_assessment"] = "Not assessed by LLM."


        shared_context["scenario_results"] = llm_enriched_results
        logger.info(f"Final scenario results (programmatic + LLM): {json.dumps(llm_enriched_results, indent=2)}")
        return shared_context