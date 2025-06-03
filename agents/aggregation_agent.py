import os
import json
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class AggregationAgent:
    """
    Decision Synthesizer agent that aggregates weighted strategy scores programmatically
    and uses an LLM to interpret results and consider qualitative factors.
    """

    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model=os.getenv("AGGREGATION_MODEL", "gpt-4o-mini"),
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for AggregationAgent: {e}")
            raise RuntimeError(f"AggregationAgent: LLM initialization failed. Is OPENAI_API_KEY set? Error: {e}")

        self.agent = Agent(
            role="Strategic Decision Aggregator",
            goal="Aggregate results from different scenarios and provide a final recommendation.",
            backstory="""You are an expert in strategic decision-making, specializing in synthesizing
            complex information from multiple scenarios to provide clear, actionable recommendations.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    def _calculate_weighted_scores_programmatic(self,
                                 scenario_results: Dict[str, Any],
                                 scenario_probabilities: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """Programmatically calculate probability-weighted scores for each strategy."""
        strategy_scores = {"MAKE": 0.0, "BUY": 0.0, "JV": 0.0}
        valid_scenarios_processed = 0

        if not scenario_probabilities:
            logger.warning("No scenario probabilities provided to AggregationAgent. Cannot calculate weighted scores.")
            return strategy_scores, "No recommendation due to missing probabilities."

        for scenario_code, probability in scenario_probabilities.items():
            if not isinstance(probability, (int, float)) or probability < 0:
                logger.warning(f"Invalid probability for scenario {scenario_code}: {probability}. Skipping.")
                continue

            if scenario_code in scenario_results:
                result_data = scenario_results[scenario_code]
                if not isinstance(result_data, dict):
                    logger.warning(f"Result data for scenario {scenario_code} is not a dict. Skipping.")
                    continue

                # Use 'strategy' (programmatic one) and 'confidence_programmatic'
                strategy = result_data.get("strategy")
                # Try to get programmatic confidence first, then a general 'confidence' as fallback
                confidence = result_data.get("confidence_programmatic", result_data.get("confidence", 0))


                if not isinstance(confidence, (int, float)):
                    logger.warning(f"Confidence for {scenario_code} strategy {strategy} is not numeric: {confidence}. Using 0.")
                    confidence = 0
                
                if strategy in strategy_scores:
                    strategy_scores[strategy] += probability * confidence
                    valid_scenarios_processed +=1
                else:
                    logger.warning(f"Unknown strategy '{strategy}' found for scenario {scenario_code}. Ignoring.")
            else:
                logger.warning(f"Scenario {scenario_code} (with probability {probability}) not found in scenario_results. Skipping.")
        
        if valid_scenarios_processed == 0 and any(scenario_probabilities.values()): # Probs exist but no matching results
             logger.error("No valid scenario results could be processed for weighting, though probabilities were provided.")
             return strategy_scores, "No recommendation due to inability to process scenario results."


        if not any(strategy_scores.values()): # All scores are zero
            final_recommendation = "Undetermined"
            logger.warning("All strategy scores are zero after weighting.")
        else:
            final_recommendation = max(strategy_scores, key=strategy_scores.get)

        # Round scores for cleaner output
        strategy_scores = {k: round(v, 3) for k, v in strategy_scores.items()}
        return strategy_scores, final_recommendation

    def _calculate_overall_confidence_programmatic(self, strategy_scores: Dict[str, float]) -> float:
        """Programmatically calculate overall confidence in the recommendation."""
        total_score = sum(strategy_scores.values())
        if total_score == 0:
            return 5.0  # Neutral confidence if no scores

        max_score = max(strategy_scores.values())
        # Confidence reflects the dominance of the max_score
        # If only one strategy has a score, confidence should be high if that score is positive
        if len([s for s in strategy_scores.values() if s > 0]) == 1 and max_score > 0:
            return 9.0 # High confidence if one clear positive winner
        
        confidence_raw = (max_score / total_score) * 10
        return min(10.0, max(1.0, round(confidence_raw, 1)))


    def _assess_risk_programmatic(self,
                       scenario_results: Dict[str, Any],
                       final_recommendation_programmatic: str,
                       num_total_scenarios: int) -> Dict[str, Any]:
        """Programmatically assess basic risk factors for the recommended strategy."""
        if final_recommendation_programmatic == "Undetermined" or num_total_scenarios == 0:
            return {
                "final_recommendation_programmatic": final_recommendation_programmatic,
                "support_ratio": 0,
                "risk_level_programmatic": "High (Undetermined Recommendation)",
                "strategy_distribution_programmatic": {"MAKE": 0, "BUY": 0, "JV": 0},
                "robustness_programmatic": "Very Low"
            }

        strategy_support_count = {"MAKE": 0, "BUY": 0, "JV": 0}
        for result_data in scenario_results.values():
            if isinstance(result_data, dict):
                strategy = result_data.get("strategy") # Programmatic strategy from TransactionLogicAgent
                if strategy in strategy_support_count:
                    strategy_support_count[strategy] += 1
        
        support_for_recommendation = strategy_support_count.get(final_recommendation_programmatic, 0)
        support_ratio = support_for_recommendation / num_total_scenarios if num_total_scenarios > 0 else 0
        
        risk_level = "Low"
        if support_ratio < 0.4: risk_level = "High"
        elif support_ratio < 0.65: risk_level = "Medium"
        
        robustness = "Low"
        if support_ratio >= 0.75: robustness = "High"
        elif support_ratio >= 0.5: robustness = "Medium"

        return {
            "final_recommendation_programmatic": final_recommendation_programmatic,
            "support_ratio": round(support_ratio, 3),
            "risk_level_programmatic": risk_level,
            "strategy_distribution_programmatic": strategy_support_count, # How many scenarios prefer each strategy
            "robustness_programmatic": robustness
        }

    def _parse_llm_aggregation_output(self, llm_result_str: str, fallback_recommendation: str) -> Dict[str, Any]:
        """ 
        Parses LLM output for qualitative aggregation insights. Expects JSON.
        Returns a structured dictionary with validated fields and appropriate fallbacks.
        """
        default_report = {
            "llm_final_recommendation": fallback_recommendation,
            "llm_confidence_qualitative": "Medium (LLM Parsing Failed or No Input)",
            "llm_overall_justification": "Could not parse LLM output for justification or no input provided to parser.",
            "llm_risk_tolerance_consideration": "Not assessed by LLM.",
            "llm_strategic_control_consideration": "Not assessed by LLM.",
            "llm_resource_constraints_consideration": "Not assessed by LLM.",
            "llm_time_horizon_consideration": "Not assessed by LLM."
        }
        
        # More robust check for empty or non-string input
        if not llm_result_str or not isinstance(llm_result_str, str) or not llm_result_str.strip():
            logger.warning(f"_parse_llm_aggregation_output: Input llm_result_str is empty, None, not a string, or only whitespace. Type: {type(llm_result_str)}. Value (repr): {repr(llm_result_str)}. Returning default report.")
            return default_report
        
        # Attempt to strip the string once before parsing
        processed_llm_result_str = llm_result_str.strip()

        try:
            # Try to find a JSON block
            # Regex to find content within ```json ... ``` or just a plain JSON object/array
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|(\{[\s\S]*\}|\[[\s\S]*\])", processed_llm_result_str, re.DOTALL)
            if not json_match:
                logger.warning("No JSON block found in LLM output. Attempting to parse entire output as JSON.")
                json_str = processed_llm_result_str
            else:
                json_str = json_match.group(1) or json_match.group(2)
                logger.info(f"Found JSON block in LLM output: {json_str[:200]}...")

            # Parse JSON with better error handling
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e}. Attempting to clean and reparse JSON.")
                # Try to clean the JSON string by removing any non-JSON text
                cleaned_json = re.sub(r'[^\x00-\x7F]+', '', json_str)  # Remove non-ASCII
                cleaned_json = re.sub(r'[\n\r\t]', ' ', cleaned_json)  # Normalize whitespace
                try:
                    parsed_data = json.loads(cleaned_json)
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse even after cleaning: {e2}")
                    default_report["llm_overall_justification"] = f"JSON parsing failed: {e2}. Raw output: {llm_result_str[:500]}"
                    return default_report

            # Validate parsed data structure
            if not isinstance(parsed_data, dict):
                logger.error(f"Parsed data is not a dictionary: {type(parsed_data)}")
                default_report["llm_overall_justification"] = "LLM output is not a valid JSON object."
                return default_report

            # Extract and validate required fields with type checking
            final_report = {}
            
            # Validate final_recommendation
            recommendation = parsed_data.get("final_recommendation")
            if not isinstance(recommendation, str):
                logger.warning(f"Invalid final_recommendation type: {type(recommendation)}")
                final_report["llm_final_recommendation"] = fallback_recommendation
            else:
                recommendation = recommendation.strip().upper()
                if recommendation not in ["MAKE", "BUY", "JV", "UNDETERMINED"]:
                    logger.warning(f"Invalid recommendation value: {recommendation}")
                    final_report["llm_final_recommendation"] = fallback_recommendation
                else:
                    final_report["llm_final_recommendation"] = recommendation

            # Validate confidence_qualitative
            confidence = parsed_data.get("confidence_qualitative")
            if not isinstance(confidence, str):
                logger.warning(f"Invalid confidence_qualitative type: {type(confidence)}")
                final_report["llm_confidence_qualitative"] = "Medium (Invalid confidence format)"
            else:
                confidence = confidence.strip()
                valid_confidences = ["High", "Medium-High", "Medium", "Medium-Low", "Low"]
                if confidence not in valid_confidences:
                    logger.warning(f"Invalid confidence value: {confidence}")
                    final_report["llm_confidence_qualitative"] = "Medium (Invalid confidence value)"
                else:
                    final_report["llm_confidence_qualitative"] = confidence

            # Extract and validate other fields with appropriate fallbacks
            for field, default in [
                ("overall_justification", "No justification provided."),
                ("risk_tolerance_consideration", "Not discussed."),
                ("strategic_control_consideration", "Not discussed."),
                ("resource_constraints_consideration", "Not discussed."),
                ("time_horizon_consideration", "Not discussed.")
            ]:
                value = parsed_data.get(field)
                if not isinstance(value, str):
                    logger.warning(f"Invalid {field} type: {type(value)}")
                    final_report[f"llm_{field}"] = default
                else:
                    final_report[f"llm_{field}"] = value.strip() or default

            logger.info(f"Successfully parsed LLM output with recommendation: {final_report['llm_final_recommendation']}")
            return final_report

        except Exception as e:
            logger.exception(f"Unexpected error parsing LLM aggregation output: {e}")
            default_report["llm_overall_justification"] = f"Unexpected error during parsing: {str(e)}"
            return default_report


    def create_llm_task(self, context: Dict[str, Any], programmatic_scores: Dict, programmatic_recommendation: str, programmatic_confidence: float, programmatic_risk_assessment: Dict) -> Task:
        """Create a task for the LLM to interpret aggregation results and consider qualitative factors."""
        
        context_summary_for_llm = f"""
        Initial Business Task: {context.get('task', 'N/A')}
        Overall Asset Specificity: {context.get('asset_specificity', 'N/A')}
        Overall Transaction Frequency: {context.get('frequency', 'N/A')}
        Overall Uncertainty: {context.get('uncertainty', 'N/A')}
        Overall Partner Trust: {context.get('partner_trust', 'N/A')}
        Scenario Probabilities: {json.dumps(context.get('scenario_probabilities', {}), indent=1)}
        Scenario Results (Strategy per scenario from TransactionLogicAgent - showing first 2 for brevity):
        {json.dumps({k:v for i,(k,v) in enumerate(context.get('scenario_results', {}).items()) if i<2}, indent=1)}
        """

        programmatic_analysis_summary = f"""
        Programmatic Weighted Scores: {json.dumps(programmatic_scores, indent=1)}
        Programmatic Recommendation: {programmatic_recommendation}
        Programmatic Confidence (0-10): {programmatic_confidence:.1f}
        Programmatic Risk Assessment: {json.dumps(programmatic_risk_assessment, indent=1)}
        """

        json_output_instruction = """
        You MUST provide your output as a single JSON object.
        The JSON object should contain the following string fields:
        - "final_recommendation": "string (Your final strategy choice: MAKE, BUY, or JV, potentially confirming or (rarely, with strong justification) differing from programmatic.)"
        - "confidence_qualitative": "string (Your qualitative confidence in this final recommendation: e.g., High, Medium-High, Medium, Medium-Low, Low)"
        - "overall_justification": "string (A concise paragraph explaining your final recommendation, integrating the weighted scores with qualitative strategic considerations. Why is this the best overall path?)"
        - "risk_tolerance_consideration": "string (How does the organization's likely risk tolerance influence this choice?)"
        - "strategic_control_consideration": "string (How important is strategic control over this capability, and how does the recommendation align?)"
        - "resource_constraints_consideration": "string (Briefly, how do potential resource constraints (financial, operational) affect the feasibility of this recommendation?)"
        - "time_horizon_consideration": "string (How does the decision's time horizon (short-term vs. long-term) impact this strategic choice?)"

        Ensure the entire output is a valid JSON object starting with { and ending with }. Do not include any markdown like ```json.
        """

        return Task(
            description=f"""
            As Strategic Decision Aggregator, synthesize all available information to provide a final, holistic strategic recommendation.
            You have programmatic analysis results, but your role is to add a layer of strategic judgment.

            **Context & Programmatic Analysis Summary:**
            {context_summary_for_llm}
            ---
            {programmatic_analysis_summary}
            ---

            **Your Task:**
            Review the programmatic analysis. Then, considering the broader strategic factors, provide your final assessment.
            Qualitative Strategic Factors to Consider:
            - **Risk Tolerance**: How does the likely organizational risk appetite affect the choice? (e.g., High tolerance might accept a strategy robust in fewer scenarios if payoff is high).
            - **Strategic Control**: How critical is maintaining deep control over this capability for competitive advantage?
            - **Resource Constraints**: Are there unspoken financial, talent, or operational limitations that might favor one strategy over another, even if scores are close?
            - **Time Horizon**: Does a short-term vs. long-term perspective shift the preference?

            Based on your synthesis of quantitative scores and these qualitative factors, provide your output.
            {json_output_instruction}
            """,
            agent=self.agent,
            expected_output="A single, valid JSON object containing the fields: final_recommendation, confidence_qualitative, overall_justification, and considerations for risk_tolerance, strategic_control, resource_constraints, and time_horizon."
        )

    def execute(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes programmatic aggregation and then uses LLM for qualitative synthesis."""
        logger.info("AggregationAgent: Starting execution.")

        scenario_results = shared_context.get('scenario_results', {})
        scenario_probabilities = shared_context.get('scenario_probabilities', {})
        
        # 1. Programmatic Calculations
        prog_scores, prog_recommendation = self._calculate_weighted_scores_programmatic(scenario_results, scenario_probabilities)
        prog_confidence = self._calculate_overall_confidence_programmatic(prog_scores)
        num_scenarios = len(scenario_probabilities) # Or len(scenario_results) if more robust
        prog_risk_assessment = self._assess_risk_programmatic(scenario_results, prog_recommendation, num_scenarios)

        shared_context["strategy_scores_programmatic"] = prog_scores
        shared_context["final_recommendation_programmatic"] = prog_recommendation
        shared_context["aggregation_confidence_programmatic"] = prog_confidence
        shared_context["risk_assessment_programmatic"] = prog_risk_assessment

        logger.info(f"Programmatic Aggregation: Scores={prog_scores}, Rec={prog_recommendation}, Conf={prog_confidence:.1f}, Risk={prog_risk_assessment}")

        # 2. LLM for Qualitative Synthesis and Final Recommendation
        llm_synthesis_output = {} # Initialize
        raw_llm_kickoff_result = None # To store the direct output of kickoff

        try:
            llm_task = self.create_llm_task(shared_context, prog_scores, prog_recommendation, prog_confidence, prog_risk_assessment)
            crew = Crew(agents=[self.agent], tasks=[llm_task], verbose=True)
            logger.info("Kicking off internal crew for LLM qualitative aggregation synthesis...")
            
            raw_llm_kickoff_result = crew.kickoff(inputs=shared_context) # Store the direct result
            
            # --- DETAILED DEBUG LOGGING ---
            logger.debug(f"AggregationAgent: Type of raw_llm_kickoff_result: {type(raw_llm_kickoff_result)}")
            logger.debug(f"AggregationAgent: Value of raw_llm_kickoff_result (repr): {repr(raw_llm_kickoff_result)}")
            if isinstance(raw_llm_kickoff_result, str):
                logger.debug(f"AggregationAgent: Length of raw_llm_kickoff_result (if string): {len(raw_llm_kickoff_result)}")
                logger.debug(f"AggregationAgent: raw_llm_kickoff_result (first 500 chars if string): {raw_llm_kickoff_result[:500]}")
            # --- END DETAILED DEBUG LOGGING ---

            # The variable passed to parsing should be the string content.
            # Let's ensure we are working with a string if kickoff returns something else that contains the string.
            llm_result_str_for_parsing = None
            if isinstance(raw_llm_kickoff_result, str):
                llm_result_str_for_parsing = raw_llm_kickoff_result
            elif isinstance(raw_llm_kickoff_result, dict) and 'result' in raw_llm_kickoff_result and isinstance(raw_llm_kickoff_result['result'], str):
                # CrewAI sometimes wraps results in a dict, e.g. if using specific output formats or tools
                logger.info("CrewAI kickoff returned a dict, extracting 'result' field as string.")
                llm_result_str_for_parsing = raw_llm_kickoff_result['result']
            elif raw_llm_kickoff_result is not None:
                # If it's something else, try to convert to string, but this might indicate a deeper issue.
                logger.warning(f"CrewAI kickoff returned an unexpected type: {type(raw_llm_kickoff_result)}. Attempting str conversion.")
                llm_result_str_for_parsing = str(raw_llm_kickoff_result)

            # Log what will be sent to the parser
            logger.info(f"AggregationAgent: Data being sent to _parse_llm_aggregation_output (type {type(llm_result_str_for_parsing)}):\n{repr(llm_result_str_for_parsing)[:500]}")

            if llm_result_str_for_parsing: # Check if we have something to parse
                llm_synthesis_output = self._parse_llm_aggregation_output(llm_result_str_for_parsing, prog_recommendation)
            else:
                logger.warning("LLM result (raw_llm_kickoff_result or extracted string) was effectively empty or None before parsing. Using default report.")
                # This is where your previous warning was likely triggered
                llm_synthesis_output = self._parse_llm_aggregation_output(None, prog_recommendation)

        except Exception as e:
            logger.exception(f"AggregationAgent: Error during LLM crew kickoff or processing for synthesis: {e}")
            logger.error(f"AggregationAgent: raw_llm_kickoff_result at time of exception: {repr(raw_llm_kickoff_result)}")
            llm_synthesis_output = self._parse_llm_aggregation_output(None, prog_recommendation) # Use defaults

        # Store LLM's synthesis
        shared_context["llm_aggregation_synthesis"] = llm_synthesis_output

        # The 'final_recommendation' in shared_context will be the one from the LLM's synthesis.
        # 'strategy_scores' will be the programmatically calculated ones.
        shared_context["final_recommendation"] = llm_synthesis_output.get("llm_final_recommendation", prog_recommendation)
        shared_context["strategy_scores"] = prog_scores # These are the quantitative scores

        logger.info(f"LLM Aggregation Synthesis: {llm_synthesis_output}")
        logger.info(f"Final Recommendation decided by LLM (or fallback): {shared_context['final_recommendation']}")
        
        return shared_context