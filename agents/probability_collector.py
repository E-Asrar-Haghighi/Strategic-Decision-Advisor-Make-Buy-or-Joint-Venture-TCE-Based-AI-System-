import os
import json
from crewai import Agent, Task, Crew
from typing import Dict, Any, Tuple # Corrected Tuple import
import logging
import re

logger = logging.getLogger(__name__)

class ProbabilityCollectorAgent:
    """
    Risk Analyst agent that validates scenario probabilities using an LLM
    and ensures they are properly formatted for the analysis.
    """
    required_scenarios = [
        "HD-S-R", "HD-S-O", "HD-V-R", "HD-V-O",
        "LD-S-R", "LD-S-O", "LD-V-R", "LD-V-O"
    ]

    def __init__(self):
        self.agent = Agent(
            role="Expert Risk Analyst and Probabilistic Forecaster",
            goal="Critically evaluate a given set of scenario probabilities for completeness, coherence, and reasonableness. Provide actionable feedback and, if necessary, suggest normalized probabilities.",
            backstory="""You are a seasoned risk analyst with deep expertise in probability assessment, Bayesian reasoning,
            and scenario planning. You understand the cognitive biases that affect probability estimation and excel
            at guiding stakeholders to refine their judgments. Your goal is to ensure that the probabilities used
            for decision analysis are as robust and well-calibrated as possible. You are not just a validator;
            you are a coach for better probabilistic thinking.""",
            config={
                "llm": {
                    "provider": "openai",
                    "model": os.getenv("PROBABILITY_COLLECTOR_MODEL", "gpt-4o-mini")
                }
            },
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    def _programmatic_probability_validation(self, probabilities: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Performs basic programmatic validation and normalization.
        This can serve as input to the LLM or a fallback.
        """
        validation_report = {
            "status": "PASS",
            "issues": [],
            "corrections_made": [],
            "quality_score": 10, # Start with a perfect score
            "notes": []
        }
        
        validated_probs = probabilities.copy()

        # 1. Completeness
        missing_scenarios = [s for s in self.required_scenarios if s not in validated_probs]
        if missing_scenarios:
            validation_report["issues"].append(f"Missing probabilities for scenarios: {', '.join(missing_scenarios)}.")
            validation_report["status"] = "FAIL" # Critical issue
            validation_report["quality_score"] -= 4
            # Fill missing with a very small placeholder to allow sum calculation, LLM should address this
            placeholder_prob = 0.001 
            for s_code in missing_scenarios:
                validated_probs[s_code] = placeholder_prob
            validation_report["corrections_made"].append(f"Temporarily assigned {placeholder_prob} to missing scenarios for calculation purposes. This needs revision.")
            validation_report["notes"].append("Critical: Probabilities are incomplete.")


        # Ensure all required scenarios are in validated_probs before proceeding
        for s_code in self.required_scenarios:
            if s_code not in validated_probs:
                 # This case should ideally be handled by the missing_scenarios check above.
                 # If it still happens, it's an issue.
                validated_probs[s_code] = 0.0 # Assign 0 if absolutely missing after first check
                validation_report["issues"].append(f"Scenario {s_code} was unexpectedly missing and set to 0. This is an error.")
                validation_report["status"] = "FAIL"


        # 2. Sum Check (Normalization)
        current_sum = sum(v for v in validated_probs.values() if isinstance(v, (int, float)))

        if abs(current_sum - 1.0) > 0.01 and current_sum > 0: # Allow for small floating point inaccuracies
            validation_report["issues"].append(f"Probabilities sum to {current_sum:.3f}, which is not 1.0.")
            validation_report["quality_score"] -= 2
            # Normalize
            normalized_probs = {k: v / current_sum for k, v in validated_probs.items()}
            validation_report["corrections_made"].append(f"Probabilities were normalized to sum to 1.0. Original sum was {current_sum:.3f}.")
            validated_probs = normalized_probs
        elif current_sum == 0 and len(validated_probs) == len(self.required_scenarios):
            validation_report["issues"].append("All probabilities are zero or invalid, cannot normalize.")
            validation_report["status"] = "FAIL"
            validation_report["quality_score"] = 0 # Drastic issue

        # 3. Range Check (on potentially normalized probabilities)
        for scenario, prob in validated_probs.items():
            if not isinstance(prob, (int,float)): # Check if prob is a number
                validation_report["issues"].append(f"Probability for {scenario} is not a valid number ({prob}).")
                validation_report["status"] = "FAIL"
                validation_report["quality_score"] -=2
                continue # Skip range check for non-numeric

            if prob < 0.01 and prob != 0: # Allow zero if it was a placeholder for truly missing
                validation_report["issues"].append(f"Probability for {scenario} ({prob:.3f}) is extremely low (less than 1%). Consider if this scenario is truly negligible or if its probability is underestimated.")
                validation_report["quality_score"] -= 1
            if prob > 0.5: # >50% for one out of 8 scenarios is very high
                validation_report["issues"].append(f"Probability for {scenario} ({prob:.3f}) is very high (greater than 50%). This suggests strong confidence in this specific outcome.")
                validation_report["quality_score"] -= 1
        
        if validation_report["quality_score"] < 0: validation_report["quality_score"] = 0
        if validation_report["quality_score"] < 7 and validation_report["status"] == "PASS":
             validation_report["status"] = "WARNING"

        return validated_probs, validation_report

    def _parse_llm_validation_output(self, llm_result_str: str, original_probs: Dict[str, float]) -> Dict[str, Any]:
        """
        Parses the LLM's string output for validation summary and corrected probabilities.
        Expects LLM to output a JSON string.
        """
        default_report = {
            "llm_assessment_status": "Parsing Failed",
            "llm_summary": "Could not parse LLM output.",
            "llm_issues_identified": [],
            "llm_recommendations": [],
            "llm_suggested_probabilities": original_probs # Fallback to original
        }
        if not llm_result_str or not isinstance(llm_result_str, str):
            return default_report

        try:
            # Look for a JSON block
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", llm_result_str)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                parsed_data = json.loads(json_str)

                # Validate structure of parsed_data
                final_report = {
                    "llm_assessment_status": parsed_data.get("assessment_status", "Status Not Provided"),
                    "llm_summary": parsed_data.get("summary", "No summary provided."),
                    "llm_issues_identified": parsed_data.get("issues_identified", []),
                    "llm_recommendations": parsed_data.get("recommendations", []),
                    "llm_suggested_probabilities": parsed_data.get("suggested_probabilities", original_probs)
                }
                # Further validation on suggested_probabilities if needed (e.g., sum to 1)
                if isinstance(final_report["llm_suggested_probabilities"], dict):
                     # Basic check to ensure all required scenarios are present
                    if not all(s_code in final_report["llm_suggested_probabilities"] for s_code in self.required_scenarios):
                        final_report["llm_issues_identified"].append("LLM suggested probabilities do not include all required scenarios. Reverting to original probabilities for safety.")
                        final_report["llm_suggested_probabilities"] = original_probs
                    # Ensure values are floats
                    try:
                        final_report["llm_suggested_probabilities"] = {k: float(v) for k,v in final_report["llm_suggested_probabilities"].items()}
                    except ValueError:
                         final_report["llm_issues_identified"].append("LLM suggested probabilities contain non-numeric values. Reverting to original probabilities.")
                         final_report["llm_suggested_probabilities"] = original_probs

                else: # Not a dict
                    final_report["llm_issues_identified"].append("LLM suggested_probabilities is not in the correct dictionary format. Reverting to original probabilities.")
                    final_report["llm_suggested_probabilities"] = original_probs
                return final_report
            else: # No JSON block found
                 default_report["llm_summary"] = "No JSON block found in LLM output. LLM might have provided a natural language response."
                 default_report["llm_issues_identified"].append(f"Raw LLM output (first 500 chars): {llm_result_str[:500]}")
                 return default_report

        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError parsing LLM output for probability validation: {e}. Output: {llm_result_str[:500]}")
            default_report["llm_summary"] = f"Error parsing LLM JSON output: {e}"
            default_report["llm_issues_identified"].append(f"Raw LLM output (first 500 chars): {llm_result_str[:500]}")
            return default_report
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM output for probability validation: {e}. Output: {llm_result_str[:500]}")
            default_report["llm_summary"] = f"Unexpected error parsing LLM output: {e}"
            default_report["llm_issues_identified"].append(f"Raw LLM output (first 500 chars): {llm_result_str[:500]}")
            return default_report


    def create_task(self, context: Dict[str, Any], programmatic_validation_results: Dict[str, Any]) -> Task:
        """Create a task for validating scenario probabilities using LLM, informed by programmatic checks."""
        
        # Prepare the programmatic validation summary for the LLM
        programmatic_summary_str = json.dumps(programmatic_validation_results, indent=2)
        initial_probabilities_str = json.dumps(context.get('scenario_probabilities', {}), indent=2)

        json_output_format_instruction = """
        You MUST provide your output as a single JSON object.
        The JSON object should contain the following fields:
        - "assessment_status": "string (e.g., 'Probabilities Valid', 'Minor Issues Found', 'Significant Revisions Recommended')"
        - "summary": "string (A brief overall assessment of the provided probabilities based on your analysis.)"
        - "issues_identified": ["array of strings (List each specific issue you found, e.g., 'Probabilities do not sum to 1.0.', 'Probability for HD-S-R is too low given the context.')"]
        - "recommendations": ["array of strings (Actionable recommendations for improving the probabilities.)"]
        - "suggested_probabilities": { "object (A dictionary with scenario codes as keys and your suggested float probabilities as values. These MUST sum to 1.0 and include all 8 required scenarios.)" }

        Example for "suggested_probabilities":
        "suggested_probabilities": {
            "HD-S-R": 0.15, "HD-S-O": 0.10, "HD-V-R": 0.12, "HD-V-O": 0.08,
            "LD-S-R": 0.18, "LD-S-O": 0.12, "LD-V-R": 0.15, "LD-V-O": 0.10
        }
        Ensure the entire output is a valid JSON object starting with { and ending with }. Do not include any markdown like ```json.
        """

        return Task(
            description=f"""
            As an Expert Risk Analyst, your task is to critically evaluate the provided scenario probabilities for a strategic decision.
            The business is considering: "{context.get('task', 'A strategic initiative')}".

            Initial Scenario Probabilities provided by the user/system:
            {initial_probabilities_str}

            A preliminary programmatic check was performed, yielding these results:
            {programmatic_summary_str}

            Your Evaluation Criteria:
            1.  **Completeness**: Do probabilities exist for all 8 required scenarios: {', '.join(self.required_scenarios)}?
            2.  **Summation**: Do the probabilities sum to 1.0 (or 100%)?
            3.  **Reasonable Range**: Are individual probabilities plausible (e.g., generally avoid extremes like <1% or >50% without strong justification for one out of 8 scenarios)?
            4.  **Logical Coherence & Distribution**:
                *   Does the distribution across demand levels (High vs. Low) make sense?
                *   Does the distribution across market stability (Stable vs. Volatile) make sense?
                *   Does the distribution across partner behavior (Reliable vs. Opportunistic) make sense?
                *   Are there any apparent biases or inconsistencies in the assignments?
            5.  **Contextual Relevance**: Briefly consider if the probabilities align with the nature of the task: "{context.get('task', 'A strategic initiative')}". For example, a highly uncertain task might warrant higher probabilities for volatile scenarios.

            Your Output:
            Provide a detailed validation report.
            {json_output_format_instruction}

            If the initial probabilities have significant issues (e.g., don't sum to 1.0, missing scenarios), your "suggested_probabilities" MUST correct these basic mathematical issues.
            If the initial probabilities are mathematically sound but you have qualitative concerns, your "suggested_probabilities" can be the same as the input if you believe they are adequate after your review, or you can propose adjustments.
            Your reasoning for any suggested changes should be clear in the "issues_identified" and "recommendations" sections.
            """,
            agent=self.agent,
            expected_output="A single, valid JSON object string containing the probability validation report, including status, summary, issues, recommendations, and suggested probabilities that sum to 1.0."
        )

    def execute(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the probability validation task using both programmatic checks and LLM review."""
        logger.info("ProbabilityCollectorAgent: Starting probability validation.")
        
        initial_probabilities = shared_context.get('scenario_probabilities', {})
        if not isinstance(initial_probabilities, dict) or not initial_probabilities:
            logger.warning("Initial scenario probabilities are missing or not a dict. Assigning empty dict.")
            initial_probabilities = {}
            # Populate with zeros if completely empty to avoid errors in programmatic check, LLM should flag this.
            if not any(initial_probabilities.values()): # if all are zero or empty
                initial_probabilities = {code: 0.0 for code in self.required_scenarios}


        # 1. Perform programmatic validation first
        programmatic_validated_probs, programmatic_report = self._programmatic_probability_validation(initial_probabilities)
        shared_context["programmatic_probability_validation"] = programmatic_report
        # Use programmatically validated (e.g., normalized) probabilities as baseline for LLM if needed
        context_for_llm = shared_context.copy()
        context_for_llm['scenario_probabilities'] = programmatic_validated_probs # Give LLM the normalized ones

        logger.info(f"Programmatic validation report: {programmatic_report}")

        # 2. Get LLM's assessment
        llm_assessment_report = { # Initialize with defaults
             "llm_assessment_status": "Not Run",
             "llm_summary": "LLM assessment was not performed.",
             "llm_issues_identified": [],
             "llm_recommendations": [],
             "llm_suggested_probabilities": programmatic_validated_probs # Fallback
        }

        try:
            task_to_run = self.create_task(context_for_llm, programmatic_report)
            crew = Crew(agents=[self.agent], tasks=[task_to_run], verbose=True)
            logger.info("Kicking off internal crew for LLM probability assessment...")
            llm_result_str = crew.kickoff(inputs=context_for_llm)
            logger.info(f"ProbabilityCollectorAgent: Raw LLM Result:\n{llm_result_str}")
            
            if llm_result_str:
                llm_assessment_report = self._parse_llm_validation_output(llm_result_str, programmatic_validated_probs)
            else:
                logger.warning("LLM returned no result for probability validation.")
                llm_assessment_report["llm_summary"] = "LLM returned no output."


        except Exception as e:
            logger.exception(f"ProbabilityCollectorAgent: Error during LLM crew kickoff or parsing: {e}")
            llm_assessment_report["llm_summary"] = f"Error during LLM processing: {e}"
            # llm_assessment_report['llm_suggested_probabilities'] will retain its fallback
        
        shared_context["llm_probability_assessment"] = llm_assessment_report
        
        # Decide which probabilities to use moving forward: LLM's suggestion or programmatic if LLM failed badly
        final_probabilities = llm_assessment_report.get("llm_suggested_probabilities", programmatic_validated_probs)
        
        # Final check on the chosen probabilities (e.g., sum to 1 again)
        final_sum = sum(v for v in final_probabilities.values() if isinstance(v, (int, float)))
        if abs(final_sum - 1.0) > 0.01 and final_sum > 0:
            logger.warning(f"Final probabilities (from LLM or programmatic) sum to {final_sum}. Normalizing as a last step.")
            final_probabilities = {k: v / final_sum for k,v in final_probabilities.items()}
            shared_context["llm_probability_assessment"]["llm_notes"] = shared_context["llm_probability_assessment"].get("llm_notes",[]) + ["Final probabilities re-normalized post-LLM processing."]
        elif final_sum == 0:
             logger.error("Final probabilities are all zero after LLM processing. This is a critical error.")
             # Potentially revert to user's original input if it was better, or raise an error.
             # For now, we pass them as is, downstream should handle zero probabilities carefully.

        shared_context["scenario_probabilities"] = final_probabilities

        logger.info(f"LLM assessment report: {llm_assessment_report}")
        logger.info(f"Final scenario probabilities after validation: {final_probabilities}")
        return shared_context