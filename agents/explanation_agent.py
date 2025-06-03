import os
import json
import logging
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ExplanationAgent:
    """
    Narrative Generator agent that creates clear, executive-friendly explanations
    of the strategic recommendation and underlying analysis using an LLM.
    """

    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model=os.getenv("EXPLANATION_MODEL", "gpt-4o-mini"),
                temperature=0.7,
                max_tokens=4000,  # Increased max tokens for longer outputs
                request_timeout=120  # Increased timeout for longer generations
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for ExplanationAgent: {e}")
            raise RuntimeError(f"ExplanationAgent: LLM initialization failed. Is OPENAI_API_KEY set? Error: {e}")

        self.agent = Agent(
            role="Chief Strategy Communications Officer",
            goal="Craft a compelling and insightful executive summary that clearly articulates the final strategic recommendation, its rationale, and key implications for executive decision-makers.",
            backstory="""You are a highly respected Chief Strategy Communications Officer renowned for your ability to
            distill complex strategic analyses into clear, concise, and persuasive narratives for C-suite executives
            and boards of directors. You excel at framing decisions, highlighting strategic imperatives, and building
            consensus through impactful communication. Your summaries are not just informative; they drive action.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5  # Increased max iterations for more thorough generation
        )

    def _prepare_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Prepares a comprehensive string representation of the context for the LLM prompt."""
        
        # Basic Context
        basic_context = {
            "Business Question Being Addressed": context.get('task', 'N/A'),
            "Asset Specificity": context.get('asset_specificity', 'N/A'),
            "Transaction Frequency": context.get('frequency', 'N/A'),
            "Overall Market/Technological Uncertainty": context.get('uncertainty', 'N/A'),
            "General Trust in Potential Partners": context.get('partner_trust', 'N/A')
        }

        # Scenario Probabilities
        scenario_probabilities = context.get('scenario_probabilities', {})
        
        # Scenario Descriptions (Summarized for prompt length)
        scenario_descriptions_summary = {}
        raw_scenario_descriptions = context.get('scenario_descriptions', {})
        if isinstance(raw_scenario_descriptions, dict):
            for code, details in raw_scenario_descriptions.items():
                if isinstance(details, dict):
                    scenario_descriptions_summary[code] = {
                        "name": details.get("name", "Unnamed"),
                        "narrative_snippet": (details.get("narrative", "")[:150] + "...") if details.get("narrative") else "N/A"
                    }
                else:
                    scenario_descriptions_summary[code] = "Invalid detail format"

        # Per-Scenario Transaction Logic Results (Summarized)
        per_scenario_analysis_summary = {}
        raw_scenario_results = context.get('scenario_results', {})
        if isinstance(raw_scenario_results, dict):
            for code, data in raw_scenario_results.items():
                if isinstance(data, dict):
                    per_scenario_analysis_summary[code] = {
                        "recommended_strategy": data.get("strategy", "N/A"),
                        "programmatic_confidence": data.get("confidence_programmatic", "N/A"),
                        "llm_reasoning_snippet": (data.get("llm_reasoning", "")[:150] + "...") if data.get("llm_reasoning") else "N/A",
                        "llm_confidence": data.get("llm_confidence", "N/A")
                    }
                else:
                    per_scenario_analysis_summary[code] = "Invalid result format"

        # Aggregation Results
        programmatic_scores = context.get('strategy_scores_programmatic', {})
        llm_aggregation_synthesis = context.get('llm_aggregation_synthesis', {})
        final_recommendation = context.get('final_recommendation', 'N/A')

        # Construct the string - BE MINDFUL OF TOKEN LIMITS FOR THE LLM
        llm_context_parts = [
            "## Key Input Data & Analysis Results for Executive Summary Generation:",
            f"\n### 1. Initial Business Context:\n{json.dumps(basic_context, indent=2)}",
            f"\n### 2. Final Overall Recommendation: {final_recommendation}",
            f"\n### 3. Aggregated Programmatic Strategy Scores (Weighted):\n{json.dumps(programmatic_scores, indent=2)}",
            f"\n### 4. LLM Synthesis from Aggregation Agent:\n{json.dumps(llm_aggregation_synthesis, indent=2)}",
            f"\n### 5. Scenario Probabilities Used:\n{json.dumps(scenario_probabilities, indent=2)}",
            f"\n### 6. Summary of Detailed Scenario Narratives (Generated by ScenarioGeneratorAgent):",
            json.dumps(scenario_descriptions_summary, indent=2, ensure_ascii=False),
            f"\n### 7. Summary of Per-Scenario Strategy Analysis (from TransactionLogicAgent):",
            json.dumps(per_scenario_analysis_summary, indent=2, ensure_ascii=False),
        ]
        
        llm_context_str = "\n\n".join(llm_context_parts)
        
        # Log context length for debugging token issues
        # A rough estimate: 1 token ~ 4 chars.
        logger.info(f"ExplanationAgent: Approximate character count for LLM context: {len(llm_context_str)}")
        if len(llm_context_str) > 60000: # Arbitrary large number, roughly 15k tokens
            logger.warning("ExplanationAgent: LLM context string is very long, potential for truncation or issues.")
            # Consider more aggressive summarization if this happens often.

        return llm_context_str.strip()

    def create_llm_task(self, context_str: str) -> Task:
        """Create a task for generating the executive summary using the LLM."""
        return Task(
            description=f"""
            You are the Chief Strategy Communications Officer. Your task is to craft a comprehensive, insightful, and persuasive
            executive summary based on the following analytical results and context provided. This summary will be presented
            to senior executives to guide a critical strategic decision.

            **Key Input Data & Analysis Results:**
            {context_str}

            **Executive Summary Requirements (Structure and Content):**

            **Title: Strategic Recommendation Report**

            **1. Introduction & Executive Recommendation** (Concise, 2-4 sentences):
                *   Clearly state the business question that was analyzed (from "Initial Business Context").
                *   State the final overall recommended strategy (e.g., MAKE, BUY, JV - from "Final Overall Recommendation").
                *   Mention the qualitative confidence level in this recommendation (from "LLM Synthesis from Aggregation Agent" -> "llm_confidence_qualitative").
                *   Provide a very brief, high-level strategic imperative driving this recommendation.

            **2. Strategic Rationale & Economic Basis** (1-2 insightful paragraphs):
                *   Explain *why* this strategy is the most suitable choice.
                *   Refer to the key overarching transaction cost factors identified (Asset Specificity, Frequency, Uncertainty, Partner Trust - from "Initial Business Context"). Explain how these factors generally influence the choice.
                *   Connect this to the **LLM's overall justification** from the "LLM Synthesis from Aggregation Agent".
                *   Briefly and accessibly allude to underlying economic principles (e.g., minimizing transaction costs, balancing governance costs with production costs, Coase/Williamson) without being overly academic.

            **3. Scenario Analysis & Robustness** (1-2 paragraphs):
                *   Begin by stating that the recommendation was tested against multiple future scenarios.
                *   Refer to the "Summary of Detailed Scenario Narratives." Briefly explain how the *nature* of these scenarios (e.g., some optimistic, some pessimistic, varying market conditions) provided a robust test. Do not list all scenario details, but give a sense of the breadth.
                *   Discuss how the **final recommended strategy performed or is justified across these varied conditions**, drawing from the "Summary of Per-Scenario Strategy Analysis" and the "LLM Synthesis from Aggregation Agent" (especially the overall justification).
                *   Mention that scenario probabilities were used for weighted scoring (refer to "Aggregated Programmatic Strategy Scores").
                *   Summarize the overall risk profile discussed in the "LLM Synthesis from Aggregation Agent" (e.g., related to risk tolerance).

            **4. Detailed Discussion of Recommended Strategy: {{{{FINAL_RECOMMENDATION_PLACEHOLDER}}}}**
                (The LLM should replace {{{{FINAL_RECOMMENDATION_PLACEHOLDER}}}} with the actual recommended strategy, e.g., "Detailed Discussion of MAKE Strategy")
                *   **Justification Deep Dive:** Elaborate further on why this specific strategy (MAKE, BUY, or JV) is superior, using insights from the "LLM Synthesis from Aggregation Agent" (overall_justification, strategic_control_consideration, resource_constraints_consideration, time_horizon_consideration).
                *   **Key Success Factors:** Identify 2-3 critical success factors for implementing *this specific* recommended strategy.
                *   **Primary Risks & Mitigations:** Highlight 2-3 primary risks associated with *this specific* strategy and suggest high-level mitigation approaches for each.

            **5. Alternative Strategies Considered** (1 paragraph or bullet points for each alternative):
                *   For each of the other two primary strategies (e.g., if MAKE was chosen, discuss BUY and JV):
                    *   Briefly state why it was considered less optimal, referring to its aggregated score (from "Aggregated Programmatic Strategy Scores") and any qualitative reasons from the `AggregationAgent`'s synthesis or the `TransactionLogicAgent`'s per-scenario analysis if relevant.
                    *   Optionally, mention specific conditions or scenarios under which this alternative might become more attractive or warrant reconsideration.

            **6. Conclusion & Next Steps** (1 brief paragraph):
                *   Reiterate the recommendation with conviction.
                *   Suggest key next steps for the executive team (e.g., detailed implementation planning, resource allocation workshops, further due diligence if BUY/JV).

            **Tone and Style Guidelines:**
            - Language: Clear, concise, professional, confident, and suitable for a C-suite audience.
            - Voice: Strategic, authoritative, and action-oriented.
            - Perspective: Balanced – confidently present the recommendation while acknowledging relevant complexities, uncertainties, and risks.
            - Focus: High-level strategic implications and justifications. Avoid excessive operational detail or jargon.
            - Length: Aim for a summary that is comprehensive yet digestible, typically 1-2 pages if printed (approx. 600-1000 words).

            **Output Format:**
            Provide the complete executive summary as a single block of well-formatted Markdown text.
            Use Markdown headings (e.g., `## 1. Introduction & Executive Recommendation`) for each section as outlined above.
            Ensure lists are properly formatted with bullets.
            Do not include any conversational preamble or postamble like "Here is the summary:".
            Do not wrap the output in a markdown code block (do not use ```markdown or ```).
            Just provide the raw markdown content.
            Replace `{{{{FINAL_RECOMMENDATION_PLACEHOLDER}}}}` in the heading for section 4 with the actual recommended strategy.
            """,
            agent=self.agent,
            expected_output="A complete, well-structured, and persuasive executive summary in Markdown format, adhering to all specified requirements regarding content, structure, tone, and style."
        )

    def execute(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the explanation generation task using the LLM."""
        logger.info("ExplanationAgent: Starting executive summary generation.")

        context_str_for_llm = self._prepare_context_for_llm(shared_context)
        
        # Log the context being sent to the LLM for debugging
        logger.debug(f"Context being sent to ExplanationAgent LLM:\n{context_str_for_llm}")

        llm_task = self.create_llm_task(context_str_for_llm)
        crew = Crew(agents=[self.agent], tasks=[llm_task], verbose=True)

        executive_summary = "Error: LLM execution failed to produce a summary."
        try:
            logger.info("Kicking off internal crew for LLM executive summary generation...")
            result = crew.kickoff()
            
            # Convert CrewOutput to string if needed
            if hasattr(result, 'raw_output'):
                result = result.raw_output
            elif hasattr(result, 'output'):
                result = result.output
            elif isinstance(result, str):
                result = result
            else:
                result = str(result)
            
            # Clean up the result
            result = result.strip()
            
            # Check if the result is a complete executive summary
            if result and len(result) > 500:  # Increased minimum length for a complete summary
                executive_summary = result
                logger.info("ExplanationAgent: Successfully generated executive summary from LLM.")
            elif result:
                logger.warning(f"ExplanationAgent: LLM produced a potentially incomplete result. Length: {len(result)} chars")
                executive_summary = f"Warning: LLM produced a brief or incomplete summary.\n\nRaw output: {result}"
            else:
                logger.error("ExplanationAgent: LLM returned no result or an empty string.")
                executive_summary = "Error: LLM returned no output for the executive summary."

        except Exception as e:
            logger.exception(f"ExplanationAgent: Error during LLM crew kickoff for summary generation: {e}")
            executive_summary = f"Error generating executive summary due to an exception: {e}"

        shared_context["executive_summary"] = executive_summary
        return shared_context