import random
import json
import os
import logging
import time
import re
import math  # Added for SQ analysis calculations
import copy  # Added for noise simulation
from typing import List, Optional, Dict, Tuple

# Assume BasePrompting and ImagePrompt are defined in 'base_prompting.py'
# Make sure to have SentenceTransformer installed: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers library not found. Semantic similarity will use a fallback value.")
    print("Install it using: pip install sentence-transformers")

from base_prompting import BasePrompting, ImagePrompt

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set to logging.DEBUG for verbose output, logging.INFO for less
logger.setLevel(logging.INFO)
# --------------------


class EoTPrompting(BasePrompting):
    """
    Implements Evolutionary Chain-of-Thought (EoT) prompting.

    This class uses an evolutionary algorithm to optimize Chain-of-Thought prompts
    for Vision-Language Models based on performance on specific tasks. It aligns
    with the concept of Darwinian Evolution as Statistical Query Learning (CS 228 Notes, Sec 15.2):
    - Organisms/DNA: The CoT prompts represent strategies (d).
    - Environment: The dataset (e.g., JUS) provides selective pressures (distribution D).
    - Fitness: The calculated score acts as a statistical query estimating the prompt's
               success rate P(chi=1) where chi represents meeting performance criteria
               against potentially noisy labels (Sec 14.1, 15.1). Fitness is averaged
               over trials to handle noise/stochasticity.
    - Evolution: Selection (elitism), crossover, and mutation iteratively adapt prompts.
    """

    def __init__(self,
                 population_size: int = 20,
                 generations: int = 3,
                 elite_size: int = 4,
                 fitness_trials: int = 1,  # Number of times to run LLM query for fitness stability
                 noise_eta: float = 0.0):  # Probability of random classification noise during fitness evaluation
        super().__init__()

        # Override results directory
        self.results_dir = "results/cot_evolutionary_results_noisy"  # Changed dir name
        os.makedirs(self.results_dir, exist_ok=True)

        # Evolution-specific parameters
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.fitness_trials = fitness_trials  # Store averaging parameter
        self.noise_eta = noise_eta  # Store noise parameter

        # Define weights for fitness metrics
        self.weights = {
            'semantic_similarity': 0.7,
            'numeric_accuracy': 0.3,
            # Add more weights if needed (e.g., response_quality)
        }

        # Base prompts remain the same
        self.base_prompts = {
            'counting': [
                "Analyze this image systematically:\n"
                "1. Divide the image into clear sections\n"
                "2. Count items in each section methodically\n"
                "3. Double-check each section\n"
                "4. If counting is impossible, explain why\n"
                "5. State your confidence level and any uncertainties\n"
                "Provide final count or detailed explanation if counting isn't feasible.",

                "Precise counting process:\n"
                "1. Assess image clarity and item visibility\n"
                "2. Use grid method: count left-to-right, top-to-bottom\n"
                "3. Note any partially visible or obscured items\n"
                "4. Verify count by counting in reverse\n"
                "5. If counting is not possible, explain specific reasons\n"
                "State final count or explain why counting cannot be done accurately."
            ],
            'descriptive': [
                "Analyze this image step by step:\n"
                "1. Describe what you can clearly see\n"
                "2. Note any ambiguous or unclear elements\n"
                "3. Address the specific question asked\n"
                "4. Provide confidence level in your answer\n"
                "Be specific and avoid assumptions.",

                "Systematic image analysis:\n"
                "1. Focus on relevant elements for the question\n"
                "2. Describe visible details methodically\n"
                "3. Address any limitations in visibility\n"
                "4. Answer the question with available evidence\n"
                "State confidence level and reasoning."
            ]
        }

        # Initialize sentence transformer model only once if available
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self._similarity_model = SentenceTransformer(
                    'all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                self._similarity_model = None
        else:
            self._similarity_model = None

    def crossover(self, parent1: str, parent2: str) -> str:
        """Improved crossover with semantic understanding"""
        parts1 = parent1.split('\n')
        parts2 = parent2.split('\n')

        # Basic instruction/steps separation (can be improved)
        instructions1 = [p for p in parts1 if not re.match(
            r"^\s*\d+\.", p.strip())]
        instructions2 = [p for p in parts2 if not re.match(
            r"^\s*\d+\.", p.strip())]
        steps1 = [p for p in parts1 if re.match(r"^\s*\d+\.", p.strip())]
        steps2 = [p for p in parts2 if re.match(r"^\s*\d+\.", p.strip())]

        # Ensure instructions/steps aren't empty before combining
        child_instructions = instructions1 if random.random() < 0.5 else instructions2
        child_steps = steps2 if child_instructions == instructions1 else steps1

        # Handle cases where one parent might lack instructions or steps
        if not child_instructions:
            child_instructions = instructions1 if instructions1 else instructions2
        if not child_steps:
            child_steps = steps1 if steps1 else steps2

        # Combine and filter empty lines
        result = [line for line in (
            child_instructions + child_steps) if line.strip()]

        return '\n'.join(result)

    def deterministic_mutate(self, prompt: str, mutation_rate: float = 0.3) -> str:
        """Enhanced mutation with semantic preservation"""
        lines = prompt.split('\n')
        mutated_lines = []

        emphasis_words = ['carefully', 'precisely', 'systematically',
                          'thoroughly', 'methodically', 'closely', 'in detail']
        counting_words = ['count', 'tally', 'enumerate', 'quantify',
                          'calculate', 'sum up', 'determine the number of']
        verification_words = ['verify', 'check', 'confirm',
                              'validate', 'ensure accuracy of', 'double-check']
        uncertainty_words = ['estimate', 'approximate',
                             'consider', 'note', 'address', 'report']

        for line in lines:
            original_line = line
            if random.random() < mutation_rate:
                applied_mutation = False
                # Add emphasis prefix
                if random.random() < 0.2:
                    line = random.choice(emphasis_words) + ' ' + line
                    applied_mutation = True

                # Replace counting words
                for cw in counting_words:
                    if cw in line.lower() and random.random() < 0.4:
                        # Use regex to replace whole word only to avoid partial matches like 'discount'
                        line = re.sub(r'\b' + re.escape(cw) + r'\b',
                                      random.choice(counting_words), line, flags=re.IGNORECASE)
                        applied_mutation = True
                        break  # Only one replacement per line

                # Add verification suffix
                if random.random() < 0.15:
                    line += f", then {random.choice(verification_words)} the result."
                    applied_mutation = True

                # Add uncertainty handling words
                if random.random() < 0.1:
                    line = random.choice(uncertainty_words) + ' ' + line
                    applied_mutation = True

                # Simple word swap (less semantic)
                if not applied_mutation and random.random() < 0.1:
                    words = line.split()
                    if len(words) > 1:
                        idx1, idx2 = random.sample(range(len(words)), 2)
                        words[idx1], words[idx2] = words[idx2], words[idx1]
                        line = ' '.join(words)

            mutated_lines.append(line)

        return '\n'.join(filter(None, mutated_lines))  # Filter out empty lines

    def intelligent_mutate(self, prompt: str, previous_response: str, metrics: dict, mutation_rate: float = 0.3) -> str:
        """
        Perform mutation guided by LLM analysis of past performance.
        This represents a form of directed evolution, using feedback (analysis)
        to guide variations rather than purely random changes. Falls back to
        deterministic_mutate if analysis fails or provides no actionable insights.
        """
        if random.random() > mutation_rate:  # Apply mutation only some of the time
            return prompt

        try:
            logger.debug("Attempting intelligent mutation...")
            # Format analysis prompt carefully
            analysis_prompt = f"""Analyze this prompt and its performance:

Original Prompt:
---
{prompt}
---

Response Generated:
---
{previous_response}
---

Performance Metrics (0-1 scale):
- Semantic Similarity: {metrics.get('semantic_similarity', 0):.3f}
- Numeric Accuracy: {metrics.get('numeric_accuracy', 0):.3f}

Identify potential issues with the prompt related to the generated response and metrics:
1. Counting Precision: Did it lead to missed items, double counting, or vague counts?
2. Verification: Does the prompt encourage or result in verification/double-checking?
3. Clarity/Structure: Are the steps clear? Is the reasoning easy to follow?
4. Uncertainty Handling: Does the prompt ask for or the response include confidence/uncertainty?

Provide specific, actionable suggestions for improving the *prompt* based ONLY on the issues observed in the response and metrics.

Give your analysis in this exact JSON format (use double quotes, ensure valid JSON):
{{
    "issues_identified": ["Brief description of issue 1 (e.g., 'Missed items in count')", "issue 2"],
    "suggested_prompt_improvements": ["Specific change 1 (e.g., 'Add step: Scan image systematically')", "change 2"],
    "key_areas_for_emphasis": ["Area 1 (e.g., 'Verification')", "Area 2 (e.g., 'Confidence Level')"]
}}
"""

            # Query LLM for analysis (use text-only model for efficiency)
            analysis_response = self.llm_query(
                analysis_prompt, image_path=None, model_override="gpt-4o-mini")  # Use text-only

            # Attempt to parse JSON robustly
            try:
                logger.debug(f"Raw analysis response: {analysis_response}")
                # Find JSON block even if surrounded by text/markdown
                match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
                if not match:
                    raise ValueError(
                        "No JSON object found in the analysis response.")
                json_str = match.group(0)
                analysis = json.loads(json_str)
                logger.debug(f"Parsed analysis JSON: {analysis}")

                # Validate required keys
                required_keys = {
                    'issues_identified', 'suggested_prompt_improvements', 'key_areas_for_emphasis'}
                if not all(key in analysis for key in required_keys):
                    raise ValueError(
                        f"Analysis JSON missing required keys. Found: {analysis.keys()}")

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(
                    f"Intelligent mutation analysis parsing failed: {e}. Falling back to deterministic mutation.")
                return self.deterministic_mutate(prompt, mutation_rate)

            # Apply mutations based on analysis - Simple strategy: Append suggestions
            mutated_prompt = prompt  # Start with original
            applied_intelligent_mutation = False

            if analysis.get("suggested_prompt_improvements"):
                mutated_prompt += "\n\n# Suggested Improvements Based on Analysis:\n"
                for improvement in analysis["suggested_prompt_improvements"]:
                    # Basic check to avoid adding duplicate lines
                    if improvement not in mutated_prompt:
                        mutated_prompt += f"- {improvement}\n"
                        applied_intelligent_mutation = True

            # If analysis didn't yield actionable suggestions, fall back
            if not applied_intelligent_mutation:
                logger.debug(
                    "No actionable improvements from LLM analysis. Using deterministic mutation.")
                return self.deterministic_mutate(prompt, mutation_rate)

            logger.debug("Applied intelligent mutation based on LLM analysis.")
            return mutated_prompt.strip()

        except Exception as e:
            logger.error(
                f"Intelligent mutation process failed: {e}", exc_info=True)
            # Fallback to deterministic mutation if anything goes wrong
            return self.deterministic_mutate(prompt, mutation_rate)

    def apply_noise(self, image_prompt: ImagePrompt) -> ImagePrompt:
        """
        Simulates Random Classification Noise (CS 228 Notes Sec 14.1).
        With probability eta, modifies the ground truth label used for fitness.
        """
        if random.random() < self.noise_eta:
            noisy_prompt = copy.deepcopy(image_prompt)  # Don't modify original
            logger.debug(
                f"Applying noise (eta={self.noise_eta}) to ground truth for {image_prompt.filename}")

            if noisy_prompt.counting and noisy_prompt.numeric_answer is not None:
                # Simple numeric noise: Add/subtract small random amount
                perturbation = random.uniform(-2, 2)  # Example: +/- 2
                noisy_prompt.numeric_answer = max(
                    0, round(noisy_prompt.numeric_answer + perturbation))
                # Also modify text answer to reflect noise if possible (simplistic)
                # Simplistic update
                noisy_prompt.text_answer = f"Approximately {noisy_prompt.numeric_answer}"
                logger.debug(
                    f"Noisy numeric answer: {noisy_prompt.numeric_answer}")
            else:
                # Simple descriptive noise: Flip sentiment or replace with generic wrong answer
                if "yes" in noisy_prompt.text_answer.lower():
                    noisy_prompt.text_answer = "No, that is incorrect."
                elif "no" in noisy_prompt.text_answer.lower():
                    noisy_prompt.text_answer = "Yes, that is correct."
                else:
                    # Replace with a generic incorrect statement
                    noisy_prompt.text_answer = "The analysis based on the noisy label is different."
                logger.debug(f"Noisy text answer: {noisy_prompt.text_answer}")
            return noisy_prompt
        else:
            # No noise applied
            return image_prompt

    def fitness(self, prompt: str, image_prompt: ImagePrompt, image_path: str) -> Tuple[float, Dict]:
        """
        Calculate fitness score for a prompt, averaged over trials and potentially using noisy labels.

        Acts like evaluating a criterion 'chi' (performance threshold) for the prompt (hypothesis)
        on the image_prompt (example). The score estimates P(chi=1). Handles LLM noise via averaging.
        Can simulate label noise via self.noise_eta (CS 228 Notes Sec 14.1, 15.1).
        """
        if self.fitness_trials <= 0:
            self.fitness_trials = 1

        all_scores = []
        all_metrics_list = []
        accumulated_numeric_accuracy = 0.0
        accumulated_semantic_similarity = 0.0
        numeric_count = 0
        semantic_count = 0

        logger.debug(
            f"Calculating fitness over {self.fitness_trials} trials for prompt: {prompt[:50]}...")

        # Determine ground truth (potentially noisy) ONCE for all trials of this fitness call
        eval_image_prompt = self.apply_noise(image_prompt)

        for trial in range(self.fitness_trials):
            logger.debug(f"Fitness trial {trial + 1}/{self.fitness_trials}")
            trial_score = 0.0
            trial_metrics = {
                'semantic_similarity': 0.0,
                'numeric_accuracy': 0.0,
                # Add more detailed metrics if needed
                'response_length': 0,
                'used_noisy_label': eval_image_prompt != image_prompt  # Track if noise was applied
            }

            try:
                # Get LLM response
                response = self.llm_query(prompt, image_path)
                trial_metrics['response_length'] = len(response.split())

                if response and not response.startswith("Error:"):
                    # --- Calculate Semantic Similarity ---
                    sim_score = 0.0
                    if self._similarity_model:
                        try:
                            response_embedding = self._similarity_model.encode(
                                response, convert_to_tensor=True)
                            answer_embedding = self._similarity_model.encode(
                                eval_image_prompt.text_answer, convert_to_tensor=True)
                            similarity = float(util.pytorch_cos_sim(
                                response_embedding, answer_embedding)[0][0])
                            # Normalize cosine similarity (-1 to 1) -> (0 to 1)
                            sim_score = max(
                                0.0, min(1.0, (similarity + 1) / 2))
                            logger.debug(
                                f"Trial {trial+1} Semantic Similarity: {sim_score:.3f}")
                        except Exception as e:
                            logger.warning(
                                f"Similarity calculation error in trial {trial+1}: {e}")
                            sim_score = 0.0  # Penalize if error occurs
                    else:
                        # Fallback if model not loaded
                        sim_score = 0.3 if len(
                            response) > 10 else 0.0  # Simple fallback

                    trial_metrics['semantic_similarity'] = sim_score
                    accumulated_semantic_similarity += sim_score
                    semantic_count += 1

                    # --- Calculate Numeric Accuracy (only for counting tasks) ---
                    num_acc_score = 0.0
                    if eval_image_prompt.counting:
                        num_acc_score = self.calculate_numeric_accuracy(
                            response, eval_image_prompt.numeric_answer, eval_image_prompt.text_answer)
                        trial_metrics['numeric_accuracy'] = num_acc_score
                        accumulated_numeric_accuracy += num_acc_score
                        numeric_count += 1
                        logger.debug(
                            f"Trial {trial+1} Numeric Accuracy: {num_acc_score:.3f}")

                    # --- Calculate Weighted Score for this trial ---
                    trial_score = self.weights['semantic_similarity'] * sim_score
                    if eval_image_prompt.counting:
                        # Only add numeric accuracy weight if it's a counting problem
                        trial_score += self.weights['numeric_accuracy'] * \
                            num_acc_score
                    else:
                        # If not counting, re-normalize semantic similarity weight potentially
                        # Or just use semantic similarity score directly if it's the only metric
                        pass  # Current logic just uses the weighted semantic score

                    all_scores.append(trial_score)
                    all_metrics_list.append(trial_metrics)

                else:
                    # Penalize errors or empty responses
                    logger.warning(
                        f"Invalid response in trial {trial+1}: {response}")
                    all_scores.append(0.0)
                    # Append metrics even for failed trials
                    all_metrics_list.append(trial_metrics)

            except Exception as e:
                logger.error(
                    f"Error during fitness trial {trial + 1}: {e}", exc_info=True)
                all_scores.append(0.0)  # Penalize trial if error occurs
                all_metrics_list.append(trial_metrics)

        # --- Averaging ---
        final_avg_score = sum(all_scores) / \
            len(all_scores) if all_scores else 0.0

        # Aggregate metrics (simple average for now)
        final_metrics = {
            'semantic_similarity': accumulated_semantic_similarity / semantic_count if semantic_count > 0 else 0.0,
            'numeric_accuracy': accumulated_numeric_accuracy / numeric_count if numeric_count > 0 else 0.0,
            'num_trials': self.fitness_trials,
            'final_score': final_avg_score,
            'noise_eta_used': self.noise_eta,
            'used_noisy_label_in_eval': eval_image_prompt != image_prompt,
            # Include raw trial scores/metrics if needed for analysis
            # 'trial_scores': all_scores,
            # 'trial_metrics': all_metrics_list
        }

        logger.debug(
            f"Fitness calculation complete. Avg Score: {final_avg_score:.4f}")
        return final_avg_score, final_metrics

    def calculate_numeric_accuracy(self, rationale: str, target: Optional[float], target_text: str) -> float:
        """Enhanced numeric accuracy calculation, handling 'impossible' cases and tolerance."""
        impossible_phrases = ['impossible to count', 'cannot count', 'unable to count',
                              'not possible to count', 'cannot provide exact count', 'no clear items', 'no items found']
        # Check rationale first
        if any(phrase in rationale.lower() for phrase in impossible_phrases):
            # Check if target also indicates impossibility
            if target is None or any(phrase in target_text.lower() for phrase in ['unknown', 'impossible', 'cannot', 'n/a', 'none']):
                logger.debug(
                    "Numeric Accuracy: Correctly identified impossibility.")
                return 1.0  # Correctly identified impossibility
            else:
                logger.debug(
                    "Numeric Accuracy: Incorrectly claimed impossibility.")
                return 0.0  # Incorrectly claimed impossibility

        # Extract numbers robustly (handle floats, commas)
        try:
            # Remove commas before searching
            rationale_no_commas = rationale.replace(',', '')
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', rationale_no_commas)
            if not numbers:
                logger.debug(
                    "Numeric Accuracy: No numbers found in rationale.")
                # If target is 0 or None, maybe this is correct?
                return 1.0 if target == 0 or target is None else 0.0

            # Try to find the *intended* final answer (often last number, or explicitly stated)
            # Look for phrases like "total is", "final count:", etc.
            predicted_num_str = numbers[-1]  # Default to last number
            match_total = re.search(
                r'(?:total|final count|answer is|count is)\s*[:\-]*\s*(\d+(?:\.\d+)?)', rationale_no_commas, re.IGNORECASE)
            if match_total:
                predicted_num_str = match_total.group(1)

            predicted = float(predicted_num_str)
            logger.debug(
                f"Numeric Accuracy: Found predicted number {predicted}")

        except ValueError:
            logger.warning(
                "Numeric Accuracy: Could not parse number from rationale.")
            return 0.0  # Cannot parse predicted number

        if target is None:
            # Predicted a number, but target is None/impossible
            logger.debug(
                "Numeric Accuracy: Predicted number, but target is None/impossible.")
            return 0.0

        # Calculate accuracy with tolerance
        # Allow a small absolute tolerance plus a relative tolerance
        # 10% or at least 1 unit absolute
        tolerance = max(0.1 * target, 1.0) if target > 0 else 1.0
        error = abs(predicted - target)

        if error <= tolerance:
            logger.debug(
                f"Numeric Accuracy: Prediction {predicted} within tolerance {tolerance} of target {target}.")
            return 1.0
        else:
            # Penalize based on how far outside tolerance
            excess_error = error - tolerance
            penalty = excess_error / target if target > 0 else excess_error
            accuracy = max(0.0, 1.0 - penalty)
            logger.debug(
                f"Numeric Accuracy: Prediction {predicted} outside tolerance {tolerance} of target {target}. Error={error:.2f}, Accuracy={accuracy:.3f}")
            return accuracy

    def evolve_population(self, image_prompt: ImagePrompt) -> List[Tuple[float, str, str, Dict]]:
        """
        Evolve population of prompts for a single image_prompt.
        Returns a list of tuples: (score, prompt, reasoning, metrics) for the elite, sorted by score.
        """
        try:
            logger.info(
                f"\n--- Starting Evolution for {image_prompt.filename} ---")
            image_path = os.path.join('images', image_prompt.filename)
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return []

            # --- Initialize population (Initial gene pool) ---
            population = []
            prompt_type = 'counting' if image_prompt.counting else 'descriptive'
            population.extend(self.base_prompts[prompt_type])
            population.append(image_prompt.prompt)  # Add original prompt
            # Add simple variations
            population.extend([
                f"{image_prompt.prompt}\nPlease provide a detailed step-by-step reasoning for your answer.",
                f"{image_prompt.prompt}\nAnalyze the image carefully and explain your reasoning step-by-step."
            ])
            # Ensure minimum population size by adding mutated base prompts if needed
            # Ensure some diversity if base prompts are few
            while len(population) < self.population_size // 2:
                population.append(self.deterministic_mutate(
                    random.choice(self.base_prompts[prompt_type])))

            # Fill remaining population with mutations of existing ones
            initial_prompts = list(population)  # Keep track of initial set
            while len(population) < self.population_size:
                population.append(self.deterministic_mutate(
                    random.choice(initial_prompts)))

            population = population[:self.population_size]  # Ensure exact size
            logger.info(f"Initial population size: {len(population)}")

            # Store best result found across all generations for this image_prompt
            best_result_for_image = None

            # --- Evolution loop (Generations) ---
            for generation in range(self.generations):
                gen_start_time = time.time()
                logger.info(
                    f"\n>>> Generation {generation + 1}/{self.generations} for {image_prompt.filename} <<<")

                # --- Evaluate current population (Calculate fitness via SQ-like estimation) ---
                # List of (score, prompt, reasoning, metrics)
                generation_results = []
                for i, prompt in enumerate(population):
                    eval_start_time = time.time()
                    logger.debug(f"Evaluating prompt {i+1}/{len(population)}")
                    try:
                        # Fitness function handles LLM query, averaging, and noise
                        score, metrics = self.fitness(
                            prompt, image_prompt, image_path)

                        # Reasoning is needed for intelligent mutation, get it from one trial
                        # Re-querying here is inefficient, TODO: modify fitness to return one reasoning sample
                        # Get reasoning for analysis
                        reasoning = self.llm_query(prompt, image_path)

                        generation_results.append(
                            (score, prompt, reasoning, metrics))
                        logger.debug(
                            f"Prompt {i+1} evaluated. Score: {score:.4f}. Time: {time.time() - eval_start_time:.2f}s")

                    except Exception as e:
                        logger.error(
                            f"Error evaluating prompt {i+1}: {e}", exc_info=True)
                        # Append a failing result to keep list lengths consistent if needed downstream
                        generation_results.append(
                            (-1.0, prompt, f"Error: {e}", {}))
                        continue  # Skip to next prompt

                # Filter out evaluation errors (score < 0) before proceeding
                valid_generation_results = [
                    res for res in generation_results if res[0] >= 0]
                logger.info(
                    f"Generation {generation + 1}: Evaluated {len(population)} prompts, {len(valid_generation_results)} valid results.")

                if not valid_generation_results:
                    logger.warning(
                        f"No valid results in generation {generation + 1}. Ending evolution for this image early.")
                    break  # Stop if all evaluations failed

                # --- Sort by score (Survival of the fittest) ---
                valid_generation_results.sort(reverse=True, key=lambda x: x[0])
                current_best_score = valid_generation_results[0][0]
                logger.info(
                    f"Generation {generation + 1}: Best score = {current_best_score:.4f}")

                # Update overall best result for this image_prompt
                if best_result_for_image is None or current_best_score > best_result_for_image[0]:
                    best_result_for_image = valid_generation_results[0]
                    logger.info(
                        f"New best score for {image_prompt.filename} found: {best_result_for_image[0]:.4f}")

                # --- Selection (Elitism) ---
                # Note: Selection uses relative ranking, not a fixed tolerance 'tau'.
                elite = valid_generation_results[:self.elite_size]
                next_population = [result[1]
                                   for result in elite]  # Keep elite prompts

                # --- Reproduction with variation (Crossover and Mutation) ---
                while len(next_population) < self.population_size:
                    try:
                        if len(elite) >= 2:
                            # Select parents from elite pool
                            parent1_res = random.choice(elite)
                            parent2_res = random.choice(elite)
                            parent1_prompt = parent1_res[1]
                            parent2_prompt = parent2_res[1]

                            # Crossover
                            child = self.crossover(
                                parent1_prompt, parent2_prompt)

                            # Mutation (Potentially intelligent using one parent's context)
                            # Using parent1's reasoning/metrics for intelligent mutation context
                            parent1_reasoning = parent1_res[2]
                            parent1_metrics = parent1_res[3]
                            # Decide mutation type (e.g., 50% chance intelligent if possible)
                            if random.random() < 0.5:
                                child = self.intelligent_mutate(
                                    child, parent1_reasoning, parent1_metrics)
                            else:
                                child = self.deterministic_mutate(child)

                            next_population.append(child)

                        elif elite:  # Only one elite member, mutate it
                            child = self.deterministic_mutate(elite[0][1])
                            next_population.append(child)
                        else:  # Should not happen if valid_generation_results is not empty
                            logger.error(
                                "Error in evolution: No elite members found despite valid results.")
                            break  # Avoid infinite loop

                    except Exception as e:
                        logger.error(
                            f"Error during crossover/mutation: {e}", exc_info=True)
                        # Add a randomly mutated base prompt to prevent population collapse
                        next_population.append(self.deterministic_mutate(
                            random.choice(self.base_prompts[prompt_type])))
                        # Ensure loop terminates eventually
                        if len(next_population) >= self.population_size * 2:
                            break

                # Assign next generation
                population = next_population[:self.population_size]
                logger.info(
                    f"Generation {generation + 1} duration: {time.time() - gen_start_time:.2f}s")
                # --- End Generation Loop ---

            logger.info(
                f"--- Evolution finished for {image_prompt.filename} ---")
            # Return the best result found *during this evolution run*
            return [best_result_for_image] if best_result_for_image else []

        except Exception as e:
            logger.error(
                f"Evolution process failed entirely for {image_prompt.filename}: {e}", exc_info=True)
            return []

    # llm_query remains largely the same, added model_override
    def llm_query(self, prompt: str, image_path: str = None, model_override: Optional[str] = None) -> str:
        """Query the LLM with the given prompt and optional image."""
        query_start_time = time.time()
        target_model = model_override if model_override else self.model
        logger.debug(f"Starting llm_query with model: {target_model}")
        logger.debug(f"Image path: {image_path}")
        logger.debug(f"Prompt preview: {prompt[:100]}...")

        try:
            if image_path is None or not os.path.exists(image_path):
                if image_path is not None:  # Log if path provided but invalid
                    logger.warning(
                        f"Image path invalid or not provided for text-only query: {image_path}")
                # Text-only query
                logger.debug("Performing text-only query.")
                response = self.client.chat.completions.create(
                    model=target_model,  # Use potentially overridden model
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,  # Consider making this configurable
                    temperature=0.5  # Adjust creativity/determinism
                )
                result = response.choices[0].message.content.strip()

            else:
                # Image query
                logger.debug("Encoding image...")
                base64_image = self.encode_image(image_path)
                if not base64_image:
                    logger.error("Image encoding failed.")
                    return "Error: Image encoding failed"

                logger.debug("Creating chat completion with image...")
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                },
                            ],
                        }
                    ],
                    max_tokens=800,
                    temperature=0.5
                )
                result = response.choices[0].message.content.strip()

            logger.debug(
                f"LLM Query successful. Duration: {time.time() - query_start_time:.2f}s")
            logger.debug(f"Response preview: {result[:100]}...")
            return result

        except Exception as e:
            logger.error(
                f"llm_query failed: {type(e).__name__}: {e}", exc_info=True)
            # Provide a more specific error if possible (e.g., API errors)
            return f"Error: LLM query failed - {type(e).__name__}"

    def generate_reasoning(self, image_prompt: ImagePrompt) -> Tuple[str, float, List[str], Dict]:
        """Generate reasoning using evolved prompts for a single image."""
        logger.debug(
            f"Starting generate_reasoning for {image_prompt.filename}")

        image_path = os.path.join('images', image_prompt.filename)
        if not os.path.exists(image_path):
            logger.error(f"Image not found for reasoning: {image_path}")
            return "Error: Image not found", 0.0, [], {}

        try:
            # evolve_population returns the best result from the run
            best_results = self.evolve_population(image_prompt)

            if not best_results:
                logger.error(
                    f"Evolution produced no valid results for {image_prompt.filename}")
                return "Error: Evolution failed to produce results", 0.0, [], {}

            # Extract details from the best result
            best_score, best_prompt, best_reasoning, best_metrics = best_results[0]
            logger.debug(
                f"Best result score: {best_score:.4f} for {image_prompt.filename}")

            return best_reasoning, best_score, [best_prompt], best_metrics

        except Exception as e:
            logger.error(
                f"generate_reasoning failed for {image_prompt.filename}: {e}", exc_info=True)
            return f"Error: {str(e)}", 0.0, [], {}

    def analyze_prompt_with_sq(self,
                               best_prompt: str,
                               criterion_threshold: float = 0.7,  # Example threshold for chi=1
                               tolerance: float = 0.05,  # tau
                               confidence: float = 0.95,  # 1 - delta
                               sq_fitness_trials: int = 1):  # Trials for SQ analysis fitness calls
        """
        Analyzes the final best prompt using simulated Statistical Queries
        over the entire dataset (distribution D). Estimates the probability
        that the prompt meets a performance criterion (fitness > threshold).
        Uses Hoeffding inequality to determine required sample size based on
        tolerance (tau) and confidence (1 - delta). (See CS 228 Notes Sec 14.2)

        Args:
            best_prompt: The prompt string to analyze.
            criterion_threshold: The fitness score threshold defining success (chi=1).
            tolerance: The desired additive error for the probability estimate (tau).
            confidence: The desired confidence level (1 - delta).
            sq_fitness_trials: Number of trials for fitness evaluation during SQ analysis.
        """
        logger.info(f"\n--- Starting SQ Analysis for Best Prompt ---")
        logger.info(f"Criterion (chi): Fitness Score > {criterion_threshold}")
        logger.info(
            f"Tolerance (tau): {tolerance}, Confidence (1-delta): {confidence:.3f} (delta={1.0-confidence:.3f})")
        logger.info(f"Fitness Trials per Sample: {sq_fitness_trials}")

        if not best_prompt:
            logger.error("SQ Analysis Error: No best prompt provided.")
            return None

        # Determine sample size needed based on Hoeffding's inequality
        # m >= (1 / (2 * tau^2)) * ln(2 / delta)
        delta_sq = 1.0 - confidence
        if tolerance <= 0 or delta_sq <= 0 or delta_sq >= 1:
            logger.error(
                f"Invalid SQ parameters: tau={tolerance}, delta={delta_sq}")
            return None
        try:
            num_samples_needed = math.ceil(
                (1 / (2 * tolerance**2)) * math.log(2 / delta_sq))
        except ValueError as e:
            logger.error(f"SQ sample size calculation error: {e}")
            return None

        logger.info(
            f"Required samples (m) for SQ estimate: {num_samples_needed}")

        dataset_size = len(self.image_prompts)
        if dataset_size == 0:
            logger.error(
                "SQ Analysis Error: Dataset (self.image_prompts) is empty.")
            return None

        if num_samples_needed > dataset_size:
            logger.warning(
                f"Required samples ({num_samples_needed}) exceed dataset size ({dataset_size}). Using full dataset for estimation.")
            samples_to_evaluate_indices = range(dataset_size)
            actual_num_samples = dataset_size
            # Note: Confidence/tolerance guarantees may not hold if dataset < m
            if actual_num_samples < num_samples_needed:
                print(
                    f"WARNING: Full dataset size ({actual_num_samples}) is less than required samples ({num_samples_needed}). Tolerance/confidence guarantees may not strictly hold.")
        else:
            # Sample without replacement
            samples_to_evaluate_indices = random.sample(
                range(dataset_size), num_samples_needed)
            actual_num_samples = num_samples_needed

        logger.info(f"Evaluating on {actual_num_samples} samples...")

        success_count = 0
        total_evaluated = 0
        failed_evaluations = 0
        sq_start_time = time.time()

        # Store original noise/trial settings to restore later
        original_noise_eta = self.noise_eta
        original_fitness_trials = self.fitness_trials
        # Use specified trials for SQ, maybe disable noise unless testing noise robustness
        self.fitness_trials = sq_fitness_trials
        # self.noise_eta = 0.0 # Optionally disable noise for pure prompt evaluation

        for i, index in enumerate(samples_to_evaluate_indices):
            image_prompt = self.image_prompts[index]
            logger.debug(
                f"SQ Sample {i+1}/{actual_num_samples}: Processing {image_prompt.filename}")
            image_path = os.path.join('images', image_prompt.filename)

            if not os.path.exists(image_path):
                logger.warning(
                    f"SQ Sample {i+1}: Image not found - {image_path}")
                failed_evaluations += 1
                continue

            try:
                # Evaluate the best prompt on this sample using current settings
                score, metrics = self.fitness(
                    best_prompt, image_prompt, image_path)
                total_evaluated += 1

                # Check if criterion (chi) is met
                if score >= criterion_threshold:
                    success_count += 1
                    logger.debug(
                        f"SQ Sample {i+1}: Success (Score {score:.3f} >= {criterion_threshold})")
                else:
                    logger.debug(
                        f"SQ Sample {i+1}: Failure (Score {score:.3f} < {criterion_threshold})")

            except Exception as e:
                logger.error(
                    f"SQ Sample {i+1}: Error evaluating fitness for {image_prompt.filename}: {e}", exc_info=True)
                failed_evaluations += 1
                continue  # Skip this sample

        # Restore original settings
        self.noise_eta = original_noise_eta
        self.fitness_trials = original_fitness_trials

        sq_duration = time.time() - sq_start_time
        logger.info(f"SQ evaluation finished. Duration: {sq_duration:.2f}s")
        logger.info(
            f"Total evaluated: {total_evaluated}, Successes (chi=1): {success_count}, Failures/Errors: {failed_evaluations}")

        if total_evaluated == 0:
            logger.error(
                "SQ Analysis Error: No samples could be successfully evaluated.")
            return None

        estimated_prob = success_count / total_evaluated
        logger.info(f"\n--- SQ Analysis Result ---")
        logger.info(f"  Best Prompt evaluated: '{best_prompt[:100]}...'")
        logger.info(
            f"  Estimated Probability (P_chi) that score > {criterion_threshold}: {estimated_prob:.4f}")
        logger.info(
            f"  (Based on {total_evaluated} samples, estimate is within +/- {tolerance} with {confidence*100:.1f}% confidence*)")
        if actual_num_samples < num_samples_needed:
            logger.info(
                f"  *Confidence guarantee may be weaker as full dataset size < required sample size.")
        logger.info(f"--- End SQ Analysis ---")

        # Return the estimated probability P_chi
        return estimated_prob


# --- Main Execution Logic ---
def main():
    # --- Configuration ---
    POPULATION_SIZE = 20
    GENERATIONS = 5  # Increased generations for potentially better results
    ELITE_SIZE = 4
    FITNESS_TRIALS = 2  # Average over 2 trials for more stable fitness
    NOISE_ETA = 0.1  # Introduce 10% random classification noise during fitness eval
    # SQ Analysis Params
    SQ_CRITERION_THRESHOLD = 0.75  # Target fitness score for success
    # Tau: +/- 43% desired accuracy (to target m=10)
    SQ_TOLERANCE = 0.43
    SQ_CONFIDENCE = 0.95        # 1-Delta: 95% confidence level
    SQ_FITNESS_TRIALS = 1       # Use 1 trial per sample during SQ analysis for speed

    # --- Initialization ---
    eot = EoTPrompting(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_size=ELITE_SIZE,
        fitness_trials=FITNESS_TRIALS,
        noise_eta=NOISE_ETA
    )
    logger.info("EoTPrompting initialized with:")
    logger.info(f"  Population Size: {eot.population_size}")
    logger.info(f"  Generations: {eot.generations}")
    logger.info(f"  Elite Size: {eot.elite_size}")
    logger.info(f"  Fitness Trials: {eot.fitness_trials}")
    logger.info(f"  Noise Eta: {eot.noise_eta}")

    # --- File Setup ---
    results_file = os.path.join(
        eot.results_dir, "cot_evolutionary_summary_results.json")
    # Optional: Clear file at start, or append/update based on image filename
    # if os.path.exists(results_file):
    #     logger.warning(f"Removing existing results file: {results_file}")
    #     os.remove(results_file)

    # Load existing results to update/compare
    existing_results_map = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                loaded_results = json.load(f)
                if isinstance(loaded_results, list):  # Ensure it's a list
                    for res in loaded_results:
                        if isinstance(res, dict) and "filename" in res:
                            existing_results_map[res["filename"]] = res
                logger.info(
                    f"Loaded {len(existing_results_map)} existing results from {results_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(
                f"Error loading existing results: {e}. Starting fresh.")
            existing_results_map = {}

    # --- Main Evolution Loop ---
    all_run_results = []  # Store results from this specific run
    for image_prompt in eot.image_prompts:
        logger.info(f"\n======= Processing {image_prompt.filename} =======")
        # ... (print prompt details)

        # Run evolution for this image
        reasoning, score, best_prompts_list, metrics = eot.generate_reasoning(
            image_prompt)

        # Prepare result dictionary for this image
        current_result = {
            "filename": image_prompt.filename,
            "original_prompt": image_prompt.prompt,
            "expected_answer": image_prompt.text_answer,
            "is_counting": image_prompt.counting,
            "numeric_answer": image_prompt.numeric_answer,
            "generated_reasoning": reasoning,
            "best_prompt_from_run": best_prompts_list[0] if best_prompts_list else None,
            # Contains final_score, trial info, noise info etc.
            "metrics": metrics
        }
        all_run_results.append(current_result)

        # --- Update Best Result Tracking ---
        filename = current_result["filename"]
        current_score = current_result.get(
            "metrics", {}).get("final_score", float('-inf'))

        # Check if current result is better than existing result for this file
        if filename not in existing_results_map or \
           current_score > existing_results_map[filename].get("metrics", {}).get("final_score", float('-inf')):
            logger.info(
                f"Updating best result for {filename} (New score: {current_score:.4f})")
            existing_results_map[filename] = current_result
        else:
            existing_score = existing_results_map[filename].get(
                "metrics", {}).get("final_score", float('-inf'))
            logger.info(
                f"Keeping existing best result for {filename} (Score: {existing_score:.4f} >= {current_score:.4f})")

        logger.info(
            f"Finished processing {image_prompt.filename}. Best score found in run: {score:.4f}")
        logger.info("=" * 60)
        # Optional: Add delay if hitting API limits
        # time.sleep(1)

    # --- Save Updated Best Results ---
    final_results_list = list(existing_results_map.values())
    final_results_list.sort(key=lambda x: x.get("metrics", {}).get(
        "final_score", 0), reverse=True)  # Sort overall results

    try:
        with open(results_file, 'w') as f:
            json.dump(final_results_list, f, indent=2)
        logger.info(
            f"Saved updated best results for {len(final_results_list)} images to {results_file}")
    except IOError as e:
        logger.error(f"Failed to save results to {results_file}: {e}")

    # --- Post-Hoc SQ Analysis ---
    if final_results_list:
        # Find the overall best prompt across all images processed
        overall_best_prompt = final_results_list[0].get("best_prompt_from_run")
        overall_best_score = final_results_list[0].get(
            "metrics", {}).get("final_score")
        logger.info(
            f"\nOverall best prompt found (Score: {overall_best_score:.4f}):")
        logger.info(f"'{overall_best_prompt}'")

        # Run SQ analysis on this prompt
        if overall_best_prompt:
            eot.analyze_prompt_with_sq(
                best_prompt=overall_best_prompt,
                criterion_threshold=SQ_CRITERION_THRESHOLD,
                tolerance=SQ_TOLERANCE,
                confidence=SQ_CONFIDENCE,
                sq_fitness_trials=SQ_FITNESS_TRIALS
            )
        else:
            logger.warning(
                "Could not determine overall best prompt for SQ analysis.")
    else:
        logger.warning(
            "No results available to determine best prompt for SQ analysis.")


if __name__ == "__main__":
    # Ensure you have base_prompting.py with BasePrompting and ImagePrompt classes defined
    # and your OpenAI API key set as an environment variable: OPENAI_API_KEY
    # Also ensure necessary directories like 'images' and 'results' exist.
    main()
