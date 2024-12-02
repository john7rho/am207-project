from base_prompting import BasePrompting, ImagePrompt
import random
import json
import os
import logging
import time
import re
from typing import List, Optional


class EoTPrompting(BasePrompting):
    def __init__(self, population_size: int = 20, generations: int = 3, elite_size: int = 4):
        super().__init__()

        # Override results directory
        self.results_dir = "results/cot_evolutionary_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Evolution-specific parameters
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size

        # Define weights for fitness metrics
        self.weights = {
            'semantic_similarity': 0.7,
            'numeric_accuracy': 0.3
        }

        # Add specialized prompts for different scenarios
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

    def crossover(self, parent1: str, parent2: str) -> str:
        """Improved crossover with semantic understanding"""
        parts1 = parent1.split('\n')
        parts2 = parent2.split('\n')

        instructions1 = [p for p in parts1 if not p.strip(
        ).startswith(('1.', '2.', '3.', '4.'))]
        instructions2 = [p for p in parts2 if not p.strip(
        ).startswith(('1.', '2.', '3.', '4.'))]
        steps1 = [p for p in parts1 if p.strip(
        ).startswith(('1.', '2.', '3.', '4.'))]
        steps2 = [p for p in parts2 if p.strip(
        ).startswith(('1.', '2.', '3.', '4.'))]

        if random.random() < 0.5:
            result = instructions1 + steps2
        else:
            result = instructions2 + steps1

        return '\n'.join(result)

    def deterministic_mutate(self, prompt: str, mutation_rate: float = 0.3) -> str:
        """Enhanced mutation with semantic preservation"""
        lines = prompt.split('\n')
        mutated_lines = []

        emphasis_words = ['carefully', 'precisely',
                          'systematically', 'thoroughly', 'methodically']
        counting_words = ['count', 'tally',
                          'enumerate', 'quantify', 'calculate']
        verification_words = ['verify', 'check',
                              'confirm', 'validate', 'ensure']

        for line in lines:
            if random.random() < mutation_rate:
                if random.random() < 0.3:
                    line = random.choice(emphasis_words) + ' ' + line

                if 'count' in line.lower():
                    line = line.replace('count', random.choice(counting_words))

                if random.random() < 0.2:
                    line += f" and {random.choice(verification_words)} the result"

            mutated_lines.append(line)

        return '\n'.join(mutated_lines)

    def intelligent_mutate(self, prompt: str, previous_response: str, metrics: dict, mutation_rate: float = 0.3) -> str:
        try:
            # Analyze previous performance with corrected JSON format
            analysis_prompt = f"""
            Analyze this prompt and its performance:

            Original Prompt:
            {prompt}

            Response Generated:
            {previous_response}

            Performance Metrics:
            - Semantic Similarity: {metrics.get('semantic_similarity', 0)}
            - Numeric Accuracy: {metrics.get('numeric_accuracy', 0)}
            - Found Numbers: {metrics.get('counting_metrics', {}).get('found_numbers', False)}

            Please identify specific issues with the prompt:
            1. Is it failing to elicit precise counting?
            2. Is it missing important verification steps?
            3. Are the instructions clear enough?
            4. Does it handle uncertainty well?

            Provide your analysis in this exact JSON format:
            {{
                "issues": [
                    "issue1",
                    "issue2"
                ],
                "suggested_improvements": [
                    "improvement1",
                    "improvement2"
                ],
                "emphasis_needed": [
                    "area1",
                    "area2"
                ]
            }}

            Ensure your response is valid JSON with these exact keys.
            """

            # Get LLM analysis and ensure it's valid JSON
            analysis_response = self.llm_query(analysis_prompt)
            try:
                # Clean the response to ensure it only contains the JSON part
                json_str = analysis_response.strip()
                if json_str.startswith('```json'):
                    json_str = json_str.split('```json')[1]
                if json_str.endswith('```'):
                    json_str = json_str.split('```')[0]

                analysis = json.loads(json_str.strip())

                # Validate expected keys
                required_keys = {
                    'issues', 'suggested_improvements', 'emphasis_needed'}
                if not all(key in analysis for key in required_keys):
                    raise ValueError("Missing required keys in analysis")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Analysis parsing failed: {str(e)}")
                return self.deterministic_mutate(prompt, mutation_rate)

            # Build mutation strategy based on issues
            mutations = []

            # Common patterns that might need enhancement
            patterns = {
                'counting': {
                    'triggers': ['imprecise count', 'missed items', 'counting error'],
                    'enhancements': [
                        "First, divide the image into clear sections or grids.\n",
                        "Count systematically from left to right, top to bottom.\n",
                        "For each section, mark items as they are counted.\n"
                    ]
                },
                'verification': {
                    'triggers': ['no verification', 'accuracy issues', 'inconsistent counts'],
                    'enhancements': [
                        "Double-check each section's count.\n",
                        "Compare section totals with the overall count.\n",
                        "If counts differ, explain the discrepancy.\n"
                    ]
                },
                'uncertainty': {
                    'triggers': ['unclear visibility', 'assumption made', 'guessing'],
                    'enhancements': [
                        "If items are partially hidden or unclear, explicitly state this.\n",
                        "Provide confidence levels for each count (High/Medium/Low).\n",
                        "Explain any limitations in counting accuracy.\n"
                    ]
                },
                'structure': {
                    'triggers': ['unclear instructions', 'confusing steps', 'poor organization'],
                    'enhancements': [
                        "1. Assessment Phase:\n   - Evaluate visibility and counting challenges\n",
                        "2. Counting Phase:\n   - Use systematic section-by-section approach\n",
                        "3. Verification Phase:\n   - Review and validate all counts\n"
                    ]
                }
            }

            # Apply targeted mutations based on analysis
            lines = prompt.split('\n')
            mutated_lines = []

            for issue in analysis.get('issues', []):
                for category, pattern_data in patterns.items():
                    if any(trigger in issue.lower() for trigger in pattern_data['triggers']):
                        mutations.extend(pattern_data['enhancements'])

            # Apply structural improvements
            if mutations:
                # Add header if not present
                if not any(line.startswith("Instructions:") for line in lines):
                    mutated_lines.append("Instructions for Precise Counting:")

                # Add mutations
                mutated_lines.extend(mutations)

                # Add existing lines that don't conflict
                for line in lines:
                    if not any(mutation.strip() in line for mutation in mutations):
                        mutated_lines.append(line)

                # Add verification section if needed
                if 'verification' in str(analysis.get('emphasis_needed', [])):
                    mutated_lines.append("\nVerification Steps:")
                    mutated_lines.extend([
                        "1. Review each section count",
                        "2. Compare section totals with overall count",
                        "3. Document any uncertainties or limitations"
                    ])

            # If no specific mutations were needed, fall back to original with minor enhancements
            if not mutated_lines:
                return self.deterministic_mutate(prompt, mutation_rate)

            # Add confidence reporting if not present
            if not any('confidence' in line.lower() for line in mutated_lines):
                mutated_lines.extend([
                    "\nConfidence Reporting:",
                    "- State confidence level for each count (High/Medium/Low)",
                    "- Explain factors affecting confidence"
                ])

            return '\n'.join(mutated_lines)

        except Exception as e:
            print(f"Intelligent mutation failed: {str(e)}")
            return self.deterministic_mutate(prompt, mutation_rate)

    def fitness(self, prompt: str, image_prompt: ImagePrompt, image_path: str) -> tuple[float, dict]:
        """Calculate fitness score for a prompt"""
        try:
            print("DEBUG: Starting fitness calculation")

            # Get LLM response only once
            response = self.llm_query(prompt, image_path)

            metrics = {
                'semantic_similarity': 0.0,
                'numeric_accuracy': 0.0,
                'response_quality': {
                    'length_score': 0.0,
                    'coherence_score': 0.0,
                    'actual_length': len(response.split()),
                    'ideal_length': 100
                },
                'weights_used': self.weights
            }

            # Calculate semantic similarity
            try:
                from sentence_transformers import SentenceTransformer, util
                if not hasattr(self, '_similarity_model'):
                    self._similarity_model = SentenceTransformer(
                        'all-MiniLM-L6-v2')

                # Calculate embeddings
                response_embedding = self._similarity_model.encode(
                    response, convert_to_tensor=True)
                answer_embedding = self._similarity_model.encode(
                    image_prompt.text_answer, convert_to_tensor=True)

                # Calculate similarity
                similarity = float(util.pytorch_cos_sim(
                    response_embedding, answer_embedding)[0][0])
                metrics['semantic_similarity'] = max(
                    0.0, min(1.0, (similarity + 1) / 2))

            except ImportError:
                print(
                    "DEBUG: sentence-transformers not available, using basic similarity")
                metrics['semantic_similarity'] = 0.3  # default fallback
            except Exception as e:
                print(f"DEBUG: Similarity calculation error: {str(e)}")
                metrics['semantic_similarity'] = 0.0

            # Calculate final score with weighted components
            final_score = (
                self.weights['semantic_similarity'] * metrics['semantic_similarity'] +
                self.weights['response_quality'] *
                metrics['response_quality']['length_score']
            )

            if image_prompt.counting:
                try:
                    import re
                    numbers = re.findall(r'\d+', response)
                    if numbers and image_prompt.numeric_answer:
                        predicted = float(numbers[0])
                        error = abs(predicted - image_prompt.numeric_answer)
                        metrics['numeric_accuracy'] = max(
                            0, 1 - (error / image_prompt.numeric_answer))
                        final_score += self.weights['numeric_accuracy'] * \
                            metrics['numeric_accuracy']
                except Exception as e:
                    print(
                        f"DEBUG: Numeric accuracy calculation error: {str(e)}")

            print(f"DEBUG: Fitness calculation complete. Score: {final_score}")
            return final_score, metrics

        except Exception as e:
            print(f"DEBUG: Fitness calculation failed - {str(e)}")
            import traceback
            print(traceback.format_exc())
            return 0.0, {}

    def calculate_numeric_accuracy(self, rationale: str, target: Optional[float], target_text: str) -> float:
        """Enhanced numeric accuracy calculation"""
        # Handle "impossible to count" cases
        impossible_phrases = ['impossible to count', 'cannot count', 'unable to count',
                              'not possible to count', 'cannot provide exact count']
        if any(phrase in rationale.lower() for phrase in impossible_phrases):
            if any(phrase in target_text.lower() for phrase in ['unknown', 'impossible', 'cannot']):
                return 1.0
            return 0.0

        # Extract numbers from rationale
        numbers = re.findall(r'\b\d+\b', rationale)
        if not numbers:
            return 0.0

        # Get the final number mentioned (usually the total)
        predicted = float(numbers[-1])

        if target is None:
            return 0.0

        # Calculate accuracy with tolerance
        tolerance = max(0.1 * target, 2)  # 10% or at least 2 units
        error = abs(predicted - target)

        if error <= tolerance:
            return 1.0
        else:
            return max(0.0, 1.0 - (error - tolerance) / target)

    def evolve_population(self, image_prompt: ImagePrompt) -> List[tuple[float, str, str, dict]]:
        """Evolve population of prompts and return best results"""
        try:
            print("\nDEBUG: Starting evolve_population")
            image_path = os.path.join('images', image_prompt.filename)
            print(f"DEBUG: Image path: {image_path}")
            print(f"DEBUG: Image exists: {os.path.exists(image_path)}")

            # Initialize population with base prompts
            population = []

            # Add counting-specific prompts if it's a counting question
            if image_prompt.counting:
                print("DEBUG: Adding counting prompts")
                population.extend(self.base_prompts['counting'])
            else:
                print("DEBUG: Adding descriptive prompts")
                population.extend(self.base_prompts['descriptive'])

            # Add the original prompt
            population.append(image_prompt.prompt)

            # Add variations of the original prompt
            population.extend([
                f"{image_prompt.prompt}\nPlease be specific and detailed in your response.",
                f"{image_prompt.prompt}\nAnalyze the image carefully before answering."
            ])

            print(f"DEBUG: Initial population size: {len(population)}")
            print("DEBUG: Initial prompts:")
            for i, p in enumerate(population):
                print(f"  {i+1}. {p[:100]}...")

            # Track all results
            all_results = []

            # Evolution loop
            for generation in range(self.generations):
                print(
                    f"\nDEBUG: Generation {generation + 1}/{self.generations}")

                # Evaluate current population
                generation_results = []
                for i, prompt in enumerate(population):
                    try:
                        print(
                            f"\nDEBUG: Evaluating prompt {i+1}/{len(population)}")
                        print(f"DEBUG: Prompt: {prompt[:100]}...")

                        print("DEBUG: Calling llm_query...")
                        reasoning = self.llm_query(prompt, image_path)
                        print(f"DEBUG: LLM response: {reasoning[:100]}...")

                        if reasoning and not reasoning.startswith("Error:"):
                            print("DEBUG: Calculating fitness...")
                            score, metrics = self.fitness(
                                prompt, image_prompt, image_path)
                            print(f"DEBUG: Fitness score: {score}")
                            generation_results.append(
                                (score, prompt, reasoning, metrics))
                        else:
                            print(f"DEBUG: Invalid response: {reasoning}")
                    except Exception as e:
                        print(f"DEBUG: Error evaluating prompt: {str(e)}")
                        import traceback
                        print("DEBUG: Traceback:")
                        print(traceback.format_exc())
                        continue

                print(
                    f"\nDEBUG: Generation {generation + 1} results: {len(generation_results)}")

                # Sort by score
                generation_results.sort(reverse=True, key=lambda x: x[0])

                # Add to all-time results
                all_results.extend(generation_results)

                # If we have valid results, evolve the population
                if generation_results:
                    # Select best performers
                    elite = generation_results[:self.elite_size]
                    population = [result[1]
                                  for result in elite]  # Keep elite prompts

                    # Generate new population through crossover and mutation
                    while len(population) < self.population_size:
                        if len(elite) >= 2:
                            parent1 = random.choice(elite)[1]
                            parent2 = random.choice(elite)[1]
                            child = self.crossover(parent1, parent2)
                            child = self.deterministic_mutate(child)
                            population.append(child)
                        else:
                            # If not enough elite members, add more variations
                            population.append(
                                self.deterministic_mutate(elite[0][1]))

                print(f"DEBUG: Generation {generation + 1} complete. Best score: "
                      f"{generation_results[0][0] if generation_results else 'No valid results'}")

            # Sort all results by score and return top results
            all_results.sort(reverse=True, key=lambda x: x[0])
            return all_results[:self.elite_size] if all_results else []

        except Exception as e:
            print(f"DEBUG: Evolution failed - {str(e)}")
            import traceback
            print("DEBUG: Full traceback:")
            print(traceback.format_exc())
            return []

    def llm_query(self, prompt: str, image_path: str = None) -> str:
        """Query the LLM with the given prompt"""
        try:
            print("\nDEBUG: Starting llm_query")
            print(f"DEBUG: Image path: {image_path}")
            print(f"DEBUG: Image exists: {os.path.exists(image_path)}")
            print(f"DEBUG: Prompt: {prompt[:100]}...")

            if image_path is None:
                print("DEBUG: Text-only query")
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800
                )
                return response.choices[0].message.content.strip()

            # Image query
            print("DEBUG: Encoding image...")
            base64_image = self.encode_image(image_path)

            print("DEBUG: Creating chat completion...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=800
            )

            print("DEBUG: Got response from API")
            result = response.choices[0].message.content.strip()
            print(f"DEBUG: Response preview: {result[:100]}...")
            return result

        except Exception as e:
            print(f"DEBUG: llm_query failed - {type(e).__name__}: {str(e)}")
            import traceback
            print("DEBUG: Full traceback:")
            print(traceback.format_exc())
            return f"Error: {str(e)}"

    def generate_reasoning(self, image_prompt: ImagePrompt) -> tuple[str, float, list[str], dict]:
        """Generate reasoning using evolved prompts"""
        print("\nDEBUG: Starting generate_reasoning")

        image_path = os.path.join('images', image_prompt.filename)
        if not os.path.exists(image_path):
            return "Error: Image not found", 0.0, [], {}

        try:
            # Get best prompts with their scores and reasoning
            best_results = self.evolve_population(image_prompt)
            print(f"DEBUG: Evolution produced {len(best_results)} results")

            if not best_results:
                return "Error: No valid results produced", 0.0, [], {}

            # Use the highest scoring result directly
            best_score, best_prompt, best_reasoning, best_metrics = best_results[0]

            print(f"\nDEBUG: Using best result:")
            print(f"Score: {best_score}")
            print(f"Prompt: {best_prompt[:100]}...")
            print(f"Reasoning: {best_reasoning[:100]}...")

            return best_reasoning, best_score, [best_prompt], best_metrics

        except Exception as e:
            print(f"DEBUG: Generate reasoning failed - {str(e)}")
            return f"Error: {str(e)}", 0.0, [], {}


def main():
    eot = EoTPrompting()

    # Clear existing results file
    results_file = os.path.join(
        eot.results_dir, "cot_evolutionary_summary_results.json")
    if os.path.exists(results_file):
        os.remove(results_file)

    all_results = []

    for image_prompt in eot.image_prompts:
        print(f"\nProcessing {image_prompt.filename}...")
        print(f"Original prompt: {image_prompt.prompt}")
        print(f"Expected answer: {image_prompt.text_answer}")
        print(f"Is counting question: {image_prompt.counting}")
        print(f"Numeric answer: {image_prompt.numeric_answer}")

        reasoning, score, best_prompts, metrics = eot.generate_reasoning(
            image_prompt)

        # Add result to all_results with detailed metrics
        result = {
            "filename": image_prompt.filename,
            "original_prompt": image_prompt.prompt,
            "expected_answer": image_prompt.text_answer,
            "generated_reasoning": reasoning,
            "best_prompts": best_prompts,
            "metrics": {
                **metrics,  # Include all metrics
                "final_score": score  # Ensure final_score is included
            }
        }
        all_results.append(result)

        print(f"\nGenerated reasoning (score: {score:.3f}):")
        print(reasoning)
        print("\n" + "="*80 + "\n")

    # Ensure results directory exists
    os.makedirs(eot.results_dir, exist_ok=True)

    # Save results
    summary_file = os.path.join(
        eot.results_dir, "cot_evolutionary_summary_results.json")

    # Load existing results
    existing_results = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    # Create a dictionary to keep track of highest scoring result for each image
    best_results = {}

    # Process existing results
    for result in existing_results:
        filename = result["filename"]
        # Get score from metrics if available, otherwise from root level
        score = (result.get("metrics", {}).get("final_score", None) or
                 result.get("score", float('-inf')))
        if not isinstance(score, (int, float)):
            score = float('-inf')

        if filename not in best_results or score > best_results[filename].get("metrics", {}).get("final_score", float('-inf')):
            best_results[filename] = result

    # Process new results
    for result in all_results:
        filename = result["filename"]
        score = result["metrics"]["final_score"]

        if filename not in best_results or score > best_results[filename].get("metrics", {}).get("final_score", float('-inf')):
            best_results[filename] = result

    # Convert back to list
    final_results = list(best_results.values())

    # Save final results
    with open(summary_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print the saved results for verification
    print("\nSaved results:")
    with open(summary_file, 'r') as f:
        saved_results = json.load(f)
        print(json.dumps(saved_results, indent=2))


if __name__ == "__main__":
    main()
