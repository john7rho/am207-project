from base_prompting import BasePrompting, ImagePrompt
import os
import logging
from typing import List, Optional, Dict


class CoTPrompting(BasePrompting):
    def __init__(self):
        super().__init__()

        # Override results directory
        self.results_dir = "results/cot_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Define weights for fitness metrics
        self.weights = {
            'semantic_similarity': 0.7,
            'numeric_accuracy': 0.3
        }

        # Add specialized prompts for different scenarios
        self.base_prompts = {
            'counting': [
                "Analyze this image step by step:\n"
                "1. Divide the image into clear sections\n"
                "2. Count items in each section methodically\n"
                "3. Double-check your count\n"
                "4. If counting is impossible, explain why\n"
                "5. State your confidence level\n"
                "Provide final count or explain why counting isn't feasible.",
            ],
            'descriptive': [
                "Analyze this image step by step:\n"
                "1. Describe what you can clearly see\n"
                "2. Note any ambiguous elements\n"
                "3. Address the specific question asked\n"
                "4. Provide your confidence level\n"
                "Be specific and avoid assumptions.",
            ]
        }

    def generate_reasoning(self, image_prompt: ImagePrompt) -> tuple[str, float, list[str], dict]:
        """Generate reasoning using chain-of-thought"""
        print("\nDEBUG: Starting generate_reasoning")

        image_path = os.path.join('images', image_prompt.filename)
        if not os.path.exists(image_path):
            return "Error: Image not found", 0.0, [], {}

        try:
            # Get appropriate base prompt
            prompt_type = 'counting' if image_prompt.counting else 'descriptive'
            base_prompt = self.base_prompts[prompt_type][0]

            # Combine with original prompt
            enhanced_prompt = (
                f"{image_prompt.prompt}\n\n"
                f"{base_prompt}\n\n"
                "Please provide your detailed reasoning step by step."
            )

            # Get response from LLM
            reasoning = self.llm_query(enhanced_prompt, image_path)

            if reasoning and not reasoning.startswith("Error:"):
                score, metrics = self.fitness(
                    enhanced_prompt, image_prompt, image_path)
                return reasoning, score, [enhanced_prompt], metrics

            return f"Error: Failed to generate reasoning", 0.0, [], {}

        except Exception as e:
            print(f"DEBUG: Generate reasoning failed - {str(e)}")
            return f"Error: {str(e)}", 0.0, [], {}


def main():
    cot = CoTPrompting()

    # Clear existing results file
    results_file = os.path.join(cot.results_dir, "cot_summary_results.json")
    if os.path.exists(results_file):
        os.remove(results_file)

    all_results = []

    for image_prompt in cot.image_prompts:
        print(f"\nProcessing {image_prompt.filename}...")
        print(f"Original prompt: {image_prompt.prompt}")
        print(f"Expected answer: {image_prompt.text_answer}")
        print(f"Is counting question: {image_prompt.counting}")
        print(f"Numeric answer: {image_prompt.numeric_answer}")

        reasoning, score, best_prompts, metrics = cot.generate_reasoning(
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

    # Save results
    with open(results_file, 'w') as f:
        import json
        json.dump(all_results, f, indent=2)

    # Print the saved results for verification
    print("\nSaved results:")
    with open(results_file, 'r') as f:
        saved_results = json.load(f)
        print(json.dumps(saved_results, indent=2))


if __name__ == "__main__":
    main()
