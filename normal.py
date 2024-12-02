from base_prompting import BasePrompting, ImagePrompt
import os
import json
from typing import List, Dict, Optional


class NormalPrompting(BasePrompting):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

        # Override results directory
        self.results_dir = "results/normal_results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Define weights for fitness metrics
        self.weights = {
            'semantic_similarity': 0.7,
            'numeric_accuracy': 0.3
        }

    def generate_reasoning(self, image_prompt: ImagePrompt) -> tuple[str, float, list[str], dict]:
        """Generate reasoning using direct prompting"""
        print("\nDEBUG: Starting generate_reasoning")

        image_path = os.path.join('images', image_prompt.filename)
        if not os.path.exists(image_path):
            return "Error: Image not found", 0.0, [], {}

        try:
            # Use the original prompt directly
            prompt = image_prompt.prompt

            # Get response from LLM
            reasoning = self.llm_query(prompt, image_path)

            if reasoning.startswith("Error:"):
                return reasoning, 0.0, [], {}

            # Calculate fitness score
            score, metrics = self.fitness(prompt, image_prompt, image_path)

            return reasoning, score, [prompt], metrics

        except Exception as e:
            print(f"DEBUG: Generate reasoning failed - {str(e)}")
            return f"Error: {str(e)}", 0.0, [], {}


def main():
    normal = NormalPrompting(debug=False)

    # Clear existing results file
    results_file = os.path.join(
        normal.results_dir, "normal_summary_results.json")
    if os.path.exists(results_file):
        os.remove(results_file)

    all_results = []

    for image_prompt in normal.image_prompts:
        print(f"\nProcessing {image_prompt.filename}...")
        print(f"Original prompt: {image_prompt.prompt}")
        print(f"Expected answer: {image_prompt.text_answer}")
        print(f"Is counting question: {image_prompt.counting}")
        print(f"Numeric answer: {image_prompt.numeric_answer}")

        reasoning, score, prompts, metrics = normal.generate_reasoning(
            image_prompt)

        # Add result to all_results with detailed metrics
        result = {
            "filename": image_prompt.filename,
            "original_prompt": image_prompt.prompt,
            "expected_answer": image_prompt.text_answer,
            "generated_reasoning": reasoning,
            "prompts_used": prompts,
            "metrics": {
                **metrics,
                "final_score": score
            }
        }
        all_results.append(result)

        print(f"\nGenerated reasoning (score: {score:.3f}):")
        print(reasoning)
        print("\n" + "="*80 + "\n")

    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print the saved results for verification
    print("\nSaved results:")
    with open(results_file, 'r') as f:
        saved_results = json.load(f)
        print(json.dumps(saved_results, indent=2))


if __name__ == "__main__":
    main()
