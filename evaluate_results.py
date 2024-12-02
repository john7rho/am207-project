import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import os


def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def evaluate_results(labels_path: str, results_path: str):
    # Load data
    labels = load_json(labels_path)
    results = load_json(results_path)

    # Create lookup dictionary for labels
    labels_dict = {img['filename']: img for img in labels['images']}

    # Initialize metrics
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    rouge = Rouge()
    metrics = {
        'semantic_similarity': [],
        'bleu_scores': [],
        'rouge_scores': [],
        'numeric_accuracy': []
    }

    # Evaluate each result
    for result in results:
        filename = result['filename']
        generated = result['generated_reasoning']
        label = labels_dict[filename]

        # Semantic Similarity
        emb1 = similarity_model.encode(generated, convert_to_tensor=True)
        emb2 = similarity_model.encode(
            label['text_answer'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        metrics['semantic_similarity'].append(similarity)

        # BLEU Score
        bleu = sentence_bleu([label['text_answer'].split()], generated.split())
        metrics['bleu_scores'].append(bleu)

        # ROUGE Score
        rouge_scores = rouge.get_scores(generated, label['text_answer'])
        metrics['rouge_scores'].append(rouge_scores[0]['rouge-l']['f'])

        # Numeric Accuracy (if applicable)
        if 'numeric_answer' in label and label['numeric_answer'] is not None:
            # Extract numbers from generated text
            import re
            numbers = [float(n) for n in re.findall(r'\d+\.?\d*', generated)]
            if numbers:
                predicted = min(numbers, key=lambda x: abs(
                    float(x) - float(label['numeric_answer'])))
                error = abs(predicted - float(label['numeric_answer']))
                metrics['numeric_accuracy'].append(error)

    # Calculate summary statistics
    summary = {
        'semantic_similarity': {
            'mean': np.mean(metrics['semantic_similarity']),
            'std': np.std(metrics['semantic_similarity'])
        },
        'bleu_scores': {
            'mean': np.mean(metrics['bleu_scores']),
            'std': np.std(metrics['bleu_scores'])
        },
        'rouge_scores': {
            'mean': np.mean(metrics['rouge_scores']),
            'std': np.std(metrics['rouge_scores'])
        }
    }

    if metrics['numeric_accuracy']:
        summary['numeric_error'] = {
            'mean': np.mean(metrics['numeric_accuracy']),
            'std': np.std(metrics['numeric_accuracy'])
        }

    return summary


def load_and_extract_metrics(file_path: str) -> tuple[List[float], List[float]]:
    """Extract final_scores and semantic_similarity from results file."""
    with open(file_path, 'r') as f:
        results = json.load(f)

    final_scores = [r['metrics']['final_score'] for r in results]
    semantic_scores = [r['metrics']['semantic_similarity'] for r in results]
    return final_scores, semantic_scores


def plot_final_scores_comparison(baseline_scores: List[float], cot_scores: List[float],
                                 cot_evo_scores: List[float], filenames: List[str], output_dir: str):
    """Create grouped bar chart for final scores comparison."""
    plt.figure(figsize=(12, 6))

    x = np.arange(len(filenames))
    width = 0.25

    plt.bar(x - width, baseline_scores, width, label='Baseline')
    plt.bar(x, cot_scores, width, label='Chain of Thought')
    plt.bar(x + width, cot_evo_scores, width, label='CoT Evolutionary')

    plt.xlabel('Images')
    plt.ylabel('Final Scores')
    plt.title('Final Scores Comparison Across Methods')
    plt.xticks(x, filenames, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    output_path = f"{output_dir}/final_scores_comparison.png"
    plt.savefig(output_path)
    print(f"Saved final scores plot to: {output_path}")
    plt.close()


def plot_score_distributions(baseline_scores: List[float], cot_scores: List[float],
                             cot_evo_scores: List[float], output_dir: str):
    """Create box plots to show score distributions across methods."""
    plt.figure(figsize=(10, 6))

    data = [baseline_scores, cot_scores, cot_evo_scores]
    labels = ['Baseline', 'Chain of Thought', 'CoT Evolutionary']

    plt.boxplot(data, labels=labels)
    plt.ylabel('Scores')
    plt.title('Score Distributions Across Methods')

    plt.tight_layout()
    output_path = f"{output_dir}/score_distributions_boxplot.png"
    plt.savefig(output_path)
    print(f"Saved score distributions plot to: {output_path}")
    plt.close()


def main():
    # Load results from each method
    baseline_scores, baseline_sem = load_and_extract_metrics(
        'results/normal_results/normal_summary_results.json')
    cot_scores, cot_sem = load_and_extract_metrics(
        'results/cot_results/cot_summary_results.json')
    cot_evo_scores, cot_evo_sem = load_and_extract_metrics(
        'results/cot_evolutionary_results/cot_evolutionary_summary_results.json')

    # Get filenames from normal results (assuming same across all)
    with open('results/normal_results/normal_summary_results.json', 'r') as f:
        results = json.load(f)
    filenames = [r['filename'] for r in results]

    # Create output directory for plots
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create plots with output directory
    plot_final_scores_comparison(baseline_scores, cot_scores,
                                 cot_evo_scores, filenames, output_dir)
    plot_score_distributions(baseline_scores, cot_scores,
                             cot_evo_scores, output_dir)


if __name__ == "__main__":
    main()
