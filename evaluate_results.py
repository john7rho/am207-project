import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple  # Changed from tuple to Tuple
# These imports are needed if you run the first evaluate_results function
# from sentence_transformers import SentenceTransformer, util
# from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
import os
import logging

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# --------------------

# This function calculates detailed metrics but isn't used by the plotting part.
# Keep it if you need it for other analyses.
# def evaluate_results_detailed(labels_path: str, results_path: str):
#     # ... (original evaluate_results function code - requires nltk, rouge, sentence-transformers)
#     pass


def load_and_extract_metrics(file_path: str) -> Tuple[List[float], List[float]]:
    """Extract final_scores and semantic_similarity from results file."""
    logger.info(f"Loading metrics from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"Results file not found: {file_path}. Returning empty lists.")
        return [], []
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from: {file_path}. Returning empty lists.")
        return [], []

    final_scores = []
    semantic_scores = []

    # Handle potential variations in how scores are stored
    for r in results:
        metrics = r.get('metrics', {})
        if metrics:
            # Default to 0.0 if missing
            final_scores.append(metrics.get('final_score', 0.0))
            semantic_scores.append(metrics.get('semantic_similarity', 0.0))
        else:
            # Fallback if 'metrics' key is missing (older format?)
            final_scores.append(r.get('score', 0.0))
            semantic_scores.append(0.0)  # Cannot extract semantic score easily

    logger.info(
        f"Extracted {len(final_scores)} final scores and {len(semantic_scores)} semantic scores.")
    return final_scores, semantic_scores


def plot_final_scores_comparison(baseline_scores: List[float],
                                 cot_scores: List[float],
                                 cot_evo_scores: List[float],
                                 # Added new scores
                                 evo_sq_scores: List[float],
                                 filenames: List[str],
                                 output_dir: str):
    """Create grouped bar chart for final scores comparison for four methods."""
    if not all([baseline_scores, cot_scores, cot_evo_scores, evo_sq_scores]):
        logger.warning(
            "One or more score lists are empty. Skipping final scores plot.")
        return

    # Ensure all lists have the same length, pad with NaN or average if necessary, or error out
    n = len(filenames)
    if not (len(baseline_scores) == n and len(cot_scores) == n and len(cot_evo_scores) == n and len(evo_sq_scores) == n):
        logger.error(
            "Score lists have different lengths than filenames. Cannot plot comparison.")
        # Option: Pad shorter lists with np.nan, but this might distort plot interpretation.
        # max_len = max(len(baseline_scores), len(cot_scores), len(cot_evo_scores), len(evo_sq_scores), n)
        # baseline_scores.extend([np.nan] * (max_len - len(baseline_scores))) # etc. for others
        return  # Error out for now

    plt.figure(figsize=(15, 7))  # Adjusted size for more bars

    x = np.arange(n)
    width = 0.20  # Adjusted width for four bars

    plt.bar(x - 1.5*width, baseline_scores, width,
            label='Baseline', color='skyblue')
    plt.bar(x - 0.5*width, cot_scores, width,
            label='CoT (Static)', color='lightcoral')
    plt.bar(x + 0.5*width, cot_evo_scores, width,
            label='EoT (Original)', color='lightgreen')
    plt.bar(x + 1.5*width, evo_sq_scores, width,
            label='EvoSQ (Noisy)', color='gold')  # Added bar

    plt.xlabel('Image Filename')
    plt.ylabel('Final Scores (Higher is Better)')
    plt.title('Final Scores Comparison Across Methods')
    plt.xticks(x, filenames, rotation=60, ha='right',
               fontsize=8)  # Adjusted rotation/fontsize
    plt.ylim(0, 1.05)  # Assuming scores are 0-1
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "final_scores_comparison.png")
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved final scores plot to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save final scores plot: {e}")
    plt.close()


def plot_score_distributions(baseline_scores: List[float],
                             cot_scores: List[float],
                             cot_evo_scores: List[float],
                             evo_sq_scores: List[float],  # Added new scores
                             output_dir: str):
    """Create box plots to show score distributions across four methods."""
    # Filter out potential NaN values if padding was used, or check for empty lists
    data = [
        [s for s in baseline_scores if not np.isnan(s)],
        [s for s in cot_scores if not np.isnan(s)],
        [s for s in cot_evo_scores if not np.isnan(s)],
        [s for s in evo_sq_scores if not np.isnan(s)]  # Added new data
    ]
    # Remove empty lists after filtering
    data = [d for d in data if d]

    if len(data) < 4:  # Check if we have data for all methods intended
        logger.warning(
            f"Missing score data for one or more methods ({len(data)} found). Box plot may be incomplete.")
        if not data:
            logger.error(
                "No score data found for any method. Skipping distribution plot.")
            return

    labels = ['Baseline', 'CoT (Static)', 'EoT (Original)',
              'EvoSQ (Noisy)']  # Added label
    # Adjust labels if some data is missing
    if len(data) != len(labels):
        # This requires a more complex way to map data back to labels if some are missing entirely
        # For now, we'll plot what we have but log a warning
        logger.warning(
            "Mismatch between available data and expected labels. Plotting available data.")
        # Simple fix assuming order is maintained and only trailing methods might be missing:
        labels = labels[:len(data)]

    plt.figure(figsize=(10, 6))

    plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    plt.ylabel('Scores (Higher is Better)')
    plt.title('Score Distributions Across Methods')
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "score_distributions_boxplot.png")
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved score distributions plot to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save score distributions plot: {e}")
    plt.close()


def main():
    # --- Define Paths ---
    baseline_results_path = 'results/normal_results/normal_summary_results.json'
    cot_results_path = 'results/cot_results/cot_summary_results.json'
    cot_evo_results_path = 'results/cot_evolutionary_results/cot_evolutionary_summary_results.json'  # Original EoT
    # New EvoSQ (noisy)
    evo_sq_results_path = 'results/cot_evolutionary_results_noisy/cot_evolutionary_summary_results.json'
    output_dir = "results/plots"

    # --- Load Data ---
    baseline_scores, _ = load_and_extract_metrics(baseline_results_path)
    cot_scores, _ = load_and_extract_metrics(cot_results_path)
    cot_evo_scores, _ = load_and_extract_metrics(cot_evo_results_path)
    evo_sq_scores, _ = load_and_extract_metrics(
        evo_sq_results_path)  # Load new scores

    # Get filenames - Assuming they are consistent across runs and present in at least one file
    # Use the EvoSQ results file as the reference for filenames, assuming it's the latest
    filenames = []
    try:
        with open(evo_sq_results_path, 'r') as f:
            results = json.load(f)
        filenames = [r['filename'] for r in results]
        if not filenames:
            logger.error("No filenames found in reference results file.")
            return
        logger.info(
            f"Using {len(filenames)} filenames for plotting from {evo_sq_results_path}")
    except FileNotFoundError:
        logger.error(
            f"Reference results file not found: {evo_sq_results_path}. Cannot get filenames.")
        return
    except Exception as e:
        logger.error(f"Error loading filenames: {e}")
        return

    # --- Create Output Directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate Plots ---
    plot_final_scores_comparison(baseline_scores, cot_scores, cot_evo_scores,
                                 evo_sq_scores, filenames, output_dir)  # Pass new scores

    plot_score_distributions(baseline_scores, cot_scores, cot_evo_scores,
                             evo_sq_scores, output_dir)  # Pass new scores

    # --- Report on SQ Analysis ---
    logger.info("\n--- Statistical Query (SQ) Analysis Note ---")
    logger.info(
        f"The plots above compare the performance scores achieved by different methods.")
    logger.info(
        f"The 'EvoSQ (Noisy)' method also includes a specific post-hoc SQ analysis.")
    logger.info(
        f"This analysis estimates the probability (P_chi) that the single best prompt found by EvoSQ")
    logger.info(
        f"meets a predefined criterion (e.g., score > 0.75) across the entire dataset,")
    logger.info(f"with a specified tolerance (tau) and confidence (delta).")
    logger.info(
        f"Please check the console output or logs of the 'EoTPrompting' script run")
    logger.info(
        f"(the one generating results in '{os.path.dirname(evo_sq_results_path)}')")
    logger.info(
        f"for the specific numerical results of the SQ analysis (estimated P_chi value).")
    logger.info(f"--- End SQ Note ---")


if __name__ == "__main__":
    # Ensure NLTK data is downloaded if using evaluate_results_detailed with BLEU:
    # import nltk
    # try:
    #     nltk.data.find('tokenizers/punkt')
    # except nltk.downloader.DownloadError:
    #     nltk.download('punkt')
    main()
