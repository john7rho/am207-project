# Japanese Uncertain Scenes Dataset (JUS-Dataset)

## Dataset Overview
The JUS-Dataset is a collection of images from Japan, designed to evaluate Vision Language Models (VLMs) on tasks involving uncertainty and counting. The dataset includes various scenes from Japan, including:
- Food items and restaurants
- Cultural sites and landmarks
- Urban scenes
- Natural landscapes
- Religious locations
- Transportation

### Dataset Structure
- `images/`: Directory containing all image files
- `labels.json`: Contains metadata for each image including:
  - `filename`: Image file name
  - `prompt`: Question about the image
  - `text_answer`: Expected textual response
  - `numeric_answer`: Expected numeric value (for counting tasks)
  - `counting`: Boolean indicating if it's a counting task

## Methods

### 1. Normal Method (normal.py)
Basic implementation that processes images with standard prompting techniques.
- Uses direct LLM queries
- Implements basic scoring
- No evolutionary or chain-of-thought components

### 2. Chain of Thought Method (cot.py)
Enhanced version that implements chain-of-thought reasoning.
- Breaks down reasoning into steps
- Provides more detailed explanations
- Improved handling of uncertainty
- Better performance on complex tasks

### 3. Evolutionary Method (evolutionary.py)
Advanced implementation using evolutionary algorithms to optimize prompts through iterative refinement.

#### Core Components:

1. **Population Management**
   - Fixed population size of 20 prompts per generation
   - Runs for 3 generations by default
   - Preserves top 4 performing prompts (elite_size) between generations
   - Each generation evolves through crossover and mutation

2. **Crossover System**
   - Splits prompts into instructions and numbered steps
   - Randomly combines instruction sets and step sequences from parent prompts
   - Maintains structural coherence while mixing effective elements
   - Preserves semantic meaning during combination

3. **Dual Mutation System**
   - Deterministic Mutation:
     - Adds emphasis words (carefully, precisely, systematically)
     - Substitutes counting-related terms
     - Adds verification steps
     - Fixed 30% mutation rate per line
   
   - Intelligent Mutation:
     - Analyzes previous performance metrics
     - Adds structural improvements (headers, verification sections)
     - Enhances confidence reporting
     - Falls back to deterministic mutation if analysis fails

4. **Elite Preservation**
   - Keeps top 4 performing prompts unchanged
   - Uses elite prompts as parents for next generation
   - Maintains performance benchmarks
   - Ensures stability in evolution process

5. **Results Management**
   - Tracks best performing prompts per image
   - Maintains comprehensive metrics including:
     - Semantic similarity
     - Numeric accuracy
     - Response quality measures
     - Counting analysis
   - Preserves highest scoring results across multiple runs

## Scoring System

The scoring system evaluates responses based on two main criteria:

### Counting Questions
```python
weights = {
    'numeric_accuracy': 0.7,
    'semantic_similarity': 0.3
}
```

### Descriptive Questions
```python
weights = {
    'semantic_similarity': 1.0
}
```

### Metrics Include:
- Semantic similarity using sentence transformers
- Numeric accuracy for counting tasks

## Installation & Setup

1. Install required packages:
```bash
pip install openai pillow torch transformers sentence-transformers nltk rouge tqdm numpy psutil matplotlib
```

2. Set up OpenAI API key:
```python
export OPENAI_API_KEY='your-api-key-here'
```

3. Prepare directory structure:
```
project/
├── images/
├── code/
│   ├── normal.py
│   ├── cot.py
│   ├── evolutionary.py
│   ├── evaluate_results.py
│   └── labels.json
```

## Running the Code

### 1. Normal Method
```bash
python normal.py
```
Outputs results to: `normal_results/summary_results.json`

### 2. Chain of Thought Method
```bash
python cot.py
```
Outputs results to: `cot_results/summary_results.json`

### 3. Evolutionary Method
```bash
python evolutionary.py
```
Outputs results to: `cot_evolutionary_results/summary_results.json`

### 4. Evaluate Results
```bash
python evaluate_results.py
```

This will:
- Generate comparison plots between all methods
- Create score distribution visualizations
- Output plots to `results/plots/`
  - `final_scores_comparison.png`
  - `score_distributions_boxplot.png`

## Note
- Ensure all images referenced in `labels.json` are present in the `images/` directory
- Results directories will be created automatically
- Each method can be run independently
- Run `evaluate_results.py` after running at least two methods for meaningful comparisons
```

The changes have been made to update the README.md file with comprehensive documentation about the dataset, methods, scoring system, and usage instructions. The file now provides clear guidance on installation, setup, and running each component of the system.