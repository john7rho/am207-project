import json
import os
import base64
import logging
import time
import re
from typing import List, Optional
from dataclasses import dataclass
import openai
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from openai import OpenAI
import psutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')


@dataclass
class ImagePrompt:
    filename: str
    prompt: str
    text_answer: str
    numeric_answer: Optional[float]
    counting: bool


class BasePrompting:
    def __init__(self):
        # Common initialization
        self.start_time = time.time()
        self.memory_usage = []

        # Model setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            self.similarity_model = self.similarity_model.half()

        # OpenAI setup
        self.client = OpenAI(api_key=openai.api_key)
        self.model = "gpt-4o"

        # Load evaluation models
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)

        # Results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Load dataset
        self.image_prompts = self.load_dataset()

        # Initialize conversation contexts
        self._conversation_contexts = {}

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"DEBUG: Image encoding failed - {str(e)}")
            raise

    def monitor_resources(self):
        """Monitor system resources."""
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)

    def load_dataset(self) -> List[ImagePrompt]:
        """Load image prompts from labels.json"""
        try:
            with open('labels.json', 'r') as f:
                data = json.load(f)

            image_prompts = []
            print(f"DEBUG: Found {len(data.get('images', []))} total images")

            # Only process first two images
            for item in data.get('images', [])[:10]:
                try:
                    # Add debug printing
                    print(
                        f"DEBUG: Processing {item.get('filename', 'unknown')}")

                    # Handle numeric_answer more carefully
                    numeric_answer = None
                    if 'numeric_answer' in item:
                        try:
                            numeric_answer = float(item['numeric_answer'])
                        except (ValueError, TypeError):
                            # If conversion fails, keep as None
                            pass

                    prompt = ImagePrompt(
                        filename=item.get('filename', ''),
                        prompt=item.get('prompt', ''),
                        text_answer=item.get('text_answer', ''),
                        numeric_answer=numeric_answer,
                        counting=item.get('counting', False)
                    )
                    image_prompts.append(prompt)
                    print(
                        f"DEBUG: Successfully added prompt for {prompt.filename}")
                except Exception as e:
                    logging.warning(f"Error processing item: {str(e)}")
                    continue

            print(f"DEBUG: Successfully loaded {len(image_prompts)} images")

            # Verify we have images
            if not image_prompts:
                logging.error("No images were successfully loaded")
                return []

            return image_prompts

        except FileNotFoundError:
            logging.error("Error: labels.json not found in current directory")
            print("DEBUG: Current directory contents:", os.listdir())
            return []
        except json.JSONDecodeError:
            logging.error("Error: labels.json is not valid JSON")
            return []
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            return []

    def calculate_cross_entropy(self, text: str) -> float:
        """Calculate cross-entropy loss for a given text using GPT-2"""
        inputs = self.tokenizer(text, return_tensors='pt').to('cpu')
        with torch.no_grad():
            outputs = self.lm_model(**inputs, labels=inputs['input_ids'])
        return outputs.loss.item()

    def verify_setup(self) -> bool:
        """Verify the OpenAI client and model setup"""
        try:
            print("\nDEBUG: Verifying OpenAI setup...")
            if not self.client.api_key:
                print("DEBUG: API key not found")
                return False

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )

            if response and response.choices:
                print("DEBUG: OpenAI setup verified successfully")
                return True
            print("DEBUG: Failed to get valid response from API")
            return False

        except Exception as e:
            print(f"DEBUG: Setup verification failed - {str(e)}")
            return False

    def verify_image_handling(self, image_path: str) -> bool:
        """Verify that images can be properly encoded and processed"""
        try:
            print(f"\nDEBUG: Verifying image handling for: {image_path}")
            if not os.path.exists(image_path):
                print(f"DEBUG: Image not found at: {image_path}")
                return False

            base64_image = self.encode_image(image_path)
            if not base64_image:
                print("DEBUG: Image encoding failed")
                return False

            print("DEBUG: Image handling verification successful")
            return True

        except Exception as e:
            print(f"DEBUG: Image handling verification failed - {str(e)}")
            return False

    def generate_reasoning(self, image_prompt: ImagePrompt) -> tuple[str, float, list[str], dict]:
        """Generate reasoning and return (reasoning, score, best_prompts, metrics)"""
        try:
            print(
                f"\nDEBUG: Starting generate_reasoning for {image_prompt.filename}")

            # Get evolved prompts and their results
            results = self.evolve_population(image_prompt)

            if not results:
                print("DEBUG: Evolution produced 0 results")
                return "Error: No valid results produced", 0.0, [], {}

            # Get best result
            best_score, best_prompt, best_reasoning, best_metrics = results[0]

            return best_reasoning, best_score, [best_prompt], best_metrics

        except Exception as e:
            print(f"DEBUG: Generate reasoning failed - {str(e)}")
            import traceback
            print("DEBUG: Full traceback:")
            print(traceback.format_exc())
            return f"Error: {str(e)}", 0.0, [], {}

    def get_base_prompts(self, image_prompt: ImagePrompt) -> List[str]:
        """Return the base prompts for the given image prompt."""
        try:
            print("DEBUG: Entering get_base_prompts")
            if not image_prompt or not image_prompt.prompt:
                print("DEBUG: Invalid image prompt or empty prompt")
                return []

            # Always include the original prompt
            prompts = [image_prompt.prompt]

            # Add some generic analysis prompts
            analysis_prompts = [
                f"{image_prompt.prompt}\nPlease analyze this image carefully and provide a detailed response.",
                f"{image_prompt.prompt}\nLook at the image thoroughly and describe what you see relevant to this question."
            ]
            prompts.extend(analysis_prompts)

            print(f"DEBUG: Generated {len(prompts)} prompts")
            return prompts

        except Exception as e:
            print(
                f"DEBUG: Error in get_base_prompts - {type(e).__name__}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return [image_prompt.prompt] if image_prompt and image_prompt.prompt else []

    def llm_query(self, prompt: str, image_path: str = None) -> str:
        """Query the LLM with the given prompt"""
        try:
            print("DEBUG: Starting llm_query")
            print(f"DEBUG: Image path: {image_path}")
            # Show first 100 chars of prompt
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

    def fitness(self, prompt: str, image_prompt: ImagePrompt, image_path: str) -> tuple[float, dict]:
        """Calculate fitness score for a prompt"""
        try:
            print("DEBUG: Starting fitness calculation")

            # Get the response only once and store it
            response = self.llm_query(prompt, image_path)

            metrics = {
                'final_score': 0.0,
                'semantic_similarity': 0.0,
                'numeric_accuracy': 0.0,
                'counting_metrics': {
                    'found_numbers': False,
                    'predicted_number': None,
                    'target_number': image_prompt.numeric_answer,
                    'numbers_found': []
                }
            }

            try:
                from sentence_transformers import SentenceTransformer, util
                model = SentenceTransformer('all-MiniLM-L6-v2')

                # Calculate embeddings
                response_embedding = model.encode(
                    response, convert_to_tensor=True)
                answer_embedding = model.encode(
                    image_prompt.text_answer, convert_to_tensor=True)

                # Calculate similarity
                similarity = float(util.pytorch_cos_sim(
                    response_embedding, answer_embedding)[0][0])
                metrics['semantic_similarity'] = similarity

            except ImportError:
                print(
                    "DEBUG: sentence-transformers not available, using basic similarity")
                metrics['semantic_similarity'] = 0.3  # default fallback value
            except Exception as e:
                print(f"DEBUG: Similarity calculation error: {str(e)}")
                metrics['semantic_similarity'] = 0.0

            # Calculate numeric accuracy if applicable
            if image_prompt.counting:
                try:
                    # First try to find numbers after specific keywords
                    final_count_match = re.search(
                        r'(?:final count|total|count):\s*(\d+)', response.lower())
                    if final_count_match:
                        predicted = float(final_count_match.group(1))
                    else:
                        # Fall back to all numbers found
                        numbers = re.findall(r'\d+', response)
                        if numbers:
                            # Use the last number as it's more likely to be the final count
                            predicted = float(numbers[-1])

                    metrics['counting_metrics']['found_numbers'] = True
                    metrics['counting_metrics']['predicted_number'] = predicted
                    metrics['counting_metrics']['numbers_found'] = [
                        float(n) for n in re.findall(r'\d+', response)]

                    if image_prompt.numeric_answer is not None:
                        error = abs(predicted - image_prompt.numeric_answer)
                        metrics['numeric_accuracy'] = max(
                            0, 1 - (error / image_prompt.numeric_answer))
                except Exception as e:
                    print(
                        f"DEBUG: Numeric accuracy calculation error: {str(e)}")

            # Calculate final score using only semantic similarity and numeric accuracy
            metrics['final_score'] = (
                self.weights['semantic_similarity'] * metrics['semantic_similarity'] +
                self.weights['numeric_accuracy'] * metrics['numeric_accuracy']
            )

            print(
                f"DEBUG: Fitness calculation complete. Score: {metrics['final_score']}")
            return metrics['final_score'], metrics

        except Exception as e:
            print(f"DEBUG: Fitness calculation failed - {str(e)}")
            import traceback
            print("DEBUG: Full traceback:")
            print(traceback.format_exc())
            return 0.0, {}
