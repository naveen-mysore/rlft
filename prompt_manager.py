# prompt_manager.py
# ---------------------------------------------------------------------
import json
import re
from typing import List, Dict, Tuple, Optional, Union


# ---------------------------------------------------------------------
# 0.  Helper – build the canonical message list used in SFT
# ---------------------------------------------------------------------
def build_messages(system: str, query: str, cot: str, answer: str) -> List[Dict[str, str]]:
    """
    Conversation that goes into `tokenizer.apply_chat_template`.

    • `cot`  – chain‑of‑thought, can be "" if you don’t store CoT
    • `answer` – numeric ground‑truth (string or number)
    """
    return [
        {
            "role": "system",
            "content": system,
        },
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": f"{cot}\n",
        },
    ]


# ---------------------------------------------------------------------
# 1.  PromptManager
# ---------------------------------------------------------------------
class PromptManager:
    """
    Handles *all* prompt construction & parsing for both SFT and PPO.
    Works with any model whose tokenizer defines a chat template
    (Llama‑2/3, Mistral, Phi‑3, …).
    """

    # -----------------------------------------------------------------
    # 1‑A  Initialise with a tokenizer
    # -----------------------------------------------------------------
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.system_instructions = """For the given query including a meal description, think step by step as follows:
                1. Parse the meal description into discrete food or beverage items along with their serving size. If the serving size of any item in the meal is not specified, assume it is a single standard serving based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate to the item name and serving size.
                2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the specific serving size.
                3. Respond with a dictionary object containing the total carbohydrates in grams as follows:
                {{"total_carbohydrates": total grams of carbohydrates for the serving}}
                For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. If you don't know the answer, set the value of "total_carbohydrates" to -1.

                Follow the format of the following examples when answering

                Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."
                Answer: 
                The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
                1 cup of oatmeal has 27g carbs.
                1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.
                1 glass of orange juice has 26g carbs.
                So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5
                Output: {{"total_carbohydrates": 66.5}}

                Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
                Answer: 
                The meal consists of scrambled eggs made with 2 eggs and 1 toast.
                Scrambled eggs made with 2 eggs has 2g carbs.
                1 toast has 13g carbs.
                So the total grams of carbs in the meal = (2 + 13) = 15
                Output: {{"total_carbohydrates": 15}}

                Query: "Half a peanut butter and jelly sandwich."
                Answer: 
                The meal consists of 1/2 a peanut butter and jelly sandwich.
                1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs
                So the total grams of carbs in the meal = 25.3
                Output: {{"total_carbohydrates": 25.3}}"""

    # -----------------------------------------------------------------
    # 1‑B  -----  SFT  -------------------------------------------------
    # -----------------------------------------------------------------
    def build_training_sample(
        self, query: str, cot: str, answer: str
    ) -> str:
        """
        Creates one *fully‑tokenised* training example containing the
        assistant answer.  You feed the resulting string to the
        tokenizer again in your `preprocess_function`.
        """
        prompt = self.tok.apply_chat_template(
            build_messages(self.system_instructions, query, cot, answer),
            add_generation_prompt=False,   # answer is *inside* the prompt
            tokenize=False,
        )
        # template already adds an <eos>; keep a paranoid check
        if not prompt.endswith(self.tok.eos_token):
            prompt += self.tok.eos_token
        return prompt

    # -----------------------------------------------------------------
    # 1‑C  -----  Inference / PPO rollout  ----------------------------
    # -----------------------------------------------------------------
    def build_inference_prompt(self, query: str) -> str:
        """
        Prompt with an *empty* assistant turn so `generate()` starts
        right after the header.
        """
        messages = [
            {"role": "system", "content": self.system_instructions},
            {"role": "user", "content": query},
        ]
        return self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,   # assistant: … (empty)
            tokenize=False,
        )

    # -----------------------------------------------------------------
    # 1‑D  -----  Parsing helpers  ------------------------------------
    # -----------------------------------------------------------------
    def _first_json_block(self, text: str) -> Optional[str]:
        """
        Finds the *last*  occurrence of something that looks like

            Output: { ... }

        and returns the JSON substring (the part between the braces,
        braces included).  We take the last occurrence because the
        model sometimes echoes the instruction examples before giving
        the real answer.
        """
        # (?s)    → DOTALL so "." also matches new‑lines
        # Output: → literal marker, possibly with extra spaces
        # \s*     → optional whitespace / new‑lines
        # (\{.*?\}) → the *smallest* {...} that follows
        pattern = re.compile(r"(?s)Output:\s*(\{.*?\})")
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def parse_cot_and_answer(self, generated: str) -> Tuple[str, str]:
        """
        Splits the *raw* model output into:

            • cot          – everything before the chosen `Output:` block
            • answer_part  – the value corresponding to "total_carbohydrates"
                             *if* we can parse the JSON, otherwise the raw
                             JSON string itself (so that downstream code
                             still has something to work with)
        """
        json_block = self._first_json_block(generated)

        # Default fall‑backs
        cot_part = generated.strip()
        answer_part = ""

        if json_block:
            # CoT is whatever precedes the *chosen* Output block
            cot_part = generated[: generated.rfind(json_block)].strip()

            # Try to interpret the JSON
            try:
                data = json.loads(json_block)
                # Accept a couple of reasonable key spellings
                for key in ("total_carbohydrates", "total_carbs", "carbohydrates"):
                    if key in data:
                        answer_part = str(data[key])
                        break
            except json.JSONDecodeError:
                # Leave answer_part empty – we’ll deal with it in the next step
                answer_part = json_block

        return cot_part, answer_part

    def extract_carbs_from_answer(self, answer_text: Union[str, int, float]) -> Optional[float]:
        """
        Returns a *float* if we can see one; otherwise None.

        Works for:
            – numeric types
            – strings such as '1.56'  or  '{"total_carbohydrates": 1.56}'
            – anything that merely *contains* a number, e.g.
              'Output: {"total_carbohydrates": 1.56}'
        """
        # Already numeric?
        if isinstance(answer_text, (int, float)):
            return float(answer_text)

        # Pull the *first* float‑looking token out of the text
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(answer_text))
        return float(match.group()) if match else None