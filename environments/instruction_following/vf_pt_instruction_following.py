# vf_pt_instruction_following.py
# Verifiers environment that wraps your PT constraints + two-shot prompt style
# Requires: verifiers >= 0.3, your pt_instructions.py in PYTHONPATH (same dir is fine)

from __future__ import annotations
from typing import List, Dict, Any
import random
import verifiers as vf
import json
from datasets import Dataset as HFDataset

# Import your existing constraint checker
# Place this file in the same directory as pt_instructions.py or install it as a pkg
import pt_instructions as pti

import logging

# -----------------------
# Few-shot exemplars (Portuguese)
# -----------------------
PT_EXEMPLARS: List[Dict[str, str]] = [
    {
        "instruction": (
            "Gostaria de saber sobre a evolução dos animais mais antigos da Terra... "
            "Forneça um exemplo em Python formatado em Markdown."
        ),
        "response": (
            "Os primeiros vertebrados surgem no Cambriano (~530 Ma).\n\n"
            "```python\nprint('exemplo')\n```"
        ),
    },
    {
        "instruction": (
            "Explique como usar SFTP (via SSH) para transferir ficheiros e compare com FTP."
        ),
        "response": (
            "Use SFTP por ser cifrado; FTP é plaintext.\n\n"
            "```markdown\n# SSH vs FTP\n- SSH: cifrado\n- FTP: texto simples\n```"
        ),
    },
]


def build_two_shot_prompt() -> str:
    """Recreates two-shot style: 2 exemplars + current instruction stub."""
    shots = random.sample(PT_EXEMPLARS, k=min(2, len(PT_EXEMPLARS)))
    parts = []
    for ex in shots:
        parts.append(f"Instruction: {ex['instruction']}\nResponse: {ex['response']}\n\n")
    return "".join(parts)


# -----------------------
# Rubric using your constraint registry
# -----------------------
class PTConstraintRubric(vf.Rubric):
    """Scores a completion against unified constraint formats supported in pt_instructions.py.

    Expects each datapoint to carry either:
      - {"constraints_used": [{"id","type","chosen_option"}, ...]}
      - {"instruction_id_list": ["pt:first_word", ...], "kwargs_pt": [{...}, ...]}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # one reward: fraction of constraints satisfied
        self.reward_funcs = [self._constraint_score]
        self.reward_weights = [1.0]

    def _constraint_score(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: Any,
        state: Dict[str, Any],
        **kwargs,
    ) -> float:
        # Full assistant message (final turn) content
        final_text = completion[-1]["content"] if completion else ""
        constraints_data = state.get("constraints_data", {})
        
        # Log the data for debugging
        logging.info(f"Final text for scoring: '{final_text}'")
        logging.info(f"Constraints data for scoring: {constraints_data}")
        
        results = pti.verify_response_unified(final_text, constraints_data)
        
        logging.info(f"Verification results: {results}")

        if not results:
            return 0.0
        # Boolean dict -> average score
        vals = list(results.values())
        # Some verifiers in pt_instructions may return floats; coerce to float, clamp [0,1]
        norm = []
        for v in vals:
            if isinstance(v, bool):
                norm.append(1.0 if v else 0.0)
            else:
                try:
                    fv = float(v)
                except Exception:
                    fv = 0.0
                norm.append(max(0.0, min(1.0, fv)))
        
        score = sum(norm) / len(norm) if norm else 0.0
        logging.info(f"Calculated score: {score}")
        return score


# -----------------------
# Environment
# -----------------------
class PTInstructionEnv(vf.SingleTurnEnv):
    """Single-turn environment that formats prompts in your two-shot style.

    Dataset item schema (min):
      {
        "instruction": str,
        # EITHER old format
        "constraints_used": [{"id","type","chosen_option"}, ...],
        # OR new format
        "instruction_id_list": ["pt:first_word", ...],
        "kwargs_pt": [{...}, ...]  # if you use new format
      }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_item = None

    def format_dataset(
        self,
        dataset: HFDataset,
        system_prompt: str | None = None,
        few_shot: list[dict] | None = None,
        question_key: str = "instruction",  # Use "instruction" instead of "question"
        answer_key: str = "answer",
    ) -> HFDataset:
        """Override to use 'instruction' field and preserve constraint information."""
        
        def format_prompt_fn(instruction):
            return self.format_messages({"instruction": instruction})
        
        # Create a modified dataset with constraint information in the info field
        def add_constraint_info(item):
            info = {}
            # Include constraint information from the original item
            if "constraints_used" in item:
                info["constraints_used"] = item["constraints_used"]
            elif "instruction_id_list" in item and (
                "kwargs_pt" in item or "kwargs" in item
            ):
                info["instruction_id_list"] = item["instruction_id_list"]
                # Accept either key for kwargs
                key = "kwargs_pt" if "kwargs_pt" in item else "kwargs"
                info["kwargs_pt"] = item[key]
            
            result = {
                "prompt": format_prompt_fn(item.get(question_key, "")),
                "info": info
            }
            
            if answer_key in item:
                result["answer"] = item[answer_key]
            
            return result
        
        # Map the dataset to include constraint info
        dataset = dataset.map(add_constraint_info)
        
        # Ensure required columns exist
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", list(range(len(dataset))))
        
        return dataset

    def format_messages(self, item: Dict[str, Any]) -> vf.Messages:
        # System message (Portuguese default like in your code)
        system = (
            "O teu nome é Amália, e és um modelo avançado de linguagem útil. "
            "Responde sempre na língua do utilizador, a menos que sejas instruído em contrário, "
            "e lembra-te que a tua língua principal é o português europeu."
        )
        two_shot = build_two_shot_prompt()
        # Current instruction with answer stub like your transform
        user_prompt = f"{two_shot}Instruction: {item.get('instruction','')}\nResponse: "
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

    async def setup_state(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        state = await super().setup_state(state, **kwargs)
        # Pass constraints through state so the rubric can access them
        constraints_data: Dict[str, Any] = {}
        
        # Try to get constraint data from state info first
        info = state.get("info", {})
        if "constraints_used" in info:
            constraints_data = {"constraints_used": info["constraints_used"]}
        elif "instruction_id_list" in info and (
            "kwargs_pt" in info or "kwargs" in info
        ):
            # accept either key for kwargs
            key = "kwargs_pt" if "kwargs_pt" in info else "kwargs"
            constraints_data = {
                "instruction_id_list": info["instruction_id_list"],
                "kwargs_pt": info[key],
            }
        # Fallback to current_item if available (for backwards compatibility)
        elif hasattr(self, 'current_item') and self.current_item is not None:
            if "constraints_used" in self.current_item:
                constraints_data = {"constraints_used": self.current_item["constraints_used"]}
            elif "instruction_id_list" in self.current_item and (
                "kwargs_pt" in self.current_item or "kwargs" in self.current_item
            ):
                # accept either key for kwargs
                key = "kwargs_pt" if "kwargs_pt" in self.current_item else "kwargs"
                constraints_data = {
                    "instruction_id_list": self.current_item["instruction_id_list"],
                    "kwargs_pt": self.current_item[key],
                }
        
        state["constraints_data"] = constraints_data
        return state


# -----------------------
# Public entry point
# -----------------------

def load_environment(dataset=None, *, num_examples: int | None = None, **kwargs):
    parser = vf.ThinkParser()
    rubric = PTConstraintRubric(parser=parser)

    # 👇 automatically handle .jsonl / .json file paths OR Python lists
    if isinstance(dataset, str):
        # User passed a path instead of a list or HF dataset
        print(f"Loading dataset from path: {dataset}")
        if dataset.endswith(".jsonl"):
            data = [json.loads(line) for line in open(dataset, "r", encoding="utf-8") if line.strip()]
        elif dataset.endswith(".json"):
            data = json.load(open(dataset, "r", encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported dataset file type: {dataset}")
        dataset = HFDataset.from_list(data)
    elif isinstance(dataset, list):
        dataset = HFDataset.from_list(dataset)
    elif dataset is None:
        dataset = HFDataset.from_list([])

    # 👇 Ensure an id column exists (verifiers expects this)
    if "id" not in dataset.column_names:
        dataset = dataset.add_column("id", list(range(len(dataset))))

    # Optional subselect for quick runs
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    return PTInstructionEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )