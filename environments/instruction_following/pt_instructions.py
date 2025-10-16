# -*- coding: utf-8 -*-
"""
pt_instructions.py

Coleção de restrições em português, no estilo do en_instructions.py.
Cada restrição é uma classe com:
  - type: string canónica
  - description(chosen_option) -> str
  - check(response: str, chosen_option) -> bool

Há um registro CONSTRAINT_REGISTRY que mapeia type -> classe,
e utilitários generate_description e verify_response.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional


# ------------- Utilitários de texto -------------

_WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ]+\b", flags=re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SIMPLE_SENT_SPLIT_RE = re.compile(r"[.!?]")
_BULLET_A_RE = re.compile(r"^\s*\*[^\*].*$", flags=re.MULTILINE)
_BULLET_B_RE = re.compile(r"^\s*-.*$", flags=re.MULTILINE)
_ROMAN_RE = re.compile(r"\b[IVXLCDM]+\b")
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F"      # emoticons
    r"\U0001F300-\U0001F5FF"       # symbols & pictographs
    r"\U0001F680-\U0001F6FF"       # transport & map
    r"\U0001F1E0-\U0001F1FF]"      # flags
)
_PLACEHOLDER_RE = re.compile(r"\[(.*?)\]")
_TITLE_RE = re.compile(r"<<.*?>>")
_SECTION_RE = re.compile(r"Secção\s+\d+", flags=re.UNICODE)
_DATE_START_RE = re.compile(r"^\s*\d{2}/\d{2}/\d{4}")
_NESTED_LIST_RE = re.compile(r"^\* .*\n\s+\* ", flags=re.MULTILINE)


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _sentences_strict(text: str) -> List[str]:
    """Divide por sentenças usando separador que preserva pontuação final."""
    parts = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts


def _sentences_simple(text: str) -> List[str]:
    """Divide por sentenças de forma simples, removendo pontuação final."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if not parts:
        # fallback simples
        parts = [s.strip() for s in _SIMPLE_SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


# ------------- Classe base -------------

class Constraint:
    type: str = ""

    def description(self, chosen_option: Any) -> str:
        raise NotImplementedError

    def check(self, response: str, chosen_option: Any) -> bool:
        raise NotImplementedError


# ------------- Implementações -------------

class MaxWordsConstraint(Constraint):
    type = "max_words"
    def description(self, n: int) -> str:
        return f"Responda com no máximo {n} palavras."
    def check(self, response: str, n: int) -> bool:
        return len(_words(response)) <= n


class MustIncludeKeywordsConstraint(Constraint):
    type = "must_include_keywords"
    def description(self, kws: List[str]) -> str:
        return f"Inclua obrigatoriamente as palavras-chave: {', '.join(kws)}."
    def check(self, response: str, kws: List[str]) -> bool:
        return all(re.search(rf"\b{re.escape(kw)}\b", response, flags=re.IGNORECASE) for kw in kws)


class ExcludeKeywordsConstraint(Constraint):
    type = "exclude_keywords"
    def description(self, kws: List[str]) -> str:
        return f"Não use as seguintes palavras: {', '.join(kws)}."
    def check(self, response: str, kws: List[str]) -> bool:
        return not any(re.search(rf"\b{re.escape(kw)}\b", response, flags=re.IGNORECASE) for kw in kws)


class ChooseFromOptionsConstraint(Constraint):
    type = "choose_from_options"
    def description(self, options: List[str]) -> str:
        return f"A resposta deve escolher exatamente uma entre: {', '.join(options)}."
    def check(self, response: str, options: List[str]) -> bool:
        return sum(bool(re.search(rf"\b{re.escape(opt)}\b", response)) for opt in options) == 1


class NthParagraphWordConstraint(Constraint):
    type = "nth_paragraph_word"
    def description(self, opt: Dict[str, Any]) -> str:
        return (f"O parágrafo {opt['paragraph_index']} deve começar com "
                f"\"{opt['start_word']}\" e terminar com \"{opt['end_word']}\".")
    def check(self, response: str, opt: Dict[str, Any]) -> bool:
        paragraphs = [p.strip() for p in response.split('---')]
        i = opt["paragraph_index"]
        if i < 0 or i >= len(paragraphs):
            return False
        p = paragraphs[i]
        return p.startswith(opt["start_word"]) and p.endswith(opt["end_word"])


class MinHighlightedSectionsConstraint(Constraint):
    type = "min_highlighted_sections"
    def description(self, n: int) -> str:
        return f"Use pelo menos {n} trechos destacados em **negrito**."
    def check(self, response: str, n: int) -> bool:
        return response.count("**") // 2 >= n


class BulletListExactConstraint(Constraint):
    type = "bullet_list_exact"
    def description(self, n: int) -> str:
        return f"Use exatamente {n} itens em lista de marcadores '*'."
    def check(self, response: str, n: int) -> bool:
        a = _BULLET_A_RE.findall(response)
        b = _BULLET_B_RE.findall(response)
        return (len(a) + len(b)) == n


class NumberOfSectionsConstraint(Constraint):
    type = "number_of_sections"
    def description(self, n: int) -> str:
        return f"Divida a resposta em exatamente {n} secções marcadas como 'Secção X'."
    def check(self, response: str, n: int) -> bool:
        return len(_SECTION_RE.findall(response)) == n


class MaxCapitalWordsConstraint(Constraint):
    type = "max_capital_words"
    def description(self, n: int) -> str:
        return f"A resposta deve conter menos de {n} palavras totalmente em MAIÚSCULAS."
    def check(self, response: str, n: int) -> bool:
        ws = _words(response)
        capitals = [w for w in ws if w.isalpha() and w.isupper()]
        return len(capitals) < n


class WrappedInQuotesConstraint(Constraint):
    type = "wrapped_in_quotes"
    def description(self, _: Any) -> str:
        return 'A resposta deve estar inteiramente envolta em aspas duplas.'
    def check(self, response: str, _: Any) -> bool:
        r = response.strip()
        return len(r) >= 2 and r.startswith('"') and r.endswith('"')


class JsonFormatConstraint(Constraint):
    type = "json_format"
    def description(self, _: Any) -> str:
        return "A resposta completa deve estar em formato JSON."
    def check(self, response: str, _: Any) -> bool:
        value = response.strip()
        
        # Remove prefixes (Python 3.8 compatible)
        for prefix in ["```json", "```Json", "```JSON", "```"]:
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        
        # Remove suffix (Python 3.8 compatible)
        if value.endswith("```"):
            value = value[:-3]
        
        value = value.strip()
        try:
            json.loads(value)
            return True
        except (ValueError, json.JSONDecodeError):
            return False


class ParagraphsExactConstraint(Constraint):
    type = "paragraphs_exact"
    def description(self, n: int) -> str:
        return f"A resposta deve conter exatamente {n} parágrafos separados por '---'."
    def check(self, response: str, n: int) -> bool:
        paragraphs = [p for p in response.split('---') if p.strip()]
        return len(paragraphs) == n


class TwoResponsesConstraint(Constraint):
    type = "two_responses"
    def description(self, _: Any) -> str:
        return "A resposta deve conter exatamente duas partes, separadas por '******'."
    def check(self, response: str, _: Any) -> bool:
        return response.count("******") == 1




class LetterFrequencyConstraint(Constraint):
    type = "letter_frequency"
    def description(self, opt: Dict[str, Any]) -> str:
        return (f"Cada frase deve conter pelo menos {opt['min_per_sentence']} "
                f"ocorrências da letra '{opt['letter']}'.")
    def check(self, response: str, opt: Dict[str, Any]) -> bool:
        letter = opt["letter"]
        try:
            min_n = int(opt["min_per_sentence"])
        except (ValueError, TypeError):
            return False
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(response) if s.strip()]
        if not sents:
            sents = [s.strip() for s in _SIMPLE_SENT_SPLIT_RE.split(response) if s.strip()]
        return all(s.count(letter) >= min_n for s in sents)


class SpecificEndingConstraint(Constraint):
    type = "specific_ending"
    def description(self, ending: str) -> str:
        return f'A resposta deve terminar exatamente com: "{ending}".'
    def check(self, response: str, ending: str) -> bool:
        return response.rstrip().endswith(ending)


class FrequencyKeywordsConstraint(Constraint):
    type = "frequency_keywords"
    def description(self, reqs: Dict[str, int]) -> str:
        parts = ", ".join(f'"{k}" {v}x' for k, v in reqs.items())
        return f"A resposta deve conter exatamente: {parts}."
    def check(self, response: str, reqs: Dict[str, int]) -> bool:
        try:
            return all(response.count(k) == int(v) for k, v in reqs.items() if v is not None)
        except (ValueError, TypeError):
            # If conversion fails, return False for safety
            return False

class EndsWithPostscriptConstraint(Constraint):
    type = "ends_with_postscript"
    def description(self, start_word: str) -> str:
        return f'A resposta deve terminar com uma pós-escrita iniciada por "{start_word}".'
    def check(self, response: str, start_word: str) -> bool:
        """
        Exige que a ÚLTIMA linha não vazia comece com o marcador (ex.: "P.P.S.", "Obs:", "Nota final:").
        Isso evita falsos positivos no meio do texto.
        """
        last = _last_nonempty_line(response)
        if not last:
            return False
        return last.startswith(start_word)


class AllLowercaseConstraint(Constraint):
    type = "all_lowercase"
    def description(self, _: Any) -> str:
        return "A resposta deve estar inteiramente em minúsculas."
    def check(self, response: str, _: Any) -> bool:
        return response == response.lower()


class TitleRepeatedConstraint(Constraint):
    type = "title_repeated"
    def description(self, count: int) -> str:
        return f"A resposta deve conter um título entre << >> repetido {count} vezes."
    def check(self, response: str, count: int) -> bool:
        try:
            return len(_TITLE_RE.findall(response)) == int(count)
        except (ValueError, TypeError):
            return False


class RepeatPromptConstraint(Constraint):
    type = "repeat_prompt"
    def description(self, _: Any) -> str:
        return "A resposta deve repetir a instrução dada antes de responder."
    def check(self, response: str, prompt_text: Optional[str]) -> bool:
        if not prompt_text:
            return False
        return response.strip().startswith(str(prompt_text).strip())


class MinPlaceholdersConstraint(Constraint):
    type = "min_placeholders"
    def description(self, opt: Dict[str, Any]) -> str:
        return (f"A resposta deve conter pelo menos {opt['count']} placeholders no formato "
                f"[opção]{' em maiúsculas' if opt.get('uppercase') else ''}.")
    def check(self, response: str, opt: Dict[str, Any]) -> bool:
        placeholders = _PLACEHOLDER_RE.findall(response)
        try:
            count = int(opt["count"])
        except (ValueError, TypeError):
            return False
        if opt.get("uppercase"):
            return sum(p.isupper() for p in placeholders) >= count
        return len(placeholders) >= count


class ExactWordsConstraint(Constraint):
    type = "exact_words"
    def description(self, n: int) -> str:
        return f"A resposta deve conter exatamente {n} palavras."
    def check(self, response: str, n: int) -> bool:
        try:
            return len(_words(response)) == int(n)
        except (ValueError, TypeError):
            return False


class ExactCharactersConstraint(Constraint):
    type = "exact_characters"
    def description(self, n: int) -> str:
        return f"A resposta deve conter exatamente {n} caracteres."
    def check(self, response: str, n: int) -> bool:
        try:
            return len(response) == int(n)
        except (ValueError, TypeError):
            return False


class ExactSentencesConstraint(Constraint):
    type = "exact_sentences"
    def description(self, n: int) -> str:
        return f"A resposta deve conter exatamente {n} frases."
    def check(self, response: str, n: int) -> bool:
        sents = _sentences_strict(response)
        if not sents:
            sents = _sentences_simple(response)
        try:
            return len(sents) == int(n)
        except (ValueError, TypeError):
            return False


class FirstWordConstraint(Constraint):
    type = "first_word"
    def description(self, word: str) -> str:
        return f'A resposta deve começar com a palavra "{word}".'
    def check(self, response: str, word: str) -> bool:
        return response.strip().startswith(word)


class LastWordConstraint(Constraint):
    type = "last_word"
    def description(self, word: str) -> str:
        return f'A resposta deve terminar com a palavra "{word}".'
    def check(self, response: str, word: str) -> bool:
        return response.strip().endswith(word)


class ContainsEmojisConstraint(Constraint):
    type = "contains_emojis"
    def description(self, n: int) -> str:
        return f"A resposta deve conter pelo menos {n} emojis."
    def check(self, response: str, n: int) -> bool:
        try:
            return len(_EMOJI_RE.findall(response)) >= int(n)
        except (ValueError, TypeError):
            return False


class ContainsNumbersConstraint(Constraint):
    type = "contains_numbers"
    def description(self, n: int) -> str:
        return f"A resposta deve conter exatamente {n} números."
    def check(self, response: str, n: int) -> bool:
        nums = re.findall(r"\d+", response)
        try:
            return len(nums) == int(n)
        except (ValueError, TypeError):
            return False


class NumbersIncreasingConstraint(Constraint):
    type = "numbers_increasing"
    def description(self, _: Any) -> str:
        return "Todos os números na resposta devem aparecer em ordem crescente."
    def check(self, response: str, _: Any) -> bool:
        try:
            nums = [int(x) for x in re.findall(r"\d+", response)]
            return all(nums[i] < nums[i+1] for i in range(len(nums)-1))
        except (ValueError, TypeError):
            return False




class AcrosticConstraint(Constraint):
    type = "acrostic"
    def description(self, word: str) -> str:
        return f'Cada frase deve começar com letras que formam a palavra "{word}".'
    def check(self, response: str, word: str) -> bool:
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(response) if s.strip()]
        if not sents:
            sents = [s.strip() for s in _SIMPLE_SENT_SPLIT_RE.split(response) if s.strip()]
        if len(sents) < len(word):
            return False
        # Verifica as primeiras len(word) frases
        for i, ch in enumerate(word):
            if not sents[i]:
                return False
            if sents[i][0].lower() != ch.lower():
                return False
        return True


class PalindromeSentenceConstraint(Constraint):
    type = "palindrome_sentence"
    def description(self, _: Any) -> str:
        return "A resposta deve conter pelo menos uma frase palíndroma."
    def check(self, response: str, _: Any) -> bool:
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(response) if s.strip()]
        if not sents:
            sents = [s.strip() for s in _SIMPLE_SENT_SPLIT_RE.split(response) if s.strip()]
        for s in sents:
            cleaned = re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ]", "", s.lower())
            if cleaned and cleaned == cleaned[::-1]:
                return True
        return False


class DialogueTwoSpeakersConstraint(Constraint):
    type = "dialogue_two_speakers"
    def description(self, _: Any) -> str:
        return "A resposta deve estar em formato de diálogo entre duas pessoas."
    def check(self, response: str, _: Any) -> bool:
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        speakers = [l.split(":")[0] for l in lines if ":" in l]
        uniq = [s for s in set(speakers) if s]
        return len(uniq) == 2


class TableFormatConstraint(Constraint):
    type = "table_format"
    def description(self, n: int) -> str:
        return f"A resposta deve conter uma tabela em Markdown com {n} linhas."
    def check(self, response: str, n: int) -> bool:
        rows = [l for l in response.splitlines() if "|" in l]
        try:
            return len(rows) == int(n)
        except (ValueError, TypeError):
            return False


class RomanNumeralsConstraint(Constraint):
    type = "roman_numerals"
    def description(self, n: int) -> str:
        return f"A resposta deve conter pelo menos {n} números em algarismos romanos."
    def check(self, response: str, n: int) -> bool:
        try:
            return len(_ROMAN_RE.findall(response)) >= int(n)
        except (ValueError, TypeError):
            return False


class MarkdownCodeBlockConstraint(Constraint):
    type = "markdown_code_block"
    def description(self, _: Any) -> str:
        return "A resposta deve conter um bloco de código em Markdown."
    def check(self, response: str, _: Any) -> bool:
        return "```" in response


class NestedListConstraint(Constraint):
    type = "nested_list"
    def description(self, _: Any) -> str:
        return "A resposta deve conter pelo menos uma lista aninhada em Markdown."
    def check(self, response: str, _: Any) -> bool:
        return bool(_NESTED_LIST_RE.search(response))


class SameWordCountSentencesConstraint(Constraint):
    type = "same_word_count_sentences"
    def description(self, n: int) -> str:
        return f"Cada frase deve ter exatamente {n} palavras."
    def check(self, response: str, n: int) -> bool:
        sents = [s.strip() for s in _SENT_SPLIT_RE.split(response) if s.strip()]
        if not sents:
            sents = [s.strip() for s in _SIMPLE_SENT_SPLIT_RE.split(response) if s.strip()]
        try:
            return all(len(_words(s)) == int(n) for s in sents)
        except (ValueError, TypeError):
            return False


class StartsWithDateConstraint(Constraint):
    type = "starts_with_date"
    def description(self, _: Any) -> str:
        return "A resposta deve começar com uma data no formato DD/MM/AAAA."
    def check(self, response: str, _: Any) -> bool:
        return bool(_DATE_START_RE.match(response))


# ------------- Registro e utilitários -------------
# near the bottom of pt_instructions.py

CONSTRAINT_CLASSES = [
    MaxWordsConstraint,
    MustIncludeKeywordsConstraint,
    ExcludeKeywordsConstraint,
    ChooseFromOptionsConstraint,
    NthParagraphWordConstraint,
    MinHighlightedSectionsConstraint,
    BulletListExactConstraint,
    NumberOfSectionsConstraint,
    MaxCapitalWordsConstraint,
    WrappedInQuotesConstraint,
    JsonFormatConstraint,
    ParagraphsExactConstraint,
    TwoResponsesConstraint,
    LetterFrequencyConstraint,
    SpecificEndingConstraint,
    FrequencyKeywordsConstraint,
    EndsWithPostscriptConstraint,
    AllLowercaseConstraint,
    TitleRepeatedConstraint,
    RepeatPromptConstraint,
    MinPlaceholdersConstraint,
    ExactWordsConstraint,
    ExactCharactersConstraint,
    ExactSentencesConstraint,
    FirstWordConstraint,
    LastWordConstraint,
    ContainsEmojisConstraint,
    ContainsNumbersConstraint,
    NumbersIncreasingConstraint,
    AcrosticConstraint,
    PalindromeSentenceConstraint,
    DialogueTwoSpeakersConstraint,
    TableFormatConstraint,
    RomanNumeralsConstraint,
    MarkdownCodeBlockConstraint,
    NestedListConstraint,
    SameWordCountSentencesConstraint,
    StartsWithDateConstraint,
]

CONSTRAINT_REGISTRY = {cls.type: cls() for cls in CONSTRAINT_CLASSES}


def generate_description(constraint_type: str, chosen_option: Any) -> str:
    """Gera a descrição em PT para a restrição informada."""
    if constraint_type not in CONSTRAINT_REGISTRY:
        return f"Aplique a restrição: {constraint_type}."
    return CONSTRAINT_REGISTRY[constraint_type].description(chosen_option)


def generate_description_new_format(instruction_id: str, kwargs: Dict[str, Any]) -> str:
    """
    Gera a descrição em PT para a restrição no novo formato.
    
    Args:
        instruction_id: ID da instrução (e.g., "pt:first_word")
        kwargs: Parâmetros da restrição
    
    Returns:
        String com a descrição da restrição
    """
    # Extract constraint type from instruction_id
    if instruction_id.startswith("pt:"):
        constraint_type = instruction_id[3:]
    else:
        constraint_type = instruction_id
    
    # Convert kwargs to chosen_option format
    chosen_option = _kwargs_to_chosen_option(constraint_type, kwargs)
    
    # Generate description using existing function
    return generate_description(constraint_type, chosen_option)


def verify_response(response: str, constraints: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Valida uma resposta contra uma lista de constraints no formato:
      [{"id": "...", "type": "...", "chosen_option": ...}, ...]
    Retorna dict {constraint_id: bool}
    """
    results: Dict[str, bool] = {}
    for c in constraints:
        cid = c.get("id")
        ctype = c.get("type")
        opt = c.get("chosen_option")
        inst = CONSTRAINT_REGISTRY.get(ctype)
        results[cid] = bool(inst and inst.check(response, opt))
    return results


def verify_response_new_format(response: str, instruction_id_list: List[str], kwargs_list: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Valida uma resposta contra constraints no novo formato:
      instruction_id_list: ["pt:first_word", "pt:json_format", ...]
      kwargs_list: [{"word": "Era"}, {}, ...]
    Retorna dict {instruction_id: bool}
    """
    results: Dict[str, bool] = {}
    
    # Handle None values
    if not instruction_id_list or not kwargs_list:
        return results
    
    # Handle case where the lists might be nested
    if instruction_id_list and isinstance(instruction_id_list[0], list):
        instruction_id_list = instruction_id_list[0]
    
    if kwargs_list and isinstance(kwargs_list[0], list):
        kwargs_list = kwargs_list[0]
    
    # Ensure both lists are still valid after flattening
    if not instruction_id_list or not kwargs_list:
        return results
    
    for i, (instruction_id, kwargs) in enumerate(zip(instruction_id_list, kwargs_list)):
        # Extract constraint type from instruction_id (remove "pt:" prefix)
        if instruction_id.startswith("pt:"):
            ctype = instruction_id[3:]  # Remove "pt:" prefix
        else:
            ctype = instruction_id
        
        # Get the constraint instance
        inst = CONSTRAINT_REGISTRY.get(ctype)
        
        if inst:
            # Convert kwargs to chosen_option format
            chosen_option = _kwargs_to_chosen_option(ctype, kwargs)
            results[instruction_id] = inst.check(response, chosen_option)
        else:
            results[instruction_id] = False
    
    return results


def verify_response_unified(response: str, constraints_data: Dict[str, Any]) -> Dict[str, bool]:
    """
    Unified verification function that handles both old and new formats.
    
    Args:
        response: The response text to verify
        constraints_data: Either:
            - {"constraints_used": [{"id": "...", "type": "...", "chosen_option": ...}, ...]}
            - {"instruction_id_list": [...], "kwargs": [...]}
            - {"instruction_id_list": [...], "kwargs_pt": [...]}
    
    Returns:
        Dictionary mapping constraint IDs to verification results
    """
    if "constraints_used" in constraints_data:
        # Old format
        return verify_response(response, constraints_data["constraints_used"])
    elif "instruction_id_list" in constraints_data and ("kwargs" in constraints_data or "kwargs_pt" in constraints_data):
        # New format - handle both 'kwargs' and 'kwargs_pt' keys
        kwargs_key = "kwargs" if "kwargs" in constraints_data else "kwargs_pt"
        return verify_response_new_format(
            response, 
            constraints_data["instruction_id_list"], 
            constraints_data[kwargs_key]
        )
    else:
        # Unknown format
        return {}


def _kwargs_to_chosen_option(constraint_type: str, kwargs: Dict[str, Any]) -> Any:
    """
    Convert kwargs dictionary to chosen_option format for a given constraint type.
    
    Args:
        constraint_type: The type of constraint (e.g., "first_word", "json_format")
        kwargs: The kwargs dictionary
    
    Returns:
        The chosen_option value in the original format
    """
    if not kwargs:
        # For constraints that don't need parameters, return appropriate default
        if constraint_type in ["json_format", "wrapped_in_quotes", "two_responses", 
                             "all_lowercase", "numbers_increasing", "palindrome_sentence", 
                             "dialogue_two_speakers", "markdown_code_block", "nested_list", 
                             "starts_with_date"]:
            return True
        return None
    
    # Map common parameter patterns
    param_mappings = {
        "max_words": "max_count",
        "must_include_keywords": "keywords",
        "exclude_keywords": "keywords", 
        "choose_from_options": "options",
        "nth_paragraph_word": "paragraph_info",
        "min_highlighted_sections": "num_highlights",
        "bullet_list_exact": "num_bullets",
        "number_of_sections": "num_sections",
        "max_capital_words": "max_count",
        "paragraphs_exact": "num_paragraphs",
        "letter_frequency": "letter_info",
        "specific_ending": "ending",
        "frequency_keywords": "keyword_freq",
        "ends_with_postscript": "postscript",
        "title_repeated": "num_repetitions",
        "repeat_prompt": "prompt_text",
        "min_placeholders": "placeholder_info",
        "exact_words": "word_count",
        "exact_characters": "char_count",
        "exact_sentences": "sentence_count",
        "first_word": "word",
        "last_word": "word",
        "contains_emojis": "num_emojis",
        "contains_numbers": "num_numbers",
        "acrostic": "target_word",
        "table_format": "num_columns",
        "roman_numerals": "num_numerals",
        "same_word_count_sentences": "word_count",
    }
    
    expected_param = param_mappings.get(constraint_type)
    
    if expected_param and expected_param in kwargs:
        return kwargs[expected_param]
    
    # Fallback: return the first value or the whole dict if it's complex
    if len(kwargs) == 1:
        return next(iter(kwargs.values()))
    else:
        return kwargs

# ---------------- Option pools and samplers ----------------

import random

# For constraints that require a chosen_option, define canonical pools/generators here.
# For constraints without options, we return sensible defaults (e.g., True or None).

_OPTION_POOLS = {
    "max_words": [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500],
    "must_include_keywords": [
        ["queijo", "pão"],
        ["lua", "jardim", "amor"],
        ["matemática", "ciência", "dados"],
        ["música", "dança", "arte"],
        ["café", "livro", "manhã"],
        ["oceano", "vento", "liberdade"],
        ["tecnologia", "futuro", "inovação"],
        ["família", "casa", "felicidade"],
        ["estrela", "sonho", "esperança"],
        ["montanha", "aventura", "coragem"],
        ["terra", "sol", "flor", "água"],
        ["cão", "gato", "rato", "queijo"],
        ["galinha", "galo", "grão", "manhã"],
        ["camisola", "calças"],
        ["carro", "sapato"],
        ["leite", "caneca", "café", "sal"],
        ["girafa", "nuvem", "terra"],
        ["livro", "letra", "vogal"],
        ["palhaço", "circo", "malabarismo", "leão", "palco"],
        ["fantasma", "computador", "cursor"],
        ["fruta", "fresca", "bens perecíveis", "importação", "banana", "abacate"],
        ["oceanografia", "baleias", "golfinhos", "orcas", "mamíferos"],
        ["música", "violoncelo", "corda", "arco", "resina"],
        ["lápis", "caneta", "folha"],
        ["portas", "paredes", "mesas"],
        ["água", "rio", "lago"],
        ["telemóvel", "chamada", "mensagem"],
        ["dicionário", "enciclopédia", "gramática"],
        ["jardim", "flores", "rosas", "orquídeas"],
        ["gelo", "patins", "patinagem", "competição"],
        ["montanha", "neve", "esqui", "campeonato"],
        ["acentuação", "métrica", "rima", "estrutura", "verso"],
        ["partitura", "pausa", "nota"],
        ["filme", "ecrã", "cinema"],
        ["mochila", "costas", "bolsa"],
        ["limpeza", "vassoura", "pó"],
        ["quadro", "giz", "professor", "aluno"],
        ["viagem", "turista", "avião", "museu"],
        ["presidente", "deputado", "lei"],
        ["café", "convívio", "amigos"],
        ["filósofo", "doutrina", "conceito", "raciocínio"],
        ["preto", "branco", "cinzento"],
        ["fogo", "gelo", "equilíbrio"],
        ["ponte", "rio", "travessia"],
        ["borboleta", "flor", "primavera"],
        ["relâmpago", "trovão", "tempestade"],
        ["diamante", "brilho", "precioso"],
        ["floresta", "árvore", "vida selvagem"],
        ["quebra-cabeças", "peça", "solução"],
        ["arco-íris", "cores", "céu"],
        ["jornada", "destino", "caminho"],
        ["sabedoria", "conhecimento", "aprendizagem"],
    ],
    "exclude_keywords": [
        ["barato", "rápido"],
        ["triste", "ódio"],
        ["perigoso", "ilegal"],
        ["feio", "sujo"],
        ["difícil", "impossível"],
        ["doente", "fraco"],
        ["antigo", "obsoleto"],
        ["falso", "mentira"],
        ["guerra", "violência"],
        ["pobre", "miserável"],
        ["primeiro", "um", "número"],
        ["mas", "porém", "no entanto", "apesar de"],
        ["como", "porque"],
        ["sim", "talvez", "não"],
        ["sempre", "nunca", "raramente"],
        ["e", "ou"],
        ["lamento", "desculpe", "desculpa", "não", "modelo"],
        ["por", "apesar", "tal"],
        ["mau", "bom", "melhor", "pior"],
        ["partido", "danificado", "falhado"],
        ["confuso", "perdido", "incerto"],
        ["aborrecido", "maçador", "tedioso"],
        ["zangado", "furioso", "raiva"],
        ["assustador", "aterrorizante", "medonho"],
        ["tóxico", "prejudicial", "venenoso"],
        ["preguiçoso", "inativo", "parado"],
        ["estúpido", "burro", "tolo"],
        ["estranho", "bizarro", "esquisito"],
        ["falso", "artificial", "sintético"],
        ["vazio", "vácuo", "oco"],
        ["cruel", "malvado", "severo"],
    ],
    "choose_from_options": [
        ["sim", "não", "talvez"],
        ["6 horas", "6,6 horas", "depende"],
        ["grelhar", "forno", "micro-ondas"],
        ["vermelho", "azul", "verde", "amarelo"],
        ["carro", "bicicleta", "a pé", "transporte público"],
        ["verão", "inverno", "primavera", "outono"],
        ["café", "chá", "água", "sumo"],
        ["cinema", "teatro", "concerto", "museu"],
        ["praia", "montanha", "cidade", "campo"],
        ["português", "inglês", "espanhol", "francês"],
    ],
    "nth_paragraph_word": [
        {"paragraph_index": 1, "start_word": "Primeiro", "end_word": "situação"},
        {"paragraph_index": 2, "start_word": "No entanto", "end_word": "comportamento"},
        {"paragraph_index": 3, "start_word": "Porém", "end_word": "verdade"},
        {"paragraph_index": 4, "start_word": "Assim", "end_word": "fim"},
        {"paragraph_index": 5, "start_word": "Finalmente", "end_word": "conclusão"},
        {"paragraph_index": 1, "start_word": "Inicialmente", "end_word": "processo"},
        {"paragraph_index": 2, "start_word": "Posteriormente", "end_word": "resultado"},
        {"paragraph_index": 3, "start_word": "Consequentemente", "end_word": "efeito"},
    ],
    "min_highlighted_sections": [1, 2, 3, 4, 5, 6, 7, 8],
    "bullet_list_exact": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
    "number_of_sections": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "max_capital_words": [1, 2, 3, 4, 5, 6, 7, 8, 10],
    "wrapped_in_quotes": [True],
    "json_format": [True],
    "paragraphs_exact": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
    "two_responses": [True],
    "letter_frequency": "generate_random",  # Special marker for dynamic generation
    "specific_ending": [
        "Obrigado.",
        "Atenciosamente.",
        "Fico à disposição para esclarecer dúvidas.",
        "Espero que isto ajude.",
        "Se precisar de mais informações, pergunte.",
        "Resposta concluída.",
        "Em conclusão, o ponto principal está acima.",
        "Em suma, esse é o resultado.",
        "Terminei a resposta.",
        "Foi um prazer ajudar."
    ],
    "frequency_keywords": [
        {"lua": 3, "jardim": 2},
        {"lua": 5, "jardim": 3},
        {"lua": 7, "jardim": 4},
        {"amor": 2, "vida": 1},
        {"amor": 4, "vida": 2},
        {"amor": 6, "vida": 3},
        {"computador": 2, "internet": 3},
        {"computador": 4, "internet": 5},
        {"sol": 3, "céu": 2},
        {"mar": 4, "onda": 3},
    ],
    "ends_with_postscript": ["P.S.", "P.P.S.", "Obs:", "Nota final:", "Adendo:", "Lembre-se:", "Atenção:", "Importante:"],
    "all_lowercase": [True],
    "title_repeated": [1, 2, 3, 4, 5],
    "repeat_prompt": ["use all these constraints:", "siga todas estas regras:", "aplique todos os requisitos:", "respeite todas as limitações:"],
    "min_placeholders": [
        {"count": 1, "uppercase": False},
        {"count": 2, "uppercase": False},
        {"count": 3, "uppercase": False},
        {"count": 4, "uppercase": False},
        {"count": 5, "uppercase": False},
        {"count": 2, "uppercase": True},
        {"count": 3, "uppercase": True},
        {"count": 5, "uppercase": True},
        {"count": 7, "uppercase": True},
        {"count": 10, "uppercase": True},
    ],
    "exact_words": [25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200, 250],
    "exact_characters": [100, 140, 200, 280, 350, 500, 600, 750, 1000, 1200],
    "exact_sentences": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    "first_word": ["Era", "Portanto", "Afinal", "Hoje", "Ontem", "Sempre", "Nunca", "Talvez", "Certamente", "Obviamente"],
    "last_word": ["fim", "verdade", "adeus", "sempre", "jamais", "eternamente", "completamente", "definitivamente", "absolutamente", "totalmente"],
    "contains_emojis": [1, 2, 3, 4, 5, 6, 7, 8, 10],
    "contains_numbers": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
    "numbers_increasing": [True],
    "acrostic": ["PAZ", "AMOR", "SOL", "VIDA", "CASA", "HOPE", "ALEGRIA", "SONHO", "TERRA", "FOGO"],
    "palindrome_sentence": [True],
    "dialogue_two_speakers": [True],
    "table_format": [2, 3, 4, 5, 6, 7, 8],
    "roman_numerals": [1, 2, 3, 4, 5, 6, 7, 8],
    "markdown_code_block": [True],
    "nested_list": [True],
    "same_word_count_sentences": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    "starts_with_date": [True],
}

def sample_option(ctype: str):
    """
    Returns a valid chosen_option for a given constraint type.
    The pools are centralized here so every consumer (prompt generator,
    synthetic data builders, etc.) is consistent.
    """
    pool = _OPTION_POOLS.get(ctype)
    if not pool:
        # No option needed or unknown type → return None
        return None
    
    # Special handling for dynamic generation
    if pool == "generate_random":
        if ctype == "letter_frequency":
            # Generate random letter from Portuguese alphabet and random number 1-10
            portuguese_letters = "abcdefghijklmnopqrstuvwxyz"
            return {
                "letter": random.choice(portuguese_letters),
                "min_per_sentence": random.randint(1, 10)
            }
    
    return random.choice(pool)
