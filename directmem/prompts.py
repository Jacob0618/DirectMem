"""Prompts used by DirectMem in the paper, reproduced verbatim.

* U2_TEMPLATE: the LoCoMo answer prompt released with the Mem0 evaluation code (mem0ai/memory-benchmarks, commit
  4b61c5d, benchmarks/locomo/prompts.py, Apache License 2.0), with one sentence of Step 7 replaced to allow abstention:
  'NEVER say "not specified", "not mentioned", "no record", or "the memories don\'t say" — if ANY memory contains relevant information, give the best answer from available evidence.'
  -> 'If the memories do not contain the answer, answer "not mentioned".'
  Used for LoCoMo and SocialMemBench.
* lme_answer_prompt: the LongMemEval-S answer prompt.
* locomo_judge_prompt: the correct-or-wrong judge used for LoCoMo and SocialMemBench (adapted from the Mem0 evaluation).
* get_anscheck_prompt: the official per-type LongMemEval judge prompts (xiaowu0162/LongMemEval, MIT License).
"""
from datetime import datetime as _datetime

U2_TEMPLATE = 'You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.\n\n## Step 1: SCAN ALL MEMORIES\nRead EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.\n\n## Step 2: ENTITY VERIFICATION\nConfirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers\' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.\n\n## Step 3: COMBINE AND CROSS-REFERENCE\n- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.\n- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.\n- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.\n- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.\n- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.\n\n## Step 4: SELECT THE BEST ANSWER\n- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate\'s relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.\n- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.\n- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.\n- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.\n- Photos depict what was IN the photo, not facts about someone\'s daily life. Prefer direct statements over photo descriptions for inferences.\n- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.\n\n## Step 5: TEMPORAL GROUNDING\nThese conversations took place around {reference_date}. All events occurred in 2022-2024.\n- Calculate time relative to this date, NOT today. Never output 2025 or 2026.\n- Use dates explicitly stated in memory text. Do not invent or estimate dates.\n- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.\n- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.\n- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.\n\n## Step 6: INCLUSION CHECK (for lists and counts)\nIf you found items during reasoning that you\'re tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.\n- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven\'t missed items from memories late in the list.\n- The question assumes something happened. Find WHAT happened, don\'t say nothing happened.\n\n## Step 7: COMMIT AND ANSWER\nGive a direct, specific answer. If the memories do not contain the answer, answer "not mentioned". No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.\n- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.\n- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):\n  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.\n  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."\n  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.\n  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."\n\n# Instructions\n\n## Misc\n\n1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people\n2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").\n3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.\n\n{memories}\n\nQuestion: {question}\n\nWork through Steps 1-7, then give your final answer after "ANSWER:".\n'

MEMORY_HEADER = "The following memories are presented in chronological order (oldest to newest)."
ABSTAIN = "not mentioned"


def to_human_date(iso_str):
    """== _to_human_date of the Mem0 evaluation code. With the '+00:00' ISO strings used here, every format fails and
    the function falls back to the date portion ('2023-05-08'); the paper's runs rendered dates this way."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _datetime.strptime(iso_str[:26].rstrip("Z"), fmt.replace("%z", "")).strftime("%A, %B %d, %Y")
        except ValueError:
            continue
    return iso_str[:10]


def render_memories(mems):
    """mems: [(iso_date, text)] in chronological order -> the memory block of the Mem0 answer prompt."""
    lines = [MEMORY_HEADER, ""]
    for iso, text in sorted(mems, key=lambda m: m[0]):  # stable: turns of one session keep their order
        lines.append(f"({to_human_date(iso)}) {text}")
    return "\n".join(lines)


def u2_prompt(question, reference_date, mems):
    return U2_TEMPLATE.format(memories=render_memories(mems), question=question, reference_date=reference_date)


def u2_parse(raw):
    """Official parser of the Mem0 evaluation code: the answer follows the last 'ANSWER:'."""
    return raw.rsplit("ANSWER:", 1)[-1].strip() if "ANSWER:" in raw else raw


def is_abstain_text(a):
    a = str(a or "").strip().lower().strip(".! ")
    return a.startswith(ABSTAIN) or a in ("unknown", "not stated", "not specified")


def locomo_judge_prompt(q, gen):
    return ("Your task is to label an answer to a question as CORRECT or WRONG. You are given (1) a question about a "
            "conversation between two people, (2) a gold (ground-truth) answer, and (3) a generated answer.\n"
            "The gold answer is usually short. The generated answer may be longer or phrased differently.\n"
            "Be generous with wording: if the generated answer refers to the same fact, entity, event or topic as the "
            "gold answer, label it CORRECT. For time questions, accept any format or granularity that refers to the "
            "same date or period (e.g. '7 May 2023' vs 'May 7th', 'the week before 9 June 2023' vs '2 June 2023'). "
            "For list answers, CORRECT if it captures the main items of the gold answer without contradicting it.\n"
            "Be strict on facts: a different person, object, place, number, date or period, or an answer saying the "
            "information is not mentioned / unknown, is WRONG.\n\n"
            f"Question: {q['question']}\nGold answer: {q['answer']}\nGenerated answer: {gen}\n\n"
            "Return JSON {\"reason\": one short sentence, \"label\": \"CORRECT\" or \"WRONG\"}.")


LOCOMO_JUDGE_SCHEMA = {"type": "OBJECT", "properties": {"reason": {"type": "STRING"},
                                                        "label": {"type": "STRING", "enum": ["CORRECT", "WRONG"]}},
                       "required": ["reason", "label"]}


def lme_answer_prompt(q, lines):
    mem = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(lines))
    return ("You are a personal assistant with long-term memory of your past chats with the user. Below are MEMORIES: "
            "rounds from earlier chat sessions (a user message and your reply), retrieved for this question, each "
            f"prefixed with the date of its session. Today is {q['question_date']}.\n\n"
            f"MEMORIES:\n{mem}\n\nUSER QUESTION: {q['question']}\n\n"
            "Instructions:\n"
            "- Answer from the memories. Work out dates, durations, counts and orderings from the session dates and "
            "today's date.\n"
            "- If memories give different values for the same thing, the more recent one is the current value.\n"
            "- If the question asks for recommendations or advice, give a short personalised answer (2-4 sentences) "
            "that uses what the memories say about the user. Otherwise answer briefly (a few words or one sentence).\n"
            "- If the memories do not contain the information needed to answer, say \"I don't know\" and briefly say "
            "what is missing; do not guess.\n"
            'Return JSON {"answer": string, "unknown": boolean = true if you said you don\'t know}.')


LME_ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "unknown": {"type": "boolean"}},
                     "required": ["answer", "unknown"]}


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


LME_JUDGE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}},
                    "required": ["verdict"]}
