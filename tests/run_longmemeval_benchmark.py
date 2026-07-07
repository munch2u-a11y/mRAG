import os
import json
import logging
import re
import shutil
import random
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from mrag.memory.belief_store import BeliefStore
from mrag.core.belief_consolidator import BeliefConsolidator
from mrag.core.token_counting import count_text_tokens, describe_token_counter
from mrag.core.vector_store import create_vector_store
from mrag import PreGenerativeInjector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mrag.longmemeval_benchmark")

# LongMemEval dates look like "2023/05/20 (Sat) 02:21" -- convert to the same
# "YYYY-MM-DD HH:MM" prefix convention used everywhere else in mrag.
_LME_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2}) \([A-Za-z]+\) (\d{2}):(\d{2})$")


def _lme_date_to_timestamp(date_str: str) -> str:
    m = _LME_DATE_RE.match(date_str.strip())
    if not m:
        return date_str
    y, mo, d, h, mi = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}"


def _select_longmemeval_instances(dataset):
    profile = os.environ.get("MRAG_LME_PROFILE", "quick").strip().lower()
    q_indices_env = os.environ.get("MRAG_LME_Q_INDICES")
    max_questions = os.environ.get("MRAG_LME_MAX_QUESTIONS", "3")
    q_start = int(os.environ.get("MRAG_LME_Q_START", "0"))

    if q_indices_env:
        indices = [int(i) for i in q_indices_env.split(",") if i.strip() != ""]
        logger.info(f"Scoped to explicit question indices {indices} via MRAG_LME_Q_INDICES.")
        return [(idx, dataset[idx]) for idx in indices], "indices"

    if profile == "full":
        logger.info("Using full LongMemEval_S dataset via MRAG_LME_PROFILE=full.")
        return list(enumerate(dataset)), "full"

    if profile in {"long", "longer", "readme"}:
        sample_size = int(os.environ.get("MRAG_LME_STRATIFIED_COUNT", "20"))
        seed = int(os.environ.get("MRAG_LME_STRATIFY_SEED", "7"))
        rng = random.Random(seed)
        buckets = {}
        for idx, inst in enumerate(dataset):
            key = (inst.get("question_type", "unknown"), inst.get("question_id", "").endswith("_abs"))
            buckets.setdefault(key, []).append((idx, inst))
        for items in buckets.values():
            rng.shuffle(items)

        ordered_keys = sorted(buckets.keys(), key=lambda item: (item[1], item[0]))
        selected = []
        while len(selected) < sample_size and any(buckets.values()):
            for key in ordered_keys:
                if buckets[key] and len(selected) < sample_size:
                    selected.append(buckets[key].pop())

        logger.info(
            "Using deterministic stratified LongMemEval_S sample via MRAG_LME_PROFILE=%s "
            "(count=%s, seed=%s).",
            profile,
            sample_size,
            seed,
        )
        return selected, profile

    scoped = list(enumerate(dataset))[q_start: q_start + int(max_questions)]
    logger.info(f"Scoped to questions [{q_start}:{q_start + int(max_questions)}) via MRAG_LME_Q_START/MAX_QUESTIONS.")
    return scoped, "quick"


def main():
    # Standard dotenv convention: searches for a .env file in the current/
    # parent directories (see .env.example). An optional override path can
    # be set via MRAG_CREDENTIALS_ENV for setups that keep credentials
    # outside the repo tree.
    load_dotenv()
    extra_env = os.environ.get("MRAG_CREDENTIALS_ENV")
    if extra_env:
        load_dotenv(extra_env, override=False)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found (set it in .env or export it directly)")
        return

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")

    def llm_callable(prompt: str) -> str:
        response = gemini_model.generate_content(prompt)
        return response.text

    logger.info("Loading LongMemEval_S dataset...")
    data_path = os.environ.get(
        "MRAG_LME_DATA_PATH",
        "/home/nemo/MicroRAG/longmemeval_data/data/longmemeval_s_cleaned.json",
    )
    with open(data_path) as f:
        dataset = json.load(f)
    dataset, profile = _select_longmemeval_instances(dataset)
    token_counter = describe_token_counter()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    date_slug = timestamp[:10]

    total_correct = 0
    total_questions = 0
    all_injected_tokens = []
    qa_details = []

    for position, (q_idx, inst) in enumerate(dataset):
        logger.info("==================================================")
        logger.info(f"Evaluating Question {q_idx} ({position + 1}/{len(dataset)} in this run)...")
        logger.info(f"Question type: {inst['question_type']}  (id={inst['question_id']})")
        logger.info("==================================================")

        db_dir = f"./longmemeval_mrag_data_q_{q_idx}"
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)

        belief_store = BeliefStore(data_dir=db_dir)
        vector_store = create_vector_store("chromadb", persist_dir=os.path.join(db_dir, "chroma"))

        consolidator = BeliefConsolidator(
            belief_store=belief_store,
            llm_callable=llm_callable,
            context_limit=8192,
            ratio=0.40,
            vector_store=vector_store,
            enable_session_synthesis=not os.environ.get("MRAG_LME_DISABLE_SESSION_SYNTHESIS"),
        )

        sessions = inst["haystack_sessions"]
        dates = inst["haystack_dates"]

        for s_idx, (session, date_str) in enumerate(zip(sessions, dates)):
            date_prefix = _lme_date_to_timestamp(date_str)
            lines = []
            for turn in session:
                role = turn.get("role", "user").upper()
                content = turn.get("content", "")
                lines.append(f"{role}: {content}")
            session_text = f"Date: {date_prefix}\n" + "\n".join(lines)
            logger.info(f"Ingesting session {s_idx + 1}/{len(sessions)} ({len(lines)} turns)...")
            consolidator.add_conversation_turn({"content": session_text})

        if consolidator._backlog:
            consolidator.run_consolidation_pass(consolidator._backlog)
            consolidator._backlog = []
            consolidator._backlog_tokens = 0

        logger.info(f"Ingestion complete. Total beliefs in store: {len(belief_store.get_all_beliefs_flat())}")

        if not os.environ.get("MRAG_LME_SKIP_CLUSTER_DISCOVERY"):
            try:
                cluster_stats = consolidator.discover_and_consolidate_clusters(min_cluster_size=8)
                logger.info(f"Structural cluster discovery: {cluster_stats}")
            except ImportError as e:
                logger.warning(f"Skipping structural cluster discovery (scikit-learn not installed): {e}")

        injector = PreGenerativeInjector(
            belief_store=belief_store,
            vector_store=vector_store,
            top_k_candidates=30,
            blacklist_memory_size=30,
        )

        question = inst["question"]
        expected = inst["answer"]
        qtype = inst["question_type"]
        is_abstention = inst["question_id"].endswith("_abs")

        injector.clear_blacklist()
        beliefs_context = injector.inject(question)

        injected_tokens = count_text_tokens(beliefs_context) if beliefs_context else 0
        all_injected_tokens.append(injected_tokens)

        qa_prompt = f"""You are a helpful assistant answering questions about a long-term conversation history.
Using the following beliefs/context extracted from the conversation, write a short, concise answer to the question.
If the beliefs do not contain the information needed to answer, say so plainly rather than guessing.
You should make logical deductions or draw direct implications from the retrieved beliefs to answer the question accurately (since the beliefs are summarized representations).

{beliefs_context}

Question: {question}
Short answer:"""

        try:
            response_text = llm_callable(qa_prompt).strip()
        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            response_text = "Error generating answer"

        logger.info(f"\n--- Q{q_idx} [{qtype}{' ABSTENTION' if is_abstention else ''}] ---")
        logger.info(f"Question: {question}")
        logger.info(f"Retrieved Beliefs:\n{beliefs_context}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Micro-RAG + Gemini: {response_text}")

        grade_prompt = f"""You are an objective grader evaluating a QA system's answer against the expected ground truth answer.
Question: {question}
Expected Ground Truth: {expected}
QA System's Answer: {response_text}

Determine if the QA system's answer is semantically correct, accurate, or equivalent to the expected ground truth answer.
Criteria:
- Be highly lenient: if the QA system's answer captures the essential correct fact, mark it as YES.
- Allow minor variations, synonyms, abbreviations, and numerical representations (e.g. "10 years" vs "ten years").
- Allow partial matches if the core correct concept is present.
- If the expected ground truth uses relative dates and the QA system resolves it to the correct calendar date (or vice versa), mark it as YES.
- If the expected ground truth indicates the information was never mentioned or is unavailable, mark YES only if the QA system's answer similarly declines to answer or indicates the info isn't present, rather than fabricating a confident answer.
- Only output "YES" if it is semantically correct/accurate, otherwise output "NO". Do not write any preamble, explanation, or punctuation."""

        try:
            grade_resp = llm_callable(grade_prompt).strip().upper()
            is_correct = "YES" in grade_resp
        except Exception as e:
            logger.warning(f"Grading API call failed: {e}")
            is_correct = str(expected).lower() in response_text.lower()

        if is_correct:
            total_correct += 1
            logger.info("Result: CORRECT")
        else:
            logger.info("Result: MISSED")

        total_questions += 1
        qa_details.append({
            "q_idx": q_idx,
            "question_type": qtype,
            "is_abstention": is_abstention,
            "question": question,
            "expected": expected,
            "actual": response_text,
            "status": "CORRECT" if is_correct else "MISSED",
            "injected_tokens": injected_tokens,
        })

        if os.environ.get("MRAG_LME_SKIP_CLEANUP"):
            logger.info(f"Skipping cleanup of {db_dir} (MRAG_LME_SKIP_CLEANUP set).")
        else:
            try:
                shutil.rmtree(db_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up {db_dir}: {e}")

    overall_accuracy = total_correct / total_questions if total_questions else 0
    avg_tokens = sum(all_injected_tokens) / len(all_injected_tokens) if all_injected_tokens else 0
    max_tokens = max(all_injected_tokens) if all_injected_tokens else 0
    min_tokens = min(all_injected_tokens) if all_injected_tokens else 0

    logger.info("\n================ OVERALL BENCHMARK RESULTS ================")
    logger.info(f"Total evaluated: {total_questions}")
    logger.info(f"Correct/Matched: {total_correct}")
    logger.info(f"Overall Accuracy/Recall rate: {overall_accuracy:.3f}")
    logger.info(f"Average Injected Tokens: {avg_tokens:.1f} (Min: {min_tokens}, Max: {max_tokens})")
    logger.info("==================================================")

    os.makedirs("./benchmarks", exist_ok=True)
    report_path = f"./benchmarks/benchmark-results-longmemeval-s-{profile}-{date_slug}.md"
    with open(report_path, "w") as f:
        f.write("# Micro-RAG LongMemEval_S QA Benchmark Results\n\n")
        f.write("- Model: `gemini-3.1-flash-lite`\n")
        f.write(f"- Profile: `{profile}`\n")
        f.write(f"- Timestamp: `{timestamp}`\n")
        f.write(f"- Total Evaluated Questions: `{total_questions}`\n")
        f.write(f"- Total Matches: `{total_correct}`\n")
        f.write(f"- Overall Accuracy: `{overall_accuracy:.3f}`\n")
        f.write(f"- Avg Injected Context Tokens: `{avg_tokens:.1f}` (Min: `{min_tokens}`, Max: `{max_tokens}`)\n")
        f.write(f"- Token counter: `{token_counter['backend']}` via `{token_counter['source']}`\n\n")
        f.write("## QA Details\n\n")
        f.write("| Q# | Type | Question | Expected | Micro-RAG Answer | Tokens | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in qa_details:
            f.write(f"| {d['q_idx']} | {d['question_type']} | {d['question']} | {d['expected']} | {d['actual']} | {d['injected_tokens']} | {d['status']} |\n")

    logger.info(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
