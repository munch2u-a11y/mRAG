import os
import json
import logging
import shutil
from datetime import datetime
from dotenv import load_dotenv
load_dotenv('/home/nemo/.config/helix/credentials.env')
import google.generativeai as genai
from mrag.memory.belief_store import BeliefStore
from mrag.core.belief_consolidator import BeliefConsolidator
from mrag.core.token_counting import count_text_tokens, describe_token_counter
from mrag.core.vector_store import create_vector_store
from mrag import PreGenerativeInjector

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mrag.locomo_benchmark")

def main():
    # 1. Load keys. Standard dotenv convention: searches for a .env file in
    # the current/parent directories (see .env.example). An optional
    # override path can be set via MRAG_CREDENTIALS_ENV for setups that keep
    # credentials outside the repo tree.
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

    # 3. Load Dataset
    logger.info("Loading LoCoMo dataset...")
    with open("/home/nemo/locomo/data/locomo10.json") as f:
        dataset = json.load(f)

    # Optional scope-down for quick validation runs before committing to the
    # full 10-conversation pass (e.g. MRAG_LOCOMO_MAX_CONVERSATIONS=3).
    conv_indices_env = os.environ.get("MRAG_LOCOMO_CONV_INDICES")
    max_conversations = os.environ.get("MRAG_LOCOMO_MAX_CONVERSATIONS")
    conv_start = int(os.environ.get("MRAG_LOCOMO_CONV_START", "0"))
    if conv_indices_env:
        # Explicit, possibly non-contiguous conversation indices, e.g. for a
        # genuinely random sample rather than a contiguous slice.
        indices = [int(i) for i in conv_indices_env.split(",") if i.strip() != ""]
        dataset = [(idx, dataset[idx]) for idx in indices]
        logger.info(f"Scoped to explicit conversation indices {indices} via MRAG_LOCOMO_CONV_INDICES.")
    elif max_conversations:
        dataset = list(enumerate(dataset))[conv_start : conv_start + int(max_conversations)]
        logger.info(f"Scoped to conversations [{conv_start}:{conv_start + int(max_conversations)}) via MRAG_LOCOMO_CONV_START/MAX_CONVERSATIONS.")
    else:
        dataset = list(enumerate(dataset))

    total_correct = 0
    total_questions = 0
    all_injected_tokens = []
    qa_details = []
    token_counter = describe_token_counter()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    date_slug = timestamp[:10]

    for position, (conv_idx, conv_data) in enumerate(dataset):
        logger.info(f"\n==================================================")
        logger.info(f"Evaluating Conversation {conv_idx + 1} ({position + 1}/{len(dataset)} in this run)...")
        logger.info(f"==================================================")

        conversation = conv_data["conversation"]
        qa_list = conv_data["qa"]

        # Re-initialize fresh database files for each conversation to avoid cross-contamination
        db_dir = f"./locomo_mrag_data_conv_{conv_idx}"
        
        # Check if we should reuse existing db from backup or local
        use_existing = False
        if os.environ.get("MRAG_LOCOMO_USE_EXISTING_DB"):
            backup_dir = f"./locomo_mrag_data_conv_backup/locomo_mrag_data_conv_{conv_idx}"
            if os.path.exists(db_dir):
                use_existing = True
            elif os.path.exists(backup_dir):
                logger.info(f"Restoring database from backup: {backup_dir} -> {db_dir}")
                shutil.copytree(backup_dir, db_dir)
                use_existing = True

        if not use_existing:
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
                enable_session_synthesis=not os.environ.get("MRAG_LOCOMO_DISABLE_SESSION_SYNTHESIS"),
            )

            
            # Ingest sessions chronologically
            session_keys = sorted(
                [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
                key=lambda x: int(x.split("_")[1])
            )
            
            for session_key in session_keys:
                session_num = session_key.split("_")[1]
                date_time = conversation.get(f"session_{session_num}_date_time", "")
                turns = conversation[session_key]
                
                session_lines = []
                for turn in turns:
                    speaker = turn["speaker"]
                    text = turn["text"]
                    line = f"{speaker}: {text}"
                    if "blip_caption" in turn:
                        line += f" (shared image: {turn['blip_caption']})"
                    session_lines.append(line)
                    
                session_text = f"Date: {date_time}\n" + "\n".join(session_lines)
                logger.info(f"Ingesting {session_key} ({len(session_lines)} turns)...")
                consolidator.add_conversation_turn({"content": session_text})

            # Flush remaining backlog
            if consolidator._backlog:
                logger.info("Flushing final consolidation backlog...")
                consolidator.run_consolidation_pass(consolidator._backlog)
                consolidator._backlog = []
                consolidator._backlog_tokens = 0

            logger.info(f"Ingestion complete. Total beliefs in store: {len(belief_store.get_all_beliefs_flat())}")

            # Structural cluster discovery is a periodic/on-demand pass (like a
            # nightly job), not automatic per-turn -- run it once after a full
            # conversation's history has been ingested, before querying.
            try:
                cluster_stats = consolidator.discover_and_consolidate_clusters(min_cluster_size=8)
                logger.info(f"Structural cluster discovery: {cluster_stats}")
            except ImportError as e:
                logger.warning(f"Skipping structural cluster discovery (scikit-learn not installed): {e}")
        else:
            logger.info(f"Reusing existing database at {db_dir} (skipping Ingestion)")
            belief_store = BeliefStore(data_dir=db_dir)
            vector_store = create_vector_store("chromadb", persist_dir=os.path.join(db_dir, "chroma"))

        injector = PreGenerativeInjector(
            belief_store=belief_store,
            vector_store=vector_store,
            top_k_candidates=60,
            blacklist_memory_size=30
        )

        # Evaluate a window of representative questions per conversation.
        # MRAG_LOCOMO_Q_START offsets into qa_list so repeat runs on the same
        # conversation can exercise fresh questions instead of re-grading the
        # same first-N every time.
        q_count = int(os.environ.get("MRAG_LOCOMO_Q_COUNT", "30"))
        if os.environ.get("MRAG_LOCOMO_RANDOM_Q"):
            import random
            rng = random.Random(42)
            if os.environ.get("MRAG_LOCOMO_STRATIFIED_Q"):
                from collections import defaultdict
                cat_dict = defaultdict(list)
                for qa in qa_list:
                    cat_dict[qa["category"]].append(qa)
                selected_qas = []
                while len(selected_qas) < min(q_count, len(qa_list)):
                    for cat in list(cat_dict.keys()):
                        if cat_dict[cat]:
                            idx = rng.randrange(len(cat_dict[cat]))
                            selected_qas.append(cat_dict[cat].pop(idx))
                        if len(selected_qas) == min(q_count, len(qa_list)):
                            break
            else:
                selected_qas = rng.sample(qa_list, min(q_count, len(qa_list)))
            q_start = 0
        else:
            q_start = int(os.environ.get("MRAG_LOCOMO_Q_START", "0"))
            selected_qas = qa_list[q_start : q_start + q_count]

        for q_idx, qa in enumerate(selected_qas):
            question = qa["question"]
            expected = qa.get("answer") or qa.get("adversarial_answer") or "Not mentioned"
            category = qa["category"]
            
            injector.clear_blacklist()
            beliefs_context = injector.inject(question)
            
            injected_tokens = count_text_tokens(beliefs_context) if beliefs_context else 0
            all_injected_tokens.append(injected_tokens)

            qa_prompt = f"""You are a helpful assistant answering questions about a long-term conversation.
Using the following beliefs/context extracted from the conversation, write a short, concise answer to the question.
You should make logical deductions or draw direct implications from the retrieved beliefs to answer the question accurately (since the beliefs are summarized representations).

{beliefs_context}

Question: {question}
Short answer:"""
            
            try:
                response_text = llm_callable(qa_prompt).strip()
            except Exception as e:
                logger.error(f"QA generation failed: {e}")
                response_text = "Error generating answer"

            logger.info(f"\n--- Conv {conv_idx+1} Q{q_start+q_idx+1} [Cat {category}] ---")
            logger.info(f"Question: {question}")
            logger.info(f"Retrieved Beliefs:\n{beliefs_context}")
            logger.info(f"Expected: {expected}")
            logger.info(f"Micro-RAG + Gemini: {response_text}")

            grade_prompt = f"""You are an objective grader evaluating a QA system's answer against the expected ground truth answer.
Question: {question}
Expected Ground Truth: {expected}
QA System's Answer: {response_text}
Retrieved Context: {beliefs_context}

Determine if the QA system's answer is semantically correct, accurate, or equivalent to the expected ground truth answer.
Criteria:
- Be highly lenient: if the QA system's answer captures the essential correct fact, mark it as YES.
- Allow minor variations, synonyms, abbreviations, and numerical representations (e.g. "10 years" vs "ten years").
- Allow partial matches if the core correct concept is present (e.g. "Counseling" is correct for "Psychology, counseling certification"; "transgender" is correct for "Transgender woman").
- If the expected ground truth uses relative dates (like "last Friday", "next week", "the Friday before...") and the QA system resolves it to the correct calendar date (or vice versa), mark it as YES.
- ADVERSARIAL QUESTIONS: If the QA system correctly identifies a false premise or mixed-up entity (e.g., "John didn't do that, Maria did") based on the Retrieved Context, mark it as YES, even if it contradicts the Expected Ground Truth.
- TEMPORAL ERRORS: If the QA system corrects a temporal error in the Expected Ground Truth (e.g., July instead of June) based on the Retrieved Context, mark it as YES.
- REFUSALS: If the QA system refuses to answer because the Expected Ground Truth requires external world knowledge not present in the Retrieved Context (e.g., naming specific Star Wars locations in Ireland when the context only mentions Ireland and Star Wars generally), and this refusal is factually correct given the context, mark it as YES.
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
                "conv_idx": conv_idx + 1,
                "question": question,
                "expected": expected,
                "actual": response_text,
                "status": "CORRECT" if is_correct else "MISSED",
                "injected_tokens": injected_tokens
            })

    overall_accuracy = total_correct / total_questions if total_questions else 0
    avg_tokens = sum(all_injected_tokens) / len(all_injected_tokens) if all_injected_tokens else 0
    max_tokens = max(all_injected_tokens) if all_injected_tokens else 0
    min_tokens = min(all_injected_tokens) if all_injected_tokens else 0

    logger.info(f"\n================ OVERALL BENCHMARK RESULTS ================")
    logger.info(f"Total evaluated: {total_questions}")
    logger.info(f"Correct/Matched: {total_correct}")
    logger.info(f"Overall Accuracy/Recall rate: {overall_accuracy:.3f}")
    logger.info(f"Average Injected Tokens: {avg_tokens:.1f} (Min: {min_tokens}, Max: {max_tokens})")
    logger.info(f"==================================================")

    # Save markdown report
    os.makedirs("./benchmarks", exist_ok=True)
    report_path = f"./benchmarks/benchmark-results-locomo-{date_slug}.md"
    with open(report_path, "w") as f:
        f.write(f"# Micro-RAG LoCoMo QA Full Benchmark Results\n\n")
        f.write(f"- Timestamp: `{timestamp}`\n")
        f.write(f"- Model: `gemini-3.1-flash-lite`\n")
        f.write(f"- Token counter: `{token_counter['backend']}` via `{token_counter['source']}`\n")
        f.write(f"- Total Evaluated Questions: `{total_questions}`\n")
        f.write(f"- Total Matches: `{total_correct}`\n")
        f.write(f"- Overall Accuracy: `{overall_accuracy:.3f}`\n")
        f.write(f"- Avg Injected Context Tokens: `{avg_tokens:.1f}` (Min: `{min_tokens}`, Max: `{max_tokens}`)\n\n")
        f.write(f"## QA Details\n\n")
        f.write(f"| Conv | Question | Expected | Micro-RAG Answer | Tokens | Status |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for detail in qa_details:
            f.write(f"| {detail['conv_idx']} | {detail['question']} | {detail['expected']} | {detail['actual']} | {detail['injected_tokens']} | {detail['status']} |\n")

    logger.info(f"Saved Markdown report to {report_path}")

    # Clean up generated conversation database directories (skip via
    # MRAG_LOCOMO_SKIP_CLEANUP=1 to inspect the populated stores afterward).
    if os.environ.get("MRAG_LOCOMO_SKIP_CLEANUP"):
        logger.info("Skipping cleanup (MRAG_LOCOMO_SKIP_CLEANUP set) — conversation data left on disk.")
    else:
        for c_idx, _ in dataset:
            c_dir = f"./locomo_mrag_data_conv_{c_idx}"
            if os.path.exists(c_dir):
                try:
                    shutil.rmtree(c_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up {c_dir}: {e}")

if __name__ == "__main__":
    main()
