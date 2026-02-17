import os
import json
import logging
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score, accuracy_score
)
from schemas import TweetTargetExample, TargetAnnotation
from data_collector import load_tweet_target_dataset
import config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("GENAI_API_KEY not found in .env")

# Models to benchmark
MODELS = [
    "deepseek-r1:14b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
]

SHOT_CONFIGS = [0, 1, 3, 5]


def load_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.GENAI_BASE_URL,
        api_key=api_key,
        model=model,
        streaming=False,
        max_tokens=50,
    )


# ──────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────

ZERO_SHOT_TARGET_PROMPT = ChatPromptTemplate.from_template(
    """Event: {event}
Tweet: {tweet_text}
Author: {author}
Author Party: {author_party}
Target Entity: {entity}

Task: Identify if the given entity is a target of the tweet.
A target entity is defined as an entity that would be present
in the full unambiguous explanation of the tweet.

Answer with exactly one word: TARGET or NOT_TARGET"""
)

ZERO_SHOT_SENTIMENT_PROMPT = ChatPromptTemplate.from_template(
    """Event: {event}
Tweet: {tweet_text}
Author: {author}
Author Party: {author_party}
Target Entity: {entity}

Task: Identify the sentiment of the tweet towards the given
target entity. Consider that the tweet is ambiguous and the
entity might be implied without being explicitly mentioned.

Answer with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL"""
)

FEW_SHOT_TARGET_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at analyzing political tweets to identify target entities.

A target entity is defined as an entity that would be present in the full
unambiguous explanation of the tweet, even if not explicitly mentioned.

Here are some examples:

{examples}

Now analyze this tweet:

Event: {event}
Tweet: {tweet_text}
Author: {author}
Author Party: {author_party}
Target Entity: {entity}

Answer with exactly one word: TARGET or NOT_TARGET"""
)

FEW_SHOT_SENTIMENT_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at analyzing sentiment in political tweets.

Consider that the tweet is ambiguous and the entity might be implied
without being explicitly mentioned.

Here are some examples:

{examples}

Now analyze this tweet:

Event: {event}
Tweet: {tweet_text}
Author: {author}
Author Party: {author_party}
Target Entity: {entity}

Answer with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL"""
)


def build_few_shot_examples(
    train_examples: list[TweetTargetExample],
    n_shots: int,
    task: str,
) -> str:
    """Build few-shot example text from training data."""
    candidates = []
    for ex in train_examples:
        for t in ex.targets:
            if task == "target":
                label = "TARGET" if t.is_target else "NOT_TARGET"
                candidates.append((ex, t, label))
            elif task == "sentiment" and t.is_target and t.sentiment:
                candidates.append((ex, t, t.sentiment.value.upper()))

    selected = random.sample(candidates, min(n_shots, len(candidates)))

    lines = []
    for i, (ex, t, label) in enumerate(selected, 1):
        lines.append(
            f"Example {i}:\n"
            f"Event: {ex.event or 'N/A'}\n"
            f"Tweet: {ex.tweet_text or 'N/A'}\n"
            f"Author: {ex.author or 'N/A'}\n"
            f"Author Party: {ex.author_party.value if ex.author_party else 'N/A'}\n"
            f"Target Entity: {t.entity}\n"
            f"Answer: {label}"
        )
    return "\n\n".join(lines)


# ──────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────

def parse_target_response(response: str) -> bool:
    cleaned = response.strip().upper()
    if "NOT_TARGET" in cleaned or "NOT TARGET" in cleaned:
        return False
    if "TARGET" in cleaned:
        return True
    if cleaned.startswith("YES"):
        return True
    return False


def parse_sentiment_response(response: str) -> str:
    cleaned = response.strip().upper()
    if "POSITIVE" in cleaned:
        return "positive"
    if "NEGATIVE" in cleaned:
        return "negative"
    return "neutral"


# ──────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────

def run_benchmark(
    llm: ChatOpenAI,
    model_name: str,
    test_examples: list[TweetTargetExample],
    n_shots: int = 0,
    train_examples: list[TweetTargetExample] | None = None,
) -> dict:
    """Run target identification + sentiment detection benchmark."""
    if n_shots == 0:
        target_prompt = ZERO_SHOT_TARGET_PROMPT
        sentiment_prompt = ZERO_SHOT_SENTIMENT_PROMPT
        target_examples_text = ""
        sentiment_examples_text = ""
    else:
        target_prompt = FEW_SHOT_TARGET_PROMPT
        sentiment_prompt = FEW_SHOT_SENTIMENT_PROMPT
        random.seed(42)
        target_examples_text = build_few_shot_examples(train_examples, n_shots, "target")
        sentiment_examples_text = build_few_shot_examples(train_examples, n_shots, "sentiment")

    target_chain = target_prompt | llm | StrOutputParser()
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()

    target_y_true = []
    target_y_pred = []
    sentiment_y_true = []
    sentiment_y_pred = []

    total_pairs = sum(len(e.targets) for e in test_examples)
    logger.info(f"[{model_name}] Running {n_shots}-shot on {len(test_examples)} tweets ({total_pairs} pairs)")

    for i, example in enumerate(test_examples):
        for target in example.targets:
            invoke_args = {
                "event": example.event or "N/A",
                "tweet_text": example.tweet_text or "N/A",
                "author": example.author or "N/A",
                "author_party": example.author_party.value if example.author_party else "N/A",
                "entity": target.entity,
            }
            if n_shots > 0:
                invoke_args["examples"] = target_examples_text

            try:
                response = target_chain.invoke(invoke_args)
                predicted_target = parse_target_response(response)
            except Exception as e:
                logger.warning(f"Target chain error: {e}")
                predicted_target = False

            target_y_true.append(target.is_target)
            target_y_pred.append(predicted_target)

            if predicted_target:
                if n_shots > 0:
                    invoke_args["examples"] = sentiment_examples_text
                try:
                    response = sentiment_chain.invoke(invoke_args)
                    predicted_sentiment = parse_sentiment_response(response)
                except Exception as e:
                    logger.warning(f"Sentiment chain error: {e}")
                    predicted_sentiment = "neutral"
            else:
                predicted_sentiment = "non-target"

            if target.is_target and target.sentiment:
                gt_sentiment = target.sentiment.value
            else:
                gt_sentiment = "non-target"

            sentiment_y_true.append(gt_sentiment)
            sentiment_y_pred.append(predicted_sentiment)

        if (i + 1) % 50 == 0:
            logger.info(f"  [{model_name}] Processed {i + 1}/{len(test_examples)} tweets")

    results = evaluate(target_y_true, target_y_pred, sentiment_y_true, sentiment_y_pred, model_name, n_shots)
    return results


def evaluate(
    target_y_true, target_y_pred,
    sentiment_y_true, sentiment_y_pred,
    model_name: str,
    n_shots: int,
) -> dict:
    """Compute and print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} | {n_shots}-shot No-Context")
    print(f"{'='*60}")

    # Target identification metrics
    target_f1 = f1_score(target_y_true, target_y_pred, average="macro")
    target_precision = precision_score(target_y_true, target_y_pred, average="macro")
    target_recall = recall_score(target_y_true, target_y_pred, average="macro")
    target_accuracy = accuracy_score(target_y_true, target_y_pred)

    print(f"\n--- Target Identification ---")
    print(f"  Macro-F1:  {target_f1:.4f}")
    print(f"  Precision: {target_precision:.4f}")
    print(f"  Recall:    {target_recall:.4f}")
    print(f"  Accuracy:  {target_accuracy:.4f}")
    print(classification_report(
        target_y_true, target_y_pred,
        target_names=["Target", "Non-Target"],
    ))

    # Sentiment metrics
    sentiment_labels = ["positive", "negative", "neutral", "non-target"]
    sentiment_f1 = f1_score(
        sentiment_y_true, sentiment_y_pred,
        labels=sentiment_labels, average="macro", zero_division=0,
    )
    sentiment_precision = precision_score(
        sentiment_y_true, sentiment_y_pred,
        labels=sentiment_labels, average="macro", zero_division=0,
    )
    sentiment_recall = recall_score(
        sentiment_y_true, sentiment_y_pred,
        labels=sentiment_labels, average="macro", zero_division=0,
    )
    sentiment_accuracy = accuracy_score(sentiment_y_true, sentiment_y_pred)

    print(f"--- Sentiment Identification ---")
    print(f"  Macro-F1:  {sentiment_f1:.4f}")
    print(f"  Precision: {sentiment_precision:.4f}")
    print(f"  Recall:    {sentiment_recall:.4f}")
    print(f"  Accuracy:  {sentiment_accuracy:.4f}")
    print(classification_report(
        sentiment_y_true, sentiment_y_pred,
        labels=sentiment_labels, zero_division=0,
    ))

    return {
        "model": model_name,
        "n_shots": n_shots,
        "target_macro_f1": target_f1,
        "target_precision": target_precision,
        "target_recall": target_recall,
        "target_accuracy": target_accuracy,
        "sentiment_macro_f1": sentiment_f1,
        "sentiment_precision": sentiment_precision,
        "sentiment_recall": sentiment_recall,
        "sentiment_accuracy": sentiment_accuracy,
        "total_examples": len(target_y_true),
    }


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_results(all_results: list[dict]) -> None:
    """Generate comparison graphs for all models and shot configurations."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = config.OUTPUT_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(set(r["model"] for r in all_results))
    shots = sorted(set(r["n_shots"] for r in all_results))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    def get_metric(model, n_shots, metric):
        for r in all_results:
            if r["model"] == model and r["n_shots"] == n_shots:
                return r[metric]
        return 0

    # ── Plot 1: Target Macro-F1 by model and shots ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(shots))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        values = [get_metric(model, s, "target_macro_f1") for s in shots]
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Number of Shots")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Target Identification: Macro-F1 by Model and Shot Count")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{s}-shot" for s in shots])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "target_f1_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved target_f1_comparison.png")

    # ── Plot 2: Sentiment Macro-F1 by model and shots ──
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [get_metric(model, s, "sentiment_macro_f1") for s in shots]
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Number of Shots")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Sentiment Identification: Macro-F1 by Model and Shot Count")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"{s}-shot" for s in shots])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_f1_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved sentiment_f1_comparison.png")

    # ── Plot 3: All metrics side-by-side for each model ──
    metrics = [
        ("target_macro_f1", "Target F1"),
        ("target_precision", "Target Prec"),
        ("target_recall", "Target Rec"),
        ("target_accuracy", "Target Acc"),
        ("sentiment_macro_f1", "Sent F1"),
        ("sentiment_precision", "Sent Prec"),
        ("sentiment_recall", "Sent Rec"),
        ("sentiment_accuracy", "Sent Acc"),
    ]

    for model in models:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / len(shots)

        for i, s in enumerate(shots):
            values = [get_metric(model, s, m[0]) for m in metrics]
            bars = ax.bar(x + i * width, values, width,
                         label=f"{s}-shot", color=colors[i % len(colors)])
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(f"All Metrics: {model}")
        ax.set_xticks(x + width * (len(shots) - 1) / 2)
        ax.set_xticklabels([m[1] for m in metrics], rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        safe_name = model.replace(":", "_").replace("/", "_")
        plt.savefig(output_dir / f"all_metrics_{safe_name}.png", dpi=150)
        plt.close()
        logger.info(f"Saved all_metrics_{safe_name}.png")

    # ── Plot 4: Line chart — F1 progression across shots ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, model in enumerate(models):
        target_vals = [get_metric(model, s, "target_macro_f1") for s in shots]
        sent_vals = [get_metric(model, s, "sentiment_macro_f1") for s in shots]
        ax1.plot(shots, target_vals, marker="o", label=model, color=colors[i % len(colors)], linewidth=2)
        ax2.plot(shots, sent_vals, marker="s", label=model, color=colors[i % len(colors)], linewidth=2)

    ax1.set_xlabel("Number of Shots")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Target Identification F1 Progression")
    ax1.set_xticks(shots)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Number of Shots")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Sentiment Identification F1 Progression")
    ax2.set_xticks(shots)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "f1_progression.png", dpi=150)
    plt.close()
    logger.info(f"Saved f1_progression.png")

    print(f"\nAll plots saved to {output_dir}/")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    all_examples = load_tweet_target_dataset()
    train_examples = [e for e in all_examples if e.split == "train"]
    test_examples = [e for e in all_examples if e.split == "test"]

    logger.info(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    all_results = []

    for model_name in MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Benchmarking model: {model_name}")
        logger.info(f"{'='*60}")

        llm = load_llm(model_name)

        for n_shots in SHOT_CONFIGS:
            results = run_benchmark(
                llm=llm,
                model_name=model_name,
                test_examples=test_examples,
                n_shots=n_shots,
                train_examples=train_examples if n_shots > 0 else None,
            )
            all_results.append(results)

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary: No-Context Benchmark (All Models)")
    print(f"{'='*70}")
    print(f"{'Model':<20} | {'Shots':>5} | {'Target F1':>9} | {'Sent F1':>7} | {'T-Prec':>6} | {'T-Rec':>5} | {'T-Acc':>5}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['model']:<20} | {r['n_shots']:>5} | "
            f"{r['target_macro_f1']:>9.4f} | {r['sentiment_macro_f1']:>7.4f} | "
            f"{r['target_precision']:>6.4f} | {r['target_recall']:>5.4f} | {r['target_accuracy']:>5.4f}"
        )

    # Save results
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {config.OUTPUT_DIR / 'benchmark_results.json'}")

    # Generate plots
    plot_results(all_results)
