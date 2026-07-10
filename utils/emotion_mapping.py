"""
Emotion label mapping from GoEmotions source labels to target study emotions.

Dataset: GoEmotions (Google Research)
- Kaggle: https://www.kaggle.com/datasets/dehbmultani/go-emotions-dataset
- Hugging Face: google-research-datasets/go_emotions

GoEmotions provides 27 fine-grained emotion categories plus Neutral, annotated on
Reddit comments. It is well suited for adaptation to student learning emotions because
it includes direct matches for confusion, curiosity, and frustration-related states,
as well as confidence-related positive affect labels.

Note: Boredom was removed from the GoEmotions taxonomy during dataset construction
because annotators rarely selected it and inter-rater agreement was low. For the
'Bored' target class, we map proxy labels that reflect disengagement and low arousal
(sadness, grief) based on educational psychology literature linking boredom to
negative affect and reduced interest.
"""

from __future__ import annotations

from typing import Iterable

# GoEmotions simplified label ID → label name (28 classes: 0–27)
GOEMOTIONS_LABEL_NAMES: dict[int, str] = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

# Source GoEmotions label → target study emotion
SOURCE_TO_TARGET: dict[str, str] = {
    # Bored — proxy labels (boredom not in GoEmotions taxonomy)
    "sadness": "Bored",
    "grief": "Bored",
    # Confident — mastery, self-efficacy, positive certainty
    "pride": "Confident",
    "optimism": "Confident",
    "admiration": "Confident",
    "approval": "Confident",
    "gratitude": "Confident",
    "joy": "Confident",
    # Confused — cognitive overload, uncertainty
    "confusion": "Confused",
    "nervousness": "Confused",
    "embarrassment": "Confused",
    "fear": "Confused",
    # Curious — interest, exploration, discovery
    "curiosity": "Curious",
    "excitement": "Curious",
    "surprise": "Curious",
    "realization": "Curious",
    "desire": "Curious",
    # Frustrated — blocked goals, irritation during learning
    "annoyance": "Frustrated",
    "anger": "Frustrated",
    "disapproval": "Frustrated",
    "disappointment": "Frustrated",
    "disgust": "Frustrated",
    "remorse": "Frustrated",
}

# When a sample has multiple source labels mapping to different targets,
# resolve conflicts using this priority (most learning-relevant first).
TARGET_PRIORITY: list[str] = [
    "Confused",
    "Frustrated",
    "Curious",
    "Confident",
    "Bored",
]

# Human-readable mapping documentation for reports and notebooks
MAPPING_DOCUMENTATION: dict[str, dict[str, list[str]]] = {
    "Bored": {
        "source_labels": ["sadness", "grief"],
        "rationale": (
            "GoEmotions excludes boredom; sadness and grief proxy disengagement "
            "and low-arousal negative affect during study."
        ),
    },
    "Confident": {
        "source_labels": ["pride", "optimism", "admiration", "approval", "gratitude", "joy"],
        "rationale": (
            "Reflects self-efficacy, mastery, and positive certainty about learning progress."
        ),
    },
    "Confused": {
        "source_labels": ["confusion", "nervousness", "embarrassment", "fear"],
        "rationale": (
            "Captures uncertainty, cognitive overload, and anxiety about understanding material."
        ),
    },
    "Curious": {
        "source_labels": ["curiosity", "excitement", "surprise", "realization", "desire"],
        "rationale": (
            "Represents interest, exploration, and the drive to learn something new."
        ),
    },
    "Frustrated": {
        "source_labels": [
            "annoyance",
            "anger",
            "disapproval",
            "disappointment",
            "disgust",
            "remorse",
        ],
        "rationale": (
            "Covers irritation and blocked goals when learning tasks are difficult or unfair."
        ),
    },
}


def label_ids_to_names(label_ids: Iterable[int]) -> list[str]:
    """Convert GoEmotions numeric label IDs to string label names."""
    return [GOEMOTIONS_LABEL_NAMES[int(label_id)] for label_id in label_ids]


def map_source_labels_to_target(source_labels: Iterable[str]) -> str | None:
    """
    Map one or more GoEmotions source labels to a single target emotion.

    Returns None when no source label maps to a target class (e.g. neutral-only rows).
    """
    mapped_targets: list[str] = []
    for label in source_labels:
        target = SOURCE_TO_TARGET.get(label.lower())
        if target and target not in mapped_targets:
            mapped_targets.append(target)

    if not mapped_targets:
        return None

    if len(mapped_targets) == 1:
        return mapped_targets[0]

    # Resolve multi-label conflicts using defined priority
    for priority_target in TARGET_PRIORITY:
        if priority_target in mapped_targets:
            return priority_target

    return mapped_targets[0]


def print_mapping_summary() -> None:
    """Print the source-to-target mapping table for exploration reports."""
    print("\n" + "=" * 70)
    print("EMOTION LABEL MAPPING: GoEmotions -> Target Study Emotions")
    print("=" * 70)
    for target, details in MAPPING_DOCUMENTATION.items():
        sources = ", ".join(details["source_labels"])
        print(f"\n  {target}")
        print(f"    Source labels : {sources}")
        print(f"    Rationale     : {details['rationale']}")
    print("\n  Unmapped labels : neutral, amusement, caring, love, relief, and others")
    print("                    not listed above are excluded from the target taxonomy.")
    print("=" * 70 + "\n")
