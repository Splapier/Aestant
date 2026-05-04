"""Proof-of-concept test script for winners/losers embeddings and similarity search.

Usage:
    python test_average_search.py
"""

from pathlib import Path

from chatbot.average_embeddings import load_state
from chatbot.similarity_search import find_winner
from chatbot.tagging_engine import DATASET_DIR


def main():
    state = load_state()

    winners = state.get("winners", {})
    losers = state.get("losers", {})

    print(
        f"Winners - Image count: {winners.get('image_count', 0)}, Tag count: {winners.get('tag_count', 0)}"
    )
    print(
        f"Losers  - Image count: {losers.get('image_count', 0)}, Tag count: {losers.get('tag_count', 0)}"
    )

    if winners.get("image_embedding_avg") is None:
        print("No winners image embeddings found. Use add_winner() first.")
        return

    print("\nSearching for winner (based on winners/losers pools)...")
    result = find_winner(DATASET_DIR, winners, losers)

    print(f"\nWinner: {result['winner']}")
    print(f"Score (sim_to_winners - sim_to_losers): {result['score']:.4f}")

    print(
        f"\nState saved to: {Path(__file__).resolve().parent / 'average_embeddings.json'}"
    )


if __name__ == "__main__":
    main()
