#!/usr/bin/env python3
"""Scrape a subreddit with PRAW and compute word frequencies."""

import argparse
import json
import os
import re
import string
from collections import Counter

import praw


# Common English stop words to filter out
STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "i'm", "if", "im", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "just", "know", "let's",
    "like", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "really", "same", "shan't",
    "she", "should", "shouldn't", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "also", "even", "going", "go", "much", "one", "still", "way", "well",
    "thing", "things", "think", "back", "make", "made", "people", "time",
    "been", "would", "will", "can", "get", "got", "use", "used", "using",
    "try", "tried", "ve", "re", "don", "didn", "doesn", "isn", "wasn",
    "couldn", "shouldn", "wouldn", "haven", "hasn", "aren", "weren",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation/URLs, and split text into word tokens."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown link syntax leftovers
    text = re.sub(r"\[|\]|\(|\)", " ", text)
    # Lowercase and strip non-alpha characters (keep hyphens inside words)
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:[-'][a-z]+)*", text)
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def scrape_subreddit(
    subreddit_name: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
    limit: int = 500,
    sort: str = "hot",
    include_comments: bool = True,
) -> Counter:
    """Scrape posts (and optionally comments) from a subreddit, return word counts."""
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    subreddit = reddit.subreddit(subreddit_name)

    fetch = {
        "hot": subreddit.hot,
        "new": subreddit.new,
        "top": subreddit.top,
        "rising": subreddit.rising,
    }
    if sort not in fetch:
        raise ValueError(f"Unknown sort: {sort!r}. Choose from {list(fetch)}")

    counter: Counter = Counter()
    posts_processed = 0

    for submission in fetch[sort](limit=limit):
        # Count title + selftext
        text = f"{submission.title} {submission.selftext}"
        counter.update(tokenize(text))

        if include_comments:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                counter.update(tokenize(comment.body))

        posts_processed += 1
        if posts_processed % 50 == 0:
            print(f"  processed {posts_processed} posts ...")

    print(f"Done — processed {posts_processed} posts from r/{subreddit_name}")
    return counter


def main():
    parser = argparse.ArgumentParser(
        description="Scrape a subreddit and compute word frequencies."
    )
    parser.add_argument(
        "subreddit",
        nargs="?",
        default="eczema",
        help="Subreddit to scrape (default: eczema)",
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("REDDIT_CLIENT_ID"),
        help="Reddit app client ID (or set REDDIT_CLIENT_ID env var)",
    )
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("REDDIT_CLIENT_SECRET"),
        help="Reddit app client secret (or set REDDIT_CLIENT_SECRET env var)",
    )
    parser.add_argument(
        "--user-agent",
        default="eczema-keyword-scraper:v0.1 (by /u/YOUR_USERNAME)",
        help="User agent string for Reddit API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max number of posts to fetch (default: 500)",
    )
    parser.add_argument(
        "--sort",
        choices=["hot", "new", "top", "rising"],
        default="hot",
        help="Sort order (default: hot)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip comment text, only use post titles/bodies",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top words to display (default: 50)",
    )
    parser.add_argument(
        "--output",
        help="Write full frequency counts to a JSON file",
    )
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        parser.error(
            "Reddit credentials required. Pass --client-id/--client-secret "
            "or set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET env vars."
        )

    print(f"Scraping r/{args.subreddit} ({args.sort}, limit={args.limit}) ...")
    freq = scrape_subreddit(
        subreddit_name=args.subreddit,
        client_id=args.client_id,
        client_secret=args.client_secret,
        user_agent=args.user_agent,
        limit=args.limit,
        sort=args.sort,
        include_comments=not args.no_comments,
    )

    print(f"\nTop {args.top_n} words in r/{args.subreddit}:\n")
    for rank, (word, count) in enumerate(freq.most_common(args.top_n), 1):
        bar = "█" * min(count // 5, 40)
        print(f"  {rank:>3}. {word:<25} {count:>6}  {bar}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(freq.most_common(), f, indent=2)
        print(f"\nFull counts written to {args.output}")


if __name__ == "__main__":
    main()
