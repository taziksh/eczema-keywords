#!/usr/bin/env python3
"""Scrape a subreddit with PRAW and compute word frequencies."""

import argparse
import json
import os
import re
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


def tokenize(text: str, filter_stopwords: bool = True) -> list[str]:
    """Lowercase, strip punctuation/URLs, and split text into word tokens."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown link syntax leftovers
    text = re.sub(r"\[|\]|\(|\)", " ", text)
    # Lowercase and strip non-alpha characters (keep hyphens inside words)
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:[-'][a-z]+)*", text)
    if filter_stopwords:
        return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]
    return [t for t in tokens if len(t) > 1]


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens, filtering those with stopwords."""
    if len(tokens) < n:
        return []
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        # Keep ngram only if at least one word is not a stopword
        if any(word not in STOP_WORDS for word in ngram):
            # Skip if starts or ends with stopword (less useful phrases)
            if ngram[0] not in STOP_WORDS and ngram[-1] not in STOP_WORDS:
                ngrams.append(ngram)
    return ngrams


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting on .!? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Clean up and filter empty
    return [s.strip() for s in sentences if s.strip()]


def find_phrase_contexts(
    sentences: list[str], phrase: str, max_contexts: int = 20
) -> list[str]:
    """Find sentences containing the given phrase, with highlighting."""
    phrase_lower = phrase.lower()
    contexts = []
    for sentence in sentences:
        if phrase_lower in sentence.lower():
            # Truncate long sentences
            if len(sentence) > 200:
                # Find phrase position and show context around it
                idx = sentence.lower().find(phrase_lower)
                start = max(0, idx - 80)
                end = min(len(sentence), idx + len(phrase) + 80)
                sentence = ("..." if start > 0 else "") + sentence[start:end] + ("..." if end < len(sentence) else "")
            contexts.append(sentence)
            if len(contexts) >= max_contexts:
                break
    return contexts


def scrape_subreddit(
    subreddit_name: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
    limit: int = 500,
    sort: str = "hot",
    time_filter: str = "all",
    include_comments: bool = True,
    ngram_size: int | None = None,
    collect_sentences: bool = False,
) -> tuple[Counter, Counter | None, list[str] | None]:
    """Scrape posts (and optionally comments) from a subreddit.

    Returns (word_counts, ngram_counts, sentences).
    ngram_counts is None if ngram_size not set.
    sentences is None if collect_sentences is False.
    """
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    subreddit = reddit.subreddit(subreddit_name)

    word_counter: Counter = Counter()
    ngram_counter: Counter | None = Counter() if ngram_size else None
    all_sentences: list[str] | None = [] if collect_sentences else None
    posts_processed = 0

    def process_text(text: str):
        """Process text for both word and ngram counting."""
        word_counter.update(tokenize(text))
        if ngram_size:
            # For ngrams, get tokens without filtering stopwords first
            tokens = tokenize(text, filter_stopwords=False)
            ngram_counter.update(extract_ngrams(tokens, ngram_size))
        if collect_sentences:
            all_sentences.extend(extract_sentences(text))

    # Get submissions based on sort type
    if sort == "hot":
        submissions = subreddit.hot(limit=limit)
    elif sort == "new":
        submissions = subreddit.new(limit=limit)
    elif sort == "top":
        submissions = subreddit.top(limit=limit, time_filter=time_filter)
    elif sort == "rising":
        submissions = subreddit.rising(limit=limit)
    else:
        raise ValueError(f"Unknown sort: {sort!r}")

    for submission in submissions:
        # Count title + selftext
        text = f"{submission.title} {submission.selftext}"
        process_text(text)

        if include_comments:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                process_text(comment.body)

        posts_processed += 1
        if posts_processed % 50 == 0:
            print(f"  processed {posts_processed} posts ...")

    print(f"Done — processed {posts_processed} posts from r/{subreddit_name}")
    return word_counter, ngram_counter, all_sentences


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
        "-t", "--time",
        choices=["all", "year", "month", "week", "day", "hour"],
        default="year",
        help="Time filter for 'top' sort (default: year)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip comment text, only use post titles/bodies",
    )
    parser.add_argument(
        "--ngrams",
        type=int,
        choices=[2, 3, 4],
        help="Extract n-grams (phrases): 2=bigrams, 3=trigrams, 4=four-grams",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top words/phrases to display (default: 50)",
    )
    parser.add_argument(
        "--output",
        help="Write full frequency counts to a JSON file",
    )
    parser.add_argument(
        "--inspect",
        metavar="PHRASE",
        help="Show sentences containing this phrase (e.g., --inspect 'side effects')",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=20,
        help="Max number of example sentences to show (default: 20)",
    )
    parser.add_argument(
        "--save-corpus",
        metavar="FILE",
        help="Save scraped sentences to a JSON file for later reuse",
    )
    parser.add_argument(
        "--corpus",
        metavar="FILE",
        help="Load sentences from a saved corpus file instead of scraping Reddit",
    )
    args = parser.parse_args()

    # Load from corpus or scrape
    if args.corpus:
        print(f"Loading corpus from {args.corpus} ...")
        with open(args.corpus) as f:
            corpus_data = json.load(f)
        sentences = corpus_data["sentences"]
        print(f"Loaded {len(sentences)} sentences from corpus.")

        # Analyze the corpus
        word_freq: Counter = Counter()
        ngram_freq: Counter | None = Counter() if args.ngrams else None
        for sentence in sentences:
            word_freq.update(tokenize(sentence))
            if args.ngrams:
                tokens = tokenize(sentence, filter_stopwords=False)
                ngram_freq.update(extract_ngrams(tokens, args.ngrams))
    else:
        if not args.client_id or not args.client_secret:
            parser.error(
                "Reddit credentials required. Pass --client-id/--client-secret "
                "or set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET env vars."
            )

        sort_desc = f"{args.sort}" + (f"/{args.time}" if args.sort == "top" else "")
        print(f"Scraping r/{args.subreddit} ({sort_desc}, limit={args.limit}) ...")
        word_freq, ngram_freq, sentences = scrape_subreddit(
            subreddit_name=args.subreddit,
            client_id=args.client_id,
            client_secret=args.client_secret,
            user_agent=args.user_agent,
            limit=args.limit,
            sort=args.sort,
            time_filter=args.time,
            include_comments=not args.no_comments,
            ngram_size=args.ngrams,
            collect_sentences=bool(args.inspect) or bool(args.save_corpus),
        )

        # Save corpus if requested
        if args.save_corpus and sentences:
            corpus_data = {
                "subreddit": args.subreddit,
                "sort": args.sort,
                "time_filter": args.time if args.sort == "top" else None,
                "limit": args.limit,
                "sentence_count": len(sentences),
                "sentences": sentences,
            }
            with open(args.save_corpus, "w") as f:
                json.dump(corpus_data, f)
            print(f"Saved {len(sentences)} sentences to {args.save_corpus}")

    # If inspect mode, just show contexts for the phrase
    if args.inspect and sentences:
        contexts = find_phrase_contexts(sentences, args.inspect, args.inspect_limit)
        if contexts:
            print(f"\nFound {len(contexts)} examples of \"{args.inspect}\":\n")
            for i, ctx in enumerate(contexts, 1):
                # Highlight the phrase in the output
                highlighted = re.sub(
                    re.escape(args.inspect),
                    f"**{args.inspect}**",
                    ctx,
                    flags=re.IGNORECASE,
                )
                print(f"  {i:>2}. \"{highlighted}\"\n")
        else:
            print(f"\nNo occurrences of \"{args.inspect}\" found.")
        return

    # Display top words
    print(f"\nTop {args.top_n} words in r/{args.subreddit}:\n")
    for rank, (word, count) in enumerate(word_freq.most_common(args.top_n), 1):
        bar = "█" * min(count // 5, 40)
        print(f"  {rank:>3}. {word:<25} {count:>6}  {bar}")

    # Display top n-grams if requested
    if ngram_freq:
        ngram_name = {2: "bigrams", 3: "trigrams", 4: "four-grams"}[args.ngrams]
        print(f"\nTop {args.top_n} {ngram_name} (phrases) in r/{args.subreddit}:\n")
        for rank, (ngram, count) in enumerate(ngram_freq.most_common(args.top_n), 1):
            phrase = " ".join(ngram)
            bar = "█" * min(count // 3, 40)
            print(f"  {rank:>3}. {phrase:<35} {count:>5}  {bar}")

    # Write output file
    if args.output:
        output_data = {"words": word_freq.most_common()}
        if ngram_freq:
            # Convert tuple keys to strings for JSON
            output_data["ngrams"] = [
                (" ".join(ngram), count) for ngram, count in ngram_freq.most_common()
            ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull counts written to {args.output}")


if __name__ == "__main__":
    main()
