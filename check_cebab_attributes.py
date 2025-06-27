"""
Script to check which attributes in the CEBaB dataset have examples with ratings.
"""

from datasets import load_dataset

def main():
    print("Loading CEBaB dataset...")
    cebab = load_dataset("CEBaB/CEBaB")

    print(f"Dataset structure: {list(cebab.keys())}")
    print(f"Test set size: {len(cebab['test'])}")

    # Get a sample example to see the keys
    sample = cebab['test'][0]
    print(f"Example keys: {list(sample.keys())}")

    # Check which attributes have examples with ratings
    attributes = ['food', 'ambiance', 'service', 'noise', 'price']

    for attribute in attributes:
        attribute_key = f"{attribute}_aspect_majority"
        if attribute_key in sample:
            # Count examples with non-None values for this attribute
            count = sum(1 for ex in cebab['test'] if ex.get(attribute_key) is not None)
            print(f"Attribute '{attribute}': {count} examples with ratings")

            # Get the distribution of ratings
            ratings = {}
            for ex in cebab['test']:
                rating = ex.get(attribute_key)
                if rating is not None:
                    ratings[rating] = ratings.get(rating, 0) + 1

            print(f"  Rating distribution: {ratings}")

            # Print a few examples
            print("  Sample examples:")
            examples_shown = 0
            for ex in cebab['test']:
                rating = ex.get(attribute_key)
                if rating is not None and rating not in ['', 'unknown'] and examples_shown < 3:
                    description = ex.get('description', '')[:100] + '...' if len(ex.get('description', '')) > 100 else ex.get('description', '')
                    print(f"    Rating: {rating}, Description: {description}")
                    examples_shown += 1
            print()
        else:
            print(f"Attribute '{attribute}': No examples with ratings")

if __name__ == "__main__":
    main()
