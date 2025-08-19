from fuzzywuzzy import fuzz

def filter_strings_by_search(search, source, n=5, threshold=50, max_difference=10):
    """
    Filter a list of strings by a search string and return the top n matches above a threshold.
    Args:
        search (str): Search string to match against.
        source (list): List of strings to search within.
        n (int): Maximum number of matches to return.
        threshold (int): Minimum matching score (0-100) to include a match.
        max_difference (int): Maximum allowed absolute difference from the highest score.
    Returns:
        list: List of tuples with best matches and their scores above the thresholds.
    """
    # Calculate scores for each item in the source list
    scored_matches = [(item, fuzz.partial_ratio(search.lower(), item.lower())) for item in source]
    
    # Filter matches based on the absolute threshold
    filtered_matches = [match for match in scored_matches if match[1] >= threshold]
    
    if filtered_matches:
        # Determine the maximum score among filtered matches
        max_score = max(match[1] for match in filtered_matches)
        
        # Apply absolute difference threshold
        filtered_matches = [match for match in filtered_matches if max_score - match[1] <= max_difference]
    
    # Sort matches by score in descending order and limit to top n
    sorted_matches = sorted(filtered_matches, key=lambda x: x[1], reverse=True)[:n]
    
    return [match[0] for match in sorted_matches]