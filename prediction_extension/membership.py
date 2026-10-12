"""Unchanged membership helpers from Batch B."""
def overlap(left,right):
    intersection,union = len(set(left)&set(right)),len(set(left)|set(right))
    return intersection,union,intersection/union if union else None


def direction_category(detected,zero,same,reverse):
    if same and reverse or zero and (same or reverse):
        raise ValueError('Contradictory direction flags')
    if not detected:
        if same or reverse:raise ValueError('Undetected taxon cannot have nonzero direction')
        return 'not_detected'
    if zero:return 'zero_difference'
    if same:return 'concordant'
    if reverse:return 'opposite'
    raise ValueError('Nonzero detected effect lacks direction')
