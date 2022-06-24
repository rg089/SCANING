def find_ops(logs, ops=["templatization", "substitution"]):
    """
    finds the index of the log in logs which contains only those corruptions present inside ops
    else, returns -1
    """
    found_idx = -1
    for i, log in enumerate(logs):
        corruptions = log.split("+")
        all_match = True

        for corruption in corruptions:
            corruption = corruption.lower().strip()
            
            matched = []
            for op in ops:
                if op in corruption: matched.append(True)
                else: matched.append(False)

            if sum(matched) == 0:
                all_match = False

        if all_match:
            found_idx = i

    return found_idx