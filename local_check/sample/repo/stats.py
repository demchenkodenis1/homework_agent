def average(numbers):
    """Среднее арифметическое списка чисел."""
    return sum(numbers) / len(numbers)


def median(numbers):
    """Медиана списка чисел."""
    ordered = sorted(numbers)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
