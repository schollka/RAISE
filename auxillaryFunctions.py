def safeFloat(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safeInt(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default