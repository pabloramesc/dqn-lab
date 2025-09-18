def format_time(seconds: float) -> str:
    seconds = int(seconds)
    d, seconds = divmod(seconds, 86400)  # 86400 s in a day
    h, seconds = divmod(seconds, 3600)  # 3600 s in an hour
    m, s = divmod(seconds, 60)  # 60 s in a minute
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0 or d > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0 or d > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
