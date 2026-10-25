"""The words the window writes for a date, a number and a length of time, the same on every computer whatever
Windows' regional setting. The look and the run's model both write them, so they live here, where neither the
toolkit nor the engine is imported: a module with no window, the run's model among them, can write a date as
the window does."""
import math
import numbers
from datetime import datetime, timedelta

# English letters on every computer: a date is written from these, never from Windows' regional setting
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def number_text(value) -> str:
    """A quantity as the window writes it on every computer, whatever Windows' regional setting: a comma between
    thousands, a full stop for the decimal mark, the keyboard's hyphen for a minus, so the text reads back in a
    field. A whole number is written whole ("20,480,000"); any other real number in the shortest form that reads
    back the same ("2,048.0", "0.05").

    The unit is the caller's: ``number_text(40.1) + " %"``. An identifier (a seed, a record's id) is not a
    quantity and is never handed to this. A truth value, a text, None or anything else raises TypeError."""
    if isinstance(value, bool):                             # asked first: Python counts True among the whole numbers
        raise TypeError("a truth value is not a quantity")
    if isinstance(value, numbers.Integral):
        return format(int(value), ",")
    if isinstance(value, numbers.Real):
        return format(float(value), ",")
    raise TypeError(f"not a quantity: {type(value).__name__}")


def date_text(when: datetime, now: datetime | None = None) -> str:
    """A date and time as the window writes it on every computer: "3 Oct 2026, 14:02", the hours on the 24-hour
    clock and the seconds dropped. Handed the present moment, a date less than seven days from it, ahead of it or
    behind it, carries its weekday and no year: "Sun 4 Oct, 12:10".

    The letters are English whatever Windows' regional setting. The fields of ``when`` are written as handed in:
    nothing is converted between zones."""
    clock = f"{when.hour:02d}:{when.minute:02d}"
    if now is not None and abs(when - now) < timedelta(days=7):
        return f"{WEEKDAYS[when.weekday()]} {when.day} {MONTHS[when.month - 1]}, {clock}"
    return f"{when.day} {MONTHS[when.month - 1]} {when.year}, {clock}"


def duration_text(seconds: float) -> str:
    """A length of time in words, counted down to whole units, in the two largest units it holds: "less than a
    minute", "12 minutes", "1 hour 12 minutes", "2 days 17 hours". A unit that comes to nought is left out ("2
    hours", not "2 hours 0 minutes"), and a length below nought, or one that is not a finite number, is less
    than a minute."""
    seconds = float(seconds)
    minutes = math.floor(seconds / 60) if math.isfinite(seconds) and seconds > 0 else 0
    if minutes < 1:
        return "less than a minute"
    days, rest = divmod(minutes, 24 * 60)
    hours, minutes = divmod(rest, 60)
    if days:
        return _counted(days, "day") + (f" {_counted(hours, 'hour')}" if hours else "")
    if hours:
        return _counted(hours, "hour") + (f" {_counted(minutes, 'minute')}" if minutes else "")
    return _counted(minutes, "minute")


def _counted(count: int, unit: str) -> str:
    """``count`` of ``unit``, the unit in the plural unless the count is one: "1 hour", "17 hours"."""
    return f"{count} {unit}{'' if count == 1 else 's'}"
