"""Each check's form: the rows its page shows under its result, as a table keyed by the check's key, one builder that
makes a check's form from its rows, and the reading of a form's rows through the rules the check's stage applies again.

A check whose rows are not here yet cannot be run from the window, and its button says so. Nothing on a check's form is
remembered between launches: every row opens at its usual value.
"""
from dataclasses import dataclass

from core import config
from core.refusals import Refusal, require_at_least

from ...widgets.form import Form
from ...widgets.labeled_inputs import IntField
from .help_text import HELP

#: What greys the button of a check whose form is not built yet.
NOT_BUILT_TEXT = "This check cannot be run from the window yet."


@dataclass(frozen=True)
class FormRow:
    """One row of a check's form: the ``setting`` it hands the check's stage, under the key the stage refuses it by,
    labelled as the window's table of controls names that key unless ``label`` is handed; its ``kind`` of field
    ("whole", a whole number), the ``usual`` value it opens at, whether it is ``folded`` among the rarely changed, its
    ``unit``, the ``least`` value its rule takes, and its ``help_text``."""
    setting: str
    label: str | None
    kind: str
    usual: object
    folded: bool = False
    unit: str = ""
    least: int = 1
    help_text: str = ""


#: Each check's rows, by its key: the calibration's test datasets and its operating points, each at least 1, the stage's
#: own rule, opening at the engine's usual sizes.
FORMS: dict[str, tuple[FormRow, ...]] = {
    "calibration": (FormRow("n_cal", None, "whole", config.SBC_N_CAL, help_text=HELP["cal_n"]),
                    FormRow("cal_n_scales", None, "whole", config.CAL_N_SCALES, help_text=HELP["cal_scales"])),
}


def build_form(key: str) -> Form:
    """The form of the check ``key``, its rows at their usual values; a check with no rows has an empty form. The page
    that holds it shows it, or keeps an empty one out of sight, once it stands inside it: a form with no parent made
    visible would be a window of its own."""
    form = Form()
    for row in FORMS.get(key, ()):
        form.add(row.setting, IntField(row.usual, grouped=True), row.unit, usual=row.usual, folded=row.folded,
                 help_text=row.help_text, label=row.label)
    return form


def problems(key: str, form: Form) -> dict:
    """Every rule the form of the check ``key`` breaks now, by field key in the form's order: the line under the field,
    built from its row's label, and what a press refuses it with, the stage's own sentence."""
    found = {}
    for one in FORMS.get(key, ()):
        row = form.rows[one.setting]
        said = row.reading()
        try:
            require_at_least(one.setting, row.value(), one.least)
        except Refusal as refused:
            found[one.setting] = (said or f"{row.label_text()} must be at least {one.least}.", refused.message)
    return found


def call_arguments(key: str, form: Form) -> dict:
    """What the form of the check ``key`` hands its stage, by the stage's own names, read through the stage's own rules;
    the first rule broken is refused, and nothing is handed."""
    found = problems(key, form)
    for setting, (_line, said) in found.items():
        raise Refusal(said, field=setting)
    return {one.setting: require_at_least(one.setting, form.rows[one.setting].value(), one.least)
            for one in FORMS.get(key, ())}
