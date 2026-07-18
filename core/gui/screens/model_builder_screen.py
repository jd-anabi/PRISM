"""The user-defined model builder (reached from Settings -> "User-defined models").

Declare state variables, then per variable type the deterministic RHS of dx/dt, the white-noise
strength D (the solver diffusion is g = sqrt(2*D); parameters only -- additive noise), and optionally
one time-driven forcing. Equations are NONDIMENSIONAL (nothing auto-nondimensionalizes); x_scale /
t_scale only label + redimensionalize the Simulate axes. Variable 1 is the observable Simulate plots.

Everything here is light (sympy parse + a 100-step batch-1 smoke integration, milliseconds), so it all
runs on the GUI thread -- no BasePanel/dispatch. Saving is refused while a task runs anywhere in the
app (refreshing the model combos mid-run would fight the app-wide control lock), and persistence goes
through core/Helpers/model_store.py (the JSON + the Bounds/Cells/Units triple).
"""
import torch
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QRadioButton, QScrollArea, QStackedWidget, QVBoxLayout,
                               QWidget)

from core import config, registry
from core.config import DT_EXP_S
from core.forcing import FORCING_PARAM_NAMES
from core.Helpers import model_store
from core.Models.user_model import ModelParseError, UserModel, parse_user_model
from core.Solvers import sdeint

from ..panels.base_panel import BasePanel
from ..widgets.help_badge import add_help_row
from ..widgets.labeled_inputs import FloatField

HELP = {
    "name": "The model's name as it appears in every model dropdown. Letters/digits/_ (max 24).",
    "variables": "Comma-separated state variable names, e.g. `x, y`. One first-order equation per "
                 "variable; the FIRST variable is the observable Simulate plots. A 2nd-order system "
                 "must be written as two first-order equations.",
    "drift": "The deterministic right-hand side of dx/dt (nondimensional; may be nonlinear). Use the "
             "declared variables, parameters (any new name), numbers, + - * / ^ ( ) and "
             "sin cos tan asin acos atan sinh cosh tanh exp log sqrt abs sign, pi.",
    "D": "White-noise strength D (<xi(t)xi(t')> = 2 D delta): the solver's noise amplitude is "
         "sqrt(2*D). Use parameters/numbers for additive noise, or reference state variables for "
         "multiplicative (state-dependent) noise; 0 = noiseless.",
    "init": "Initial condition for this variable (nondimensional).",
    "forcing": "Optional time-driven external force added to this variable's RHS. For a restoring "
               "(spring) force, put -k*(x - x0) directly in the RHS instead -- it depends on state.",
    "x_scale": "Length scale (nm per ND unit) used to redimensionalize the Simulate displacement axis.",
    "t_scale": "Time scale (seconds per ND time unit) used to map ND time to the seconds axis.",
}

# (kind key, display label); "" = no forcing.
_FORCE_KINDS = (("", "None"), ("sin", "Sinusoidal"), ("step", "Step"),
                ("triangular", "Triangular"), ("exponential", "Exponential"))
_FORCE_DEFAULTS = {"amp": 0.0, "freq": 1.0, "phase": 0.0, "offset": 0.0, "t0": 0.0, "tau": 1.0}
_SMOKE_STEPS = 100          # validation integration length (batch 1 -- milliseconds)


class _VarRow(QGroupBox):
    """One state variable's definition: drift | D | init | forcing kind + its parameters."""

    def __init__(self, var_name: str, parent=None):
        super().__init__(f"d{var_name}/dt", parent)
        self.var_name = var_name
        form = QFormLayout(self)

        self.drift = QLineEdit()
        self.drift.setPlaceholderText(f"e.g. mu*{var_name} - {var_name}^3")
        self.noise = QLineEdit("0")
        self.init = FloatField(0.0)

        self.force_kind = QComboBox()
        for _, label in _FORCE_KINDS:
            self.force_kind.addItem(label)
        self.force_stack = QStackedWidget()
        self._force_fields = {}                      # kind -> {pname: FloatField}
        self._exp_grow = None
        for kind, _ in _FORCE_KINDS:
            page = QWidget()
            pf = QFormLayout(page)
            pf.setContentsMargins(0, 0, 0, 0)
            if kind:
                fields = {}
                for pname in FORCING_PARAM_NAMES[kind]:
                    fields[pname] = FloatField(_FORCE_DEFAULTS[pname])
                    pf.addRow(pname, fields[pname])
                if kind == "exponential":
                    self._exp_grow = QRadioButton("Grow (+t/tau)")
                    self._exp_decay = QRadioButton("Decay (-t/tau)")
                    self._exp_decay.setChecked(True)
                    sign_row = QHBoxLayout()
                    sign_row.addWidget(self._exp_grow)
                    sign_row.addWidget(self._exp_decay)
                    pf.addRow(sign_row)
                self._force_fields[kind] = fields
            self.force_stack.addWidget(page)
        self.force_kind.currentIndexChanged.connect(self.force_stack.setCurrentIndex)

        add_help_row(form, "drift", self.drift, HELP["drift"])
        add_help_row(form, "D", self.noise, HELP["D"])
        add_help_row(form, "init", self.init, HELP["init"])
        add_help_row(form, "forcing", self.force_kind, HELP["forcing"])
        form.addRow(self.force_stack)

    def values(self) -> dict:
        kind = _FORCE_KINDS[self.force_kind.currentIndex()][0]
        forcing = None
        if kind:
            forcing = {"kind": kind,
                       "params": {p: f.value() for p, f in self._force_fields[kind].items()}}
            if kind == "exponential":
                forcing["sign"] = 1 if self._exp_grow.isChecked() else -1
        return {"name": self.var_name, "drift": self.drift.text(), "D": self.noise.text() or "0",
                "init": self.init.value(), "forcing": forcing}

    def populate(self, v: dict) -> None:
        self.drift.setText(str(v.get("drift", "")))
        self.noise.setText(str(v.get("D", "0")))
        self.init.setText(repr(float(v.get("init", 0.0))))
        forcing = v.get("forcing") or None
        kind = forcing["kind"] if forcing else ""
        idx = next((i for i, (k, _) in enumerate(_FORCE_KINDS) if k == kind), 0)
        self.force_kind.setCurrentIndex(idx)
        if forcing:
            for pname, fld in self._force_fields[kind].items():
                fld.setText(repr(float(forcing["params"].get(pname, _FORCE_DEFAULTS[pname]))))
            if kind == "exponential":
                # Radios are auto-exclusive: check the TARGET, never un-check the other.
                (self._exp_grow if forcing.get("sign", -1) == 1 else self._exp_decay).setChecked(True)


class ModelBuilderScreen(QWidget):
    def __init__(self, on_saved=None, on_back=None, parent=None):
        """``on_saved(name)`` fires after a successful save (MainWindow refreshes the model combos);
        ``on_back()`` returns to the Settings screen."""
        super().__init__(parent)
        self._on_saved = on_saved
        self._on_back = on_back
        self._editing_name = None                 # set by load_existing(); None = creating new
        self._var_rows = []                       # [_VarRow] in declared order
        self._param_fields = {}                   # name -> FloatField (rebuilt by _detect_params)

        heading = QLabel("Model builder")
        heading.setProperty("type", "heading")

        definition = QGroupBox("Definition")
        dform = QFormLayout(definition)
        self.name_edit = QLineEdit()
        self.vars_edit = QLineEdit()
        self.vars_edit.setPlaceholderText("x, y")
        btn_vars = QPushButton("Set variables")
        btn_vars.clicked.connect(self._set_variables)
        vrow = QHBoxLayout()
        vrow.addWidget(self.vars_edit, 1)
        vrow.addWidget(btn_vars)
        add_help_row(dform, "name", self.name_edit, HELP["name"])
        vrow_w = QWidget()
        vrow_w.setLayout(vrow)
        add_help_row(dform, "variables", vrow_w, HELP["variables"])

        self._rows_host = QVBoxLayout()           # the per-variable _VarRow stack

        params_box = QGroupBox("Parameter values")
        pv = QVBoxLayout(params_box)
        btn_detect = QPushButton("Detect parameters")
        btn_detect.clicked.connect(self._detect_params)
        self._params_form = QFormLayout()
        pv.addWidget(btn_detect)
        pv.addLayout(self._params_form)

        scales_box = QGroupBox("Display scales")
        sform = QFormLayout(scales_box)
        self.x_scale = FloatField(10.0)
        self.t_scale = FloatField(0.01)
        add_help_row(sform, "x_scale (nm)", self.x_scale, HELP["x_scale"])
        add_help_row(sform, "t_scale (s)", self.t_scale, HELP["t_scale"])

        self.status = QLabel("")
        self.status.setWordWrap(True)

        btn_validate = QPushButton("Validate")
        btn_validate.clicked.connect(self._validate_clicked)
        self.btn_save = QPushButton("Save model")
        self.btn_save.setProperty("accent", True)              # primary CTA (Fluent accent)
        self.btn_save.clicked.connect(self._save)
        btn_back = QPushButton("Back to settings")
        btn_back.clicked.connect(lambda: self._on_back and self._on_back())
        btns = QHBoxLayout()
        btns.addWidget(btn_validate)
        btns.addWidget(self.btn_save)
        btns.addStretch(1)
        btns.addWidget(btn_back)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.addWidget(heading)
        iv.addWidget(definition)
        iv.addLayout(self._rows_host)
        iv.addWidget(params_box)
        iv.addWidget(scales_box)
        iv.addWidget(self.status)
        iv.addLayout(btns)
        iv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    # ── dynamic rows ──────────────────────────────────────────────────────────
    def _set_variables(self):
        names = [n.strip() for n in self.vars_edit.text().split(",") if n.strip()]
        if not names:
            self._set_status("Type at least one variable name (e.g. `x` or `x, y`).", error=True)
            return
        old = {row.var_name: row.values() for row in self._var_rows}
        for row in self._var_rows:
            self._rows_host.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._var_rows = []
        for name in names:
            row = _VarRow(name)
            if name in old:
                row.populate(old[name])           # keep typed content across a variable-list edit
            self._rows_host.addWidget(row)
            self._var_rows.append(row)
        self._set_status(f"{len(names)} variable(s): fill in each equation, then Detect parameters.")

    def _detect_params(self):
        variables = [row.values() for row in self._var_rows]
        if not variables:
            self._set_status("Set the variables first.", error=True)
            return
        try:
            compiled = parse_user_model(variables)
        except ModelParseError as e:
            self._set_status(str(e), error=True)
            return
        old = {name: fld.value() for name, fld in self._param_fields.items()}
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_fields = {}
        for name in compiled.param_names:
            fld = FloatField(old.get(name, 1.0))
            self._param_fields[name] = fld
            self._params_form.addRow(name, fld)
        self._set_status(f"Found {len(compiled.param_names)} parameter(s): "
                         f"{', '.join(compiled.param_names) or '(none)'}. Set values, then Validate.")
        return compiled

    # ── validate / save ───────────────────────────────────────────────────────
    def _assemble_doc(self) -> dict:
        return {
            "schema_version": model_store.SCHEMA_VERSION,
            "name": self.name_edit.text().strip(),
            "variables": [row.values() for row in self._var_rows],
            "params": {name: fld.value() for name, fld in self._param_fields.items()},
            "rescale": {"x_scale": self.x_scale.value(), "t_scale": self.t_scale.value()},
        }

    def _validate(self):
        """Parse + compile + a short batch-1 smoke integration. Returns the doc, or None (status set)."""
        if BasePanel._running:
            # The smoke integration's tqdm writes to the process-wide redirected streams while a
            # worker runs -- it would interleave into that run's progress pane. Refuse instead.
            self._set_status("A task is running -- wait for it to finish before validating.", error=True)
            return None
        if not self._var_rows:
            self._set_status("Set the variables first.", error=True)
            return None
        try:
            name = model_store.validate_name(self.name_edit.text())
        except ValueError as e:
            self._set_status(str(e), error=True)
            return None
        doc = self._assemble_doc()
        doc["name"] = name
        try:
            compiled = parse_user_model(doc["variables"])
        except ModelParseError as e:
            self._set_status(str(e), error=True)
            return None
        missing = [p for p in compiled.param_names if p not in self._param_fields]
        if missing:
            self._set_status(f"New parameter(s) {missing} have no value yet -- "
                             "click 'Detect parameters' first.", error=True)
            return None
        doc["params"] = {p: doc["params"][p] for p in compiled.param_names}
        t_scale = doc["rescale"]["t_scale"]
        if doc["rescale"]["x_scale"] <= 0 or t_scale <= 0:
            self._set_status("x_scale and t_scale must be > 0.", error=True)
            return None

        # Smoke integration at the stream's fine ND step (dt_exp over the t_scale upper bound).
        try:
            n_vars = len(compiled.var_names)
            params = torch.tensor([[doc["params"][p] for p in compiled.param_names]])
            force = torch.zeros((1, n_vars, _SMOKE_STEPS + 1))
            model = UserModel(compiled, torch.unbind(params, dim=1), force, batch_size=1)
            inits = torch.tensor([[float(v["init"]) for v in doc["variables"]]])
            dt_nd = DT_EXP_S / (2.0 * t_scale)
            res = sdeint.Solver().euler(model, inits, (0.0, _SMOKE_STEPS * dt_nd), _SMOKE_STEPS + 1,
                                        state_dep_drift=compiled.state_dep_noise)
            if not bool(torch.isfinite(res).all()):
                self._set_status("Validation integration diverged (NaN/inf) -- check the drift/noise "
                                 "expressions or the initial conditions.", error=True)
                return None
        except Exception as e:                       # noqa: BLE001 -- any numeric failure is user-facing
            self._set_status(f"Validation integration failed: {e}", error=True)
            return None
        return doc

    def _validate_clicked(self):
        doc = self._validate()
        if doc is not None:
            self._set_status(f"'{doc['name']}' is valid: {len(doc['variables'])} variable(s), "
                             f"{len(doc['params'])} parameter(s). Ready to save.")

    def _save(self):
        if BasePanel._running:
            self._set_status("A task is running -- wait for it to finish before saving.", error=True)
            return
        doc = self._validate()
        if doc is None:
            return
        json_path = config.MODELS_PATH / f"{doc['name']}.json"
        if doc["name"] != self._editing_name and json_path.exists():
            self._set_status(f"A model named '{doc['name']}' already exists -- edit it from Settings "
                             "or pick another name.", error=True)
            return
        try:
            model_store.save_user_model(doc)
        except Exception as e:                       # noqa: BLE001 -- surface, keep the screen alive
            self._set_status(f"Save failed: {e}", error=True)
            return
        registry.load_user_models()                  # (re-)register; combos are refreshed by on_saved
        self._editing_name = doc["name"]
        self._set_status(f"Saved '{doc['name']}'. It is now available in the Simulate model list.")
        if self._on_saved:
            self._on_saved(doc["name"])

    # ── edit / reset ─────────────────────────────────────────────────────────
    def load_existing(self, name: str) -> None:
        """Populate the form from a saved model (the Settings 'Edit' flow). Renaming saves a copy."""
        doc = model_store.load_user_model(config.MODELS_PATH / f"{str(name).upper()}.json")
        self._editing_name = doc["name"]
        self.name_edit.setText(doc["name"])
        self.vars_edit.setText(", ".join(v["name"] for v in doc["variables"]))
        self._set_variables()
        for row, v in zip(self._var_rows, doc["variables"]):
            row.populate(v)
        self.x_scale.setText(repr(float(doc["rescale"]["x_scale"])))
        self.t_scale.setText(repr(float(doc["rescale"]["t_scale"])))
        self._detect_params()
        for pname, fld in self._param_fields.items():
            if pname in doc["params"]:
                fld.setText(repr(float(doc["params"][pname])))
        self._set_status(f"Editing '{doc['name']}'. Renaming saves a copy under the new name.")

    def reset(self) -> None:
        """Blank the form (the Settings 'Open model builder' flow)."""
        self._editing_name = None
        self.name_edit.clear()
        self.vars_edit.clear()
        for row in self._var_rows:
            self._rows_host.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._var_rows = []
        while self._params_form.rowCount():
            self._params_form.removeRow(0)
        self._param_fields = {}
        self._set_status("")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(("⚠ " if error else "") + text)
