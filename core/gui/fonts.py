"""The window's lettering: Inter for everything operated or read off, a book face for the few sentences
the program says, and a fixed-width face for text quoted from the console.

Every font is made here, in code: a style sheet can name a family but cannot switch on the digits of
equal width. Those are on where numbers stand under one another, in lists and tables and their headers,
and for a caller that asks for them. They are not on everywhere, because in the bundled files the same
feature also widens the hyphen and the asterisk to a digit's width (the brackets and other marks change
by a little), and a compound word then reads as if it were spaced. Sizes come from the look's table,
scaled by the text factor. The two bundled Inter files are handed to the toolkit the first time a font
is asked for, so a painted device has its face in a test as well, where no application font is loaded.
"""
import logging
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QGuiApplication
from PySide6.QtWidgets import QApplication

from . import design

log = logging.getLogger(__name__)

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FILES = ("Inter-Regular.ttf", "Inter-SemiBold.ttf")      # the two weights the look uses: 400 and 600
# The widget classes the toolkit keeps a font of their own for: the application's font does not reach them.
_CLASSES = ("QAbstractItemView", "QHeaderView", "QMessageBox", "QMenu", "QMenuBar", "QTipLabel", "QStatusBar")
_IN_COLUMNS = ("QAbstractItemView", "QHeaderView")       # of those, the ones whose numbers stand under one another

_registered = False               # the two files are with the toolkit
_inter: dict = {}                 # (kind, weight, pixel size, equal digits) -> the font, built once
_book_face: str | None = None     # the face a sentence is set in on this computer, once looked up
_with_matplotlib: bool | None = None      # None until tried; then whether both files went in


def _toolkit_is_up() -> bool:
    """Whether the application object exists. Before it does, the toolkit's font database ends the
    process, with no traceback, when it is handed a file or asked for a family; a core application
    with no windowing counts as none, for the same reason."""
    return isinstance(QGuiApplication.instance(), QGuiApplication)


def register_fonts() -> None:
    """Hand the two bundled files to the toolkit, once per process. Before the application exists this
    does nothing and remembers nothing, so the first call after it does the work. A file that will not
    load is not an error: the program still starts, in whatever face the toolkit finds, and a warning
    names the file."""
    global _registered
    if _registered or not _toolkit_is_up():
        return
    for name in _FILES:
        if QFontDatabase.addApplicationFont(str(_FONT_DIR / name)) < 0:      # the toolkit's answer for a refusal
            log.warning("The lettering file %s did not load; the window is set in whatever face the toolkit finds.",
                        name)
    _registered = True


def inter(kind: str = "body", weight: int | None = None, *, tabular: bool = False) -> QFont:
    """The lettering font of one row of the look's type table, at the text size in force. ``weight``
    replaces the row's own; the files hold 400 and 600. ``tabular`` switches on the digits of equal
    width, for where numbers stand under one another. They are off unless asked for: the same feature
    widens the hyphen and the asterisk to a digit's width. The caller gets a copy and may change it."""
    register_fonts()                                        # on every call: the application may exist by now
    size, _line, own = design.TYPE[kind]
    weight = weight or own
    key = (kind, weight, design.scaled(size), bool(tabular))    # the size too: a font keeps the size it was built at
    font = _inter.get(key)
    if font is None:
        font = QFont(design.LETTERING)
        font.setPixelSize(key[2])
        font.setWeight(QFont.Weight(weight))
        if tabular:
            font.setFeature(QFont.Tag.fromString("tnum"), 1)    # digits of equal width
        _inter[key] = font
    return QFont(font)


def book_family(available) -> str:
    """The face a sentence is set in: the first book face ``available`` holds, else the lettering face.
    ``available`` is any iterable of family names, and is read once."""
    available = set(available)
    return next((face for face in design.BOOK_FACES if face in available), design.LETTERING)


def _book_face_here() -> str:
    """``book_family`` for this computer, looked up once the application exists and kept from then on."""
    global _book_face
    if _book_face is None:
        if not _toolkit_is_up():
            return design.LETTERING                         # nothing is kept: the next call may find a book face
        register_fonts()
        _book_face = book_family(QFontDatabase.families())
    return _book_face


def book(bold: bool = False) -> QFont:
    """The font of the program's own sentences, at the book's size whichever face is found, so a
    sentence keeps its line height on a computer with no book face. There it is the lettering font at
    that size, and its bold is 600: the bundled files hold no 700."""
    face = _book_face_here()
    if face == design.LETTERING:
        return inter("book", 600 if bold else None)
    font = QFont(face)
    font.setPixelSize(design.scaled(design.TYPE["book"][0]))
    font.setBold(bold)
    return font


def fixed() -> QFont:
    """The fixed-width font for text quoted from the console, at the body's size: the look's faces in
    their order, then the system's own fixed-width face. Nothing is registered and nothing is kept."""
    families = list(design.FIXED_FACES)
    if _toolkit_is_up():
        families.append(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family())
    font = QFont(families[0])                               # built from a name: nothing of the application's font
    font.setFamilies(families)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    font.setPixelSize(design.scaled(design.TYPE["body"][0]))
    return font


def register_with_matplotlib() -> bool:
    """Hand the two bundled files to the figure library as well, so that a figure can name the lettering
    face. Tried once per process and the outcome kept: True when both files went in. Never raises: a
    failure is said in a warning. The figure library is imported here and not at the top, so building a
    font does not load it."""
    global _with_matplotlib
    if _with_matplotlib is None:
        _with_matplotlib = False
        try:
            from matplotlib import font_manager
            for name in _FILES:
                font_manager.fontManager.addfont(str(_FONT_DIR / name))
            _with_matplotlib = True
        except Exception as error:                          # noqa: BLE001 -- a figure then keeps its own face
            log.warning("The lettering files did not reach the figure library; a figure keeps that library's "
                        "own face (%s: %s).", type(error).__name__, error)
    return _with_matplotlib


def apply_class_fonts(app=None) -> None:
    """Set the lettering font on the widget classes the toolkit keeps a font of their own for: menus,
    lists and tables with their headers, message boxes, tool tips and status bars. Lists, tables and
    their headers get it with the digits of equal width, since numbers stand under one another there;
    the other classes get it without, so that a hyphen keeps its own width in them. A change of style,
    the first style sheet and an application font set with no class name each empty these again, so
    this is called after them. ``app`` is the running application when none is handed in; with no
    application nothing is done."""
    app = app if app is not None else QApplication.instance()
    if app is None:
        return
    for name in _CLASSES:
        app.setFont(inter("body", tabular=name in _IN_COLUMNS), name)


def load_app_font(app) -> str:
    """Make the lettering font the application's own, so that every widget inherits it, hand the
    bundled files to the figure library, and return the family the application's font resolved to.
    The digits of equal width are not switched on in it: the same feature widens the hyphen to a
    digit's width, in every word the window shows."""
    register_fonts()
    register_with_matplotlib()
    app.setFont(inter("body"))
    return QFontInfo(app.font()).family()
