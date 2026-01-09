ACTION_DICT = {
    "noop": 0,
    "tab": 1,
    "content": 2,
    "selection_command": 3,
    "selection_mouse": 4,
    "selection_keyboard": 5,
    "terminal_command": 6,
    "terminal_output": 7,
    "terminal_focus": 8,
    "git_branch_checkout": 9
}

NUM_ACTIONS = len(ACTION_DICT)

KEY_DICT = {
    "<pad>": 0,      # padding / no key
    "<unk>": 1,      # unknown key
    "<backspace>": 2,
    "<delete>": 3,
    "<enter>": 4,
    "<tab>": 5,
    "<space>": 6,
    # Lowercase letters (7-32)
    "a": 7, "b": 8, "c": 9, "d": 10, "e": 11, "f": 12, "g": 13,
    "h": 14, "i": 15, "j": 16, "k": 17, "l": 18, "m": 19, "n": 20,
    "o": 21, "p": 22, "q": 23, "r": 24, "s": 25, "t": 26, "u": 27,
    "v": 28, "w": 29, "x": 30, "y": 31, "z": 32,
    # Uppercase letters (33-58)
    "A": 33, "B": 34, "C": 35, "D": 36, "E": 37, "F": 38, "G": 39,
    "H": 40, "I": 41, "J": 42, "K": 43, "L": 44, "M": 45, "N": 46,
    "O": 47, "P": 48, "Q": 49, "R": 50, "S": 51, "T": 52, "U": 53,
    "V": 54, "W": 55, "X": 56, "Y": 57, "Z": 58,
    # Digits (59-68)
    "0": 59, "1": 60, "2": 61, "3": 62, "4": 63,
    "5": 64, "6": 65, "7": 66, "8": 67, "9": 68,
    # Punctuation and symbols (69-95)
    "!": 69, '"': 70, "#": 71, "$": 72, "%": 73, "&": 74, "'": 75,
    "(": 76, ")": 77, "*": 78, "+": 79, ",": 80, "-": 81, ".": 82,
    "/": 83, ":": 84, ";": 85, "<": 86, "=": 87, ">": 88, "?": 89,
    "@": 90, "[": 91, "\\": 92, "]": 93, "^": 94, "_": 95, "`": 96,
    "{": 97, "|": 98, "}": 99, "~": 100,
}

NUM_KEYS = len(KEY_DICT)