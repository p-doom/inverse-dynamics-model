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
    "<enter>": 2,
    "<space>": 3,
    # Lowercase letters (7-32)
    "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10,
    "h": 11, "i": 12, "j": 13, "k": 14, "l": 15, "m": 16, "n": 17,
    "o": 18, "p": 19, "q": 20, "r": 21, "s": 22, "t": 23, "u": 24,
    "v": 25, "w": 26, "x": 27, "y": 28, "z": 29,
    # Uppercase letters (33-58)
    "A": 30, "B": 31, "C": 32, "D": 33, "E": 34, "F": 35, "G": 36,
    "H": 37, "I": 38, "J": 39, "K": 40, "L": 41, "M": 42, "N": 43,
    "O": 44, "P": 45, "Q": 46, "R": 47, "S": 48, "T": 49, "U": 50,
    "V": 51, "W": 52, "X": 53, "Y": 54, "Z": 55,
    # Digits (59-68)
    "0": 56, "1": 57, "2": 58, "3": 59, "4": 60,
    "5": 61, "6": 62, "7": 63, "8": 64, "9": 65,
    # Punctuation and symbols (69-95)
    "!": 66, '"': 67, "#": 68, "$": 69, "&": 70, "'": 71,
    "(": 72, ")": 73, "*": 74, "+": 75, ",": 76, "-": 77, ".": 78,
    "/": 79, ":": 80, ";": 81, "<": 82, "=": 83, ">": 84, "?": 85,
    "@": 86, "[": 87, "\\": 88, "]": 89, "^": 90, "_": 91, 
    "{": 92, "|": 93, "}": 94, 
}

NUM_KEYS = len(KEY_DICT)