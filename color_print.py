class ColorPrint:
    # The following methods allow for colored printing

    @staticmethod
    def print_pink(s: str):
        print(ColorPrint.fg.pink + s + ColorPrint.reset)

    @staticmethod
    def print_red(s: str):
        print(ColorPrint.fg.red + s + ColorPrint.reset)

    @staticmethod
    def print_orange(s: str):
        print(ColorPrint.fg.orange + s + ColorPrint.reset)

    @staticmethod
    def print_yellow(s: str):
        print(ColorPrint.fg.yellow + s + ColorPrint.reset)

    @staticmethod
    def print_lime(s: str):
        print(ColorPrint.fg.lightgreen + s + ColorPrint.reset)

    @staticmethod
    def print_green(s: str):
        print(ColorPrint.fg.green + s + ColorPrint.reset)

    @staticmethod
    def print_cyan(s: str):
        print(ColorPrint.fg.lightcyan + s + ColorPrint.reset)

    @staticmethod
    def print_blue(s: str):
        print(ColorPrint.fg.blue + s + ColorPrint.reset)

    @staticmethod
    def print_purple(s: str):
        print(ColorPrint.fg.purple + s + ColorPrint.reset)

    @staticmethod
    def print_gray(s: str):
        print(ColorPrint.fg.lightgrey + s + ColorPrint.reset)

    @staticmethod
    def print_black(s: str):
        print(ColorPrint.fg.black + s + ColorPrint.reset)

    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
