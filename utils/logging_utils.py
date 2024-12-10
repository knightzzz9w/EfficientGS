import rich
import time

_log_styles = {
    "MonoGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="MonoGS"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)






class TicToc:
    def __init__(self):
        self.start_time = None
    def tic(self):
        self.start_time = time.time()

    def toc(self):
        temp_time = self.start_time
        self.start_time = time.time()
        return (time.time() - temp_time)*1e3
