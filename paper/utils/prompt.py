from termcolor import colored, cprint

def print_title(t):
    s = '# -------------------------------'
    print(f'{s}\n# {t}\n{s}')

def print_stats(name, s):
    print(f'{name}: ({s[0]:.2f}, {s[1]:.2e})')

def highlight(*s, color = 'red'):
    ext = colored(f'[HERE]', color, attrs=['bold', 'blink'])
    print(ext, *s)