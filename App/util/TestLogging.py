file = None

def init_logger(filename):
    global file
    file = open(filename, 'a')
    file.write("--- Logger started ---\n")
    file.flush()

def log(message):
    global file
    file.write(f"{message}\n")
    file.flush()
