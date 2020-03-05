import os

# contruct path relative to location of current script
def relpath(relp):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relp)
    return(filename)
