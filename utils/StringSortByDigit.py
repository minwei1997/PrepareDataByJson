import re

def atoi(text): 
    return int(text) if text.isdigit() else text    # test whether the text only have digit or not. 

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]    # split text by string and digit seperately


if __name__ == '__main__':

    alist=[
    "something1",
    "something12",
    "something17",
    "something2",
    "something25",
    "something29"]
    
    alist.sort(key=natural_keys)
    print(alist)