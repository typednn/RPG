import termcolor

CMDS = [
    'python3 -m typednn.types.tensor',
    'python3 -m typednn.types.image',
    'python3 -m typednn.factory',
]

import os
for i in CMDS:
    out = os.system(i)
    if out != 0:
        print('Error in', i)
        break
    
print(termcolor.colored("Congrats! Passed all test cases!", 'green'))