import termcolor

CMDS ="""
python3 -m typednn.types.tensor
python3 -m typednn.types.image
python3 -m typednn.factory
python3 -m typednn.types.pointcloud
python3 -m typednn.tester.test_shadow
python3 -m typednn.tester.test_kwargs
"""

import os
fail = False
for i in CMDS.split('\n'):
    out = os.system(i.strip())
    if out != 0:
        print('Error in', i)
        fail = True
        break
    
if not fail:
    print(termcolor.colored("Congrats! Passed all test cases!", 'green'))