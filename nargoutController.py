# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:12:22 2019

Rewritten by: Mohsen Ghamary Asl (m.ghamary@gmail.com)

Reference: https://stackoverflow.com/questions/14147675/nargout-in-python
"""

import traceback

def nargoutController(*args):
    
   callInfo = traceback.extract_stack()
   
   callLine = str(callInfo[-3].line)
   
   split_equal = callLine.split('=')
   
   split_comma = split_equal[0].split(',')
   
   num = len(split_comma)
   
   
   return args[0:num] if num > 1 else args[0]