#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 08:33:35 2020

@author: nereapruneau
"""

def PGCD (a, b) :
    r = a%b
    gcd = 0
    if r==0 : 
        return b
    else : 
        gcd = PGCD(b, r)
    return gcd
        

print(PGCD(108, 30))    