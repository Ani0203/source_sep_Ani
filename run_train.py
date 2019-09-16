#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:24:38 2019

@author: aniruddha
"""

from subprocess import call


command = ['python' , 'train.py']
arguments = ['--target'+'vocals' , '--dataset' + 'musdb' , '--root' + '../rec_data/', '--output'+'../out_unmix' , '--model' + '../model_checkpoint']
command.extend(arguments)
call(command)