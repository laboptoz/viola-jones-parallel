#!/usr/bin/python

import os
import subprocess

num_runs = 5
vj_seq = './vj'

def kill_process():
  pass

def make_output_dir():
  pass

def run():
  sum_time = 0
  for i in range(num_runs):
    p = subprocess.Popen([vj_seq], stdout=subprocess.PIPE)
    out = p.stdout.read()
    time_txt = out.split('Total Execution Time: ')[1]
    time = time_txt.split(' ')[0]

    sum_time += int(time) 
    print('----------------- Run #' + str(i) + ' -----------------')
    print(out)
  
  avg_time = sum_time / num_runs
  print('================ Final Results (' + str(num_runs) + ' runs) ================')
  print('Average Total Execution Time: ' + str(avg_time) + ' microseconds')

def copy_to_current():
  pass

if __name__ == "__main__":
  run()