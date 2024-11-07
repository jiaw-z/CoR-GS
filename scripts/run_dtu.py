import os


scenes = ['scan8', 'scan21', 'scan30', 'scan31', 'scan34', 'scan38', 'scan40', 'scan41', 'scan45', 'scan55', 'scan63', 'scan82', 'scan103', 'scan110', 'scan114']
gpu_id = 0

for scan in scenes:
    cmd = f"bash scripts/run_dtu.sh {gpu_id} {scan} ./output/DTU/{scan}/randbg"
    print(cmd)
    os.system(cmd)