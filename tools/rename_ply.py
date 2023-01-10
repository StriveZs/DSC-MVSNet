import os
import sys
import os.path as osp

scene_list = ["scan1", "scan4", "scan9", "scan10", "scan11", "scan12", "scan13", "scan15", "scan23",
                  "scan24", "scan29", "scan32", "scan33", "scan34", "scan48", "scan49", "scan62", "scan75",
                  "scan77", "scan110", "scan114", "scan118"]

filepath = ""
out_path = ''

if not os.path.exists(out_path):
    os.mkdir(out_path)

for scene in scene_list:
    scan_path = filepath + '/' + scene
    dirs = os.listdir(scan_path)
    filename = None
    for item in dirs:
        if item[:16] == 'consistencyCheck':
            filename = item
        else:
            # delete operation
            rm_cmd = 'rm -rf ' + filepath + '/' + scene + '/' + item
            os.system(rm_cmd)
    if filename is None:
        continue
    rename_path = scan_path + '/' + filename + '/' + 'final3d_model.ply'
    rename_file = out_path + '/' + 'mvsnet{:0>3}_l3.ply'.format(scene[4:])
    rename_cmd = "cp -r " + rename_path + " " + rename_file
    os.system(rename_cmd)
print('rename over and delete over')