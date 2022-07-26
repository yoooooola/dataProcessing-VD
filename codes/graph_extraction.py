import os
from tqdm import tqdm
from glob import glob
import re
from multiprocessing import Pool

from listup import *


def mkdir_move(dirpath):
    # path should be the directory path
    flist = listup(dirpath, "c")

    for f in tqdm(flist, desc="mkdir and move"):
        fname = f.split(".c")[0]
        fname_split = fname.split("/")[-1]

        # dir/000.c
        if len(fname_split) < 3:
            newdir_name = dirpath + "/0/"

            if not os.path.exists(newdir_name):
                os.system(f"mkdir {newdir_name}")
            
            newdir_name = newdir_name + fname_split[-1] 

        # dir/cwe/000.c
        else:
            newdir_name = dirpath + "/" + fname_split[-1] 

        # run mkdir command
        os.system(f"mkdir {newdir_name}")

        # run mv command
        mv_path = newdir_name + "/"
        os.system(f"mv {f} {mv_path}")


def joern_command(dirpath):
    dlist = glob((dirpath + "/*"))

    output_path = "../joern_add/"

    if not os.path.exists(output_path):
        os.system(f"mkdir {output_path}")

    dirname = dirpath.split("/")[-1]
    output_mkdir = output_path + dirname + "/"

    os.system(f"mkdir {output_mkdir}")

    # extract graphs with joern
    for d in tqdm(dlist, desc="joern command"):
        os.system(f"joern-parse {d}")
        output_dir = output_mkdir + d.split("/")[-1]
        print(output_dir)
        os.system(f"joern-export cpg.bin --repr cpg14 --out {output_dir}")


def concat_graphs(dirpath):
    # regular expression
    p = re.compile('\"\d+\" \[') # distinguish node
    q = re.compile('\d+') # extract node id

    # dlist = glob("../joern/*") # original code
    glob_path = "../joern_add/" + dirpath + "/*"
    dlist = glob(glob_path)
    flist_dir = []

    if len(dlist) == 0:
        dlist.append(("../joern_add/"))

    for d in dlist:
        flist = []
        for (root, directories, files) in os.walk(d):
            for file in files:
                file_path = os.path.join(root, file)
                flist.append(file_path)
        flist_dir.append(flist)

    # dlist indexing
    dir_idx = 0

    # traverse each dir
    for flist in tqdm(flist_dir, desc="concatenation"):
        newlines = []
        node_set = set() # to avoid duplication

        global_num = 0

        for f in flist:
            with open(f, "r") as fr:
                flines = []
                flines = fr.readlines()

                global_flag = False

                for line in flines:
                    if line.find("digraph") != -1 and line.find("<global>") != -1:
                        global_flag = True
                        global_num += 1
            
                    if line.find("digraph \"<operator>.") != -1:
                        break
            
                    # ignore second global graph
                    if global_num > 1: 
                        global_flag = False

                    if global_flag:                
                        if line.find("}") != -1:
                            continue

                        # save node id in global graph
                        if p.match(line) is not None: # node
                            node_id = q.findall(line)[0]
                            node_id = int(node_id)
                            node_set.add(node_id)

                        newlines.append(line)

                    else: # global graph X
                        if line.find("digraph") != -1 or line.find("}") != -1: # removal of graph structure
                            continue

                        # save node only if in global graph
                        all_id = q.findall(line)
                        if all_id is not None:
                            for id in all_id:
                                node_id = int(id)
                                if node_id in node_set:
                                    newlines.append(line)

        newlines.append("}")

        d_name = dlist[dir_idx].split("/")[-1]
    
        dir_idx += 1

        save_path = "../complete_add/"

        # save as new file
        if not os.path.exists(save_path):
            os.system(f"mkdir {save_path}")

        newdir_path = save_path + dirpath

        if (dir_idx == 1) and (os.path.exists(newdir_path) != True):
            os.system(f"mkdir {newdir_path}")
        
        newfile_path = newdir_path + "/" + d_name + ".dot"

        with open(newfile_path, "w") as newfile:
            # print(newfile_path)
            # using set() causes disorder so that I uses for loop to remove duplicate lines
            no_dup_newlines = []
            for line in newlines:
                if line not in no_dup_newlines:
                    no_dup_newlines.append(line)

            newfile.writelines(no_dup_newlines)


"""
def command_sequence(dirpath):
    mkdir_move(dirpath)
    joern_command(dirpath)
    concat_graphs(dirpath)
"""


def run_extraction(path):
    # path == removed annotation path
    dlist = listup(path, None, True)

    # complete the path
    for i in range(0, len(dlist)):
        dlist[i] = path + dlist[i]
    
    # multiprocessing
    pool = Pool()
    
    #pool.map(mkdir_move, dlist)

    # for multiple use of mp.pool
    #pool.close()
    #pool.join()

    for d in dlist:
        joern_command(d)

    # set new dlist with directory: ../joern/
    dlist = listup("../joern_add/", None, True)

    
    #pool = Pool()
    with pool:
        pool.map(concat_graphs, dlist)
    
    return