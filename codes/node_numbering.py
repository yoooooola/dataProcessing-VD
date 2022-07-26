import re
from multiprocessing import Pool
from listup import *


def graph_numbering(f):
    # regular expressions
    p = re.compile('\"\d+\" \[') # specify line for node
    q = re.compile('\"\d+\"') # for node id
    #r = re.compile('\"\d+\"\s\s') # for edge

    dic = {} # key-value pairs
    num = 0 # start numbering from 0

    fp = open(f, "r+") # open file with read and write mode
    lines = fp.readlines()
    new_lines = []

    for line in lines:
        if p.match(line) is not None: # is node - create key-value pair
            node_id = q.findall(line)[0]
            changed_id = "\"" + str(num) + "\"" # add " "
            dic[node_id] = changed_id # add key-value pair
            line = line.replace(node_id, changed_id) # replace the node id
            num += 1 # update num
        
        elif line.find("->") != -1: # is edge - just replace
            all_node = q.findall(line)
            if all_node is not None:
                for id in all_node:
                    if dic.get(id) != None: # key exist
                        changed_id = dic[id] # take value with key
                        line = line.replace(id, changed_id) # replace
                    else: # key does not exist -> remove the line from files
                        line = ""
                        continue
        
        else:
            pass

        if line != "":
            new_lines.append(line)

    fp.seek(0) # change the location of file pointer to overwrite the file
    fp.writelines(new_lines) # overwrite the file
    fp.truncate() # overwrite inplace

    fp.close()

    return


def run_graph_numbering(path):
    # path should be the output path of graph extraction
    proclist = listup(path, "d")

    pool = Pool()
    with pool:
        pool.map(graph_numbering, proclist)
    
    return
