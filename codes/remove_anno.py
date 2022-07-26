import os
import re
from multiprocessing import Pool
from tqdm import tqdm

from listup import *


def remove_annotations(dirpath):
    file_idx = 1
    dirname = dirpath.split("/")[-1]
    flist = listup(dirpath, "c")

    p = "\/\*[\s\S]*?\*\/" # remove /* */
    q = "\/\/[\s\S]*" # remove //

    newdir_path = "../removed_anno/"

    # if no directory for saving code without annotation
    if not os.path.exists(newdir_path):
        os.system(f"mkdir {newdir_path}")

    newdir_name = newdir_path + dirname
    os.system(f'mkdir {newdir_name}')
    print(newdir_name)

    for f in tqdm(flist, desc="remove annotations"):
        # read the file
        with open(f, "r") as fr:
            flines = fr.read()
            
            # remove annotations using regex
            flines = re.sub(p, '', flines)
            flines = re.sub(q, '', flines)

            # naming new file
            ext = os.path.splitext(f)[-1]
            new_file = newdir_name + "/" + str(file_idx) + ext

            # update index
            file_idx += 1
            
            # write as new file
            with open(new_file, "w") as fw:
                fw.write(flines)


def run_remove_annotations(path):
    # path = input path
    dlist = listup(path, None, True)

    """
    # count the number of directories
    count = 0
    for x in os.listdir(path):
        if os.path.isdir(os.path.join(path, x)):
            count += 1

    # if there is no directory, mkdir 0 and move all file to /0
    if count == 0:
        path_ = path + "/0"
        print(path_)
        os.system(f"mkdir {path_}")
        os.system(f"find {path} -maxdepth 1 -type f | xargs mv -t {path_}")
    """
    
    # complete the path
    for i in range(0, len(dlist)):
        dlist[i] = path + dlist[i]

    # multiprocessing
    pool = Pool()
    with pool:
        pool.map(remove_annotations, dlist)

    return