import os 


def listup(path, opt="p", dir=False, onlydir=False):
    # list up directories
    if dir:
        dlist = []

        if onlydir:
            for d in os.listdir(path):
                if os.path.exists(d) != False:
                    dlist.append(d)
            
            if len(dlist) == 0:
                dlist.append('')

        else:
            for d in os.listdir(path):
                dlist.append(d)
        
        return dlist
    
    # list up files
    else:
        # only 3 options: c (code) / d (dot) / p (pickle)
        if opt != "c" and opt != "d" and opt != "p":
            print("ERROR: UNAVAILABLE OPTION")
            exit(-1)

        # initialize the list
        flist = []

        for (root, directories, files) in os.walk(path):
            for file in files:
                ext = os.path.splitext(file)[-1]
                filepath = os.path.join(root, file)

                # 코드 너무 조잡함,, 고칠수 있으면 고치기,,
                # source code
                if opt == "c" and (ext == ".c" or ext == ".cpp"):
                    flist.append(filepath)
                # graph files
                elif opt == "d" and ext == ".dot":
                    flist.append(filepath)
                # vector pickles
                else:
                    if ext == ".pickle":
                        flist.append(filepath)
                
        return flist