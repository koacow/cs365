# SYSTEM IMPORTS
import os
import scipy.io as spio
import scipy.sparse as sp
import requests
import tarfile


# PYTHON PROJECT IMPORTS


def load() -> sp.coo_matrix:
    local_filename: str = "p2p-Gnutella09"
    cd: str = os.path.abspath(os.path.dirname(__file__)) # get path to the directory this file is in
    data_file_path: str = os.path.join(cd, "%s.mtx" % local_filename) # data file we want

    if not os.path.exists(data_file_path):
        # download the tar file
        http_url: str = "https://www.cise.ufl.edu/research/sparse/MM/SNAP/%s.tar.gz" % local_filename
        req = requests.get(http_url)
        if req.status_code != 200:
            raise Exception("ERROR is downloading data file. Status code [%s]" % r.status_code)
        tarfile_path: str = os.path.join(cd, "%s.tar.gz" % local_filename)
        with open(tarfile_path, "wb") as f:
            f.write(req.content)

        # uncompress only the .mtx file inside
        with tarfile.open(tarfile_path, "r:gz") as tf:
            print(tf.getmembers())
            print(tf.getnames())
            with tf.extractfile("{0}/{0}.mtx".format(local_filename)) as ef: # path=target_path)
                with open(data_file_path, "wb") as f:
                    f.write(ef.read())

    A: sp.coo_matrix = spio.mmread(data_file_path)
    A.setdiag(1)
    return A


if __name__ == "__main__":
    # test
    A: sp.coo_matrix = load()
    print(type(A), A.shape)

