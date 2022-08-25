import os
import json

def mpi_rank():
    if 'PMI_RANK' in os.environ:
        return os.getenv('PMI_RANK')
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return os.getenv('OMPI_COMM_WORLD_RANK')


def nodes():
    nodefile = 'hostfile'
    if 'PBS_NODEFILE' in os.environ:
        nodefile = os.getenv('PBS_NODEFILE')

        print(nodefile)

        lines = []
        with open (nodefile) as nfile:
            lines = nfile.readlines()
            lines = [line.rstrip() for line in lines]
    
    else:
        lines = ['rbdgx1','rbdgx2']


    return lines

def tf_config(port=123456):
    config = {
            'cluster': {
                'worker': nodes()
                },
            'task': {
                'type': 'worker',
                'index': mpi_rank()
                }
            }

    # add port to each worker
    list=[]
    for worker in config['cluster']['worker']:
        list.append(worker + ":" + str(port))
        port = port + 1
    config['cluster']['worker']=list

    return config


def test():
    print(nodes())
    print('{}'.format(json.dumps(tf_config())))
    #print(mpi_rank())


if __name__ == "__main__":
    test()
