import os
import json

def nodes():
    '''
    Retruns a python list of node names from PBS_NODEFILE
    '''

    lines = []
    with open (os.getenv('PBS_NODEFILE')) as nfile:
        lines = nfile.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def tf_config(port=123456):
    print(nodes())
    print(os.getenv('PMI_RANK'))

    config = {
            'cluster': {
                'worker': nodes()
                },
            'task': {
                'type': 'worker',
                'index': os.getenv('PMI_RANK')
                }
            }

    # add port to each worker
    list=[]
    for worker in config['cluster']['worker']:
        list.append(worker + ":" + str(port))
        port = port + 1
    config['cluster']['worker']=list

    return config


def pmi_rank():
    return os.getenv('PMI_RANK')

def test():
    print('{}'.format(nodes()))
    print('RANK {}'.format(pmi_rank()))
    print('{}'.format(json.dumps(tf_config(), indent=4)))


if __name__ == "__main__":
    test()
