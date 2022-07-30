import os
import json

def nodes():
    lines = []
    with open (os.getenv('PBS_NODEFILE')) as nfile:
        lines = nfile.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def tf_config(port=123456):
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
    print('{}'.format(json.dumps(tf_config())))
    print('{}'.format(json.dumps(tf_config(), indent=4)))
    print(pmi_rank())


if __name__ == "__main__":
    test()
