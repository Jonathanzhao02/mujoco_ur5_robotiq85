import h5py
import numpy as np

DIST_MAX = 0.75
AVG_CHANGE = 0.15

if __name__ == '__main__':
    targets_type = np.dtype([('obj1', 'S64'), ('obj2', 'S64'), ('pos', '<f4', 3), ('rot', '<f4', 3)])
    objective_type = np.dtype([('timestep', 'uint'), ('action', 'S64'), ('targets', targets_type)])

    valid = 0

    for i in range(1, 2001):
        with h5py.File(f'demos/demo{i}.data', 'r') as f:
            # del f['objective']

            if f.attrs['success']:
                valid_demo = False
                objs = f['objs']
                start = f['pos'][0]
                end = f['pos'][f.attrs['final_timestep'] - 1]

                dpos = end - start
                dist = np.linalg.norm(dpos, axis=1)[1:] # exclude EE

                if np.max(dist) < DIST_MAX and np.sum(dist > AVG_CHANGE) <= 1:
                    idx = -1
                
                    for k in range(len(objs) - 1):
                        if dist[k] > AVG_CHANGE:
                            idx = k + 1
                    
                    if idx != -1:
                        sim_idx = -1
                        sim = np.inf

                        for k in range(1,len(objs)):
                            d = np.linalg.norm(end[k] - end[idx])
                            if k != idx and d < sim:
                                sim = d
                                sim_idx = k

                        obj1 = bytes.decode(objs[idx])
                        obj2 = bytes.decode(objs[sim_idx])

                        # objectives = f.create_group('objectives')
                        # obj0 = objectives.create_group('0')
                        # obj0.attrs['timestep'] = 0
                        # obj0.attrs['action'] = 'stack'
                        # obj0.attrs['obj1'] = bytes.decode(objs[idx])
                        # obj0.attrs['obj2'] = bytes.decode(objs[sim_idx])

                        if sim < 0.08:
                            valid_demo = True
                            print(f'{i} : {obj1} onto {obj2}')
            
                if not valid_demo:
                    # f.attrs['success'] = False
                    print(f'invalid for {i}')
                else:
                    valid += 1

    print(valid)

    # valid = 0

    # for i in range(1,2000):
    #     with h5py.File(f'demos/demo{i}.data', 'r') as f:
    #         if f.attrs['success']:
    #             objs = f['objs']
    #             start = f['pos'][0]
    #             end = f['pos'][f.attrs['final_timestep'] - 1]

    #             if valid > 1:
    #                 import code
    #                 code.interact(local=locals())

    #             dpos = end - start
    #             dist = np.linalg.norm(dpos, axis=1)[1:] # exclude EE

    #             if np.max(dist) < DIST_MAX:
    #                 if np.sum(dist > AVG_CHANGE) <= 1:
    #                     valid += 1
    
    # print(valid)

