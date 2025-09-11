from scipy.interpolate import BSpline, insert
import numpy as np
import math
import torch
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from scipy.optimize import root_scalar

'''def transformSplines(splines, new_start, new_end):
    pos_count = splines[0].c.shape[0]

    x_pos = np.array([splines[0].c[i, 1] for i in range(pos_count)])
    y_pos = np.array([splines[1].c[i, 1] for i in range(pos_count)])
    z_pos = np.array([splines[2].c[i, 1] for i in range(pos_count)])

    pos = np.ones((4, pos_count))
    pos[0, :] = x_pos
    pos[1, :] = y_pos
    pos[2, :] = z_pos

    old_start = pos[:3, 0]
    old_end = pos[:3, pos_count-1]

    if new_start is None:
        new_start = old_start
    if new_end is None:
        new_end = old_end

    transMat1 = np.identity(4)
    transMat1[:3, 3] = -old_start

    transMat2 = np.identity(4)
    transMat2[:3, 3] = new_start

    #print(transMat)

    old_dir = old_end - old_start
    new_dir = new_end - new_start

    old_scale = np.linalg.norm(old_dir)
    new_scale = np.linalg.norm(new_dir)


    #scale = np.full((3,3), 1)
    scaleMat = np.identity(4)
    #scaleMat[:3, :3] = scale
    scaleMat[np.diag_indices(3)] = new_scale / old_scale
    #scaleMat[0,0] = 0.01
    #scaleMat[1,1] = 3
    scaleMat[2,2] = 1

    #print(scaleMat)

    old_dir /= old_scale
    new_dir /= new_scale

    v = np.cross(old_dir, new_dir)
    c = np.dot(old_dir, new_dir)

    v_skew = np.zeros((3,3))
    v_skew[1,0] = v[2]
    v_skew[0,1] = -v[2]
    v_skew[2,0] = -v[1]
    v_skew[0,2] = v[1]
    v_skew[2,1] = v[0]
    v_skew[1,2] = -v[0]

    R = np.identity(3) + v_skew + np.dot(v_skew, v_skew) / (1 + c)

    rotMat = np.identity(4)
    rotMat[:3, :3] = R

    transformMat = transMat2 @ scaleMat @ rotMat @ transMat1

    transPos = (transformMat @ pos)[:3, :]

    for i in range(pos_count):
        splines[0].c[i, 1] = transPos[0, i]
        splines[1].c[i, 1] = transPos[1, i]
        splines[2].c[i, 1] = transPos[2, i]


    return splines'''

def transformSplines(coordinates, new_start, new_end):
    start_shift = new_start - coordinates[:, 0]
    end_shift = new_end - coordinates[:, coordinates.shape[1]-1]


    # determine when hand moves object
    start_frame = 0
    for i in range(0, coordinates.shape[1]):
        #print(np.linalg.norm(coordinates[:, 0] - coordinates[:, i]))
        if np.linalg.norm(coordinates[:, 0] - coordinates[:, i]) < 0.01:
            start_frame += 1
        else:
            break

    # determine when hand stops moving object
    end_frame = start_frame + 1
    for i in range(end_frame, coordinates.shape[1]):
        if np.linalg.norm(coordinates[:, coordinates.shape[1]-1] - coordinates[:, i]) > 0.02:
            end_frame += 1
        else:
            break

    #print(start_frame, end_frame)
    start_frame = 0
    end_frame = coordinates.shape[1]


    # shift frames before start_frame
    for i in range(0, start_frame):
        coordinates[:, i] += start_shift

    # shift frames after end_frame
    for i in range(end_frame, coordinates.shape[1]):
        coordinates[:, i] += end_shift

    # shift frames between start and end frames by object_shift * weight
    for i in range(start_frame, end_frame):
        weight = (1 - (i - start_frame) / (end_frame - start_frame))**2
        coordinates[:, i] += weight * start_shift

        weight = ((i - start_frame) / (end_frame - start_frame))**2
        coordinates[:, i] += weight * end_shift
    
    return coordinates


# determine when the object starts moving using a Gaussian mixture model
def motionStartEnd(pos):
    # speed = np.linalg.norm(np.diff(pos, axis=1), axis = 0).reshape(-1, 1)

    # gmm = GaussianMixture(n_components=2)
    # labels = gmm.fit_predict(speed)

    # cluster_means = [np.mean(speed[labels == label]) for label in [0, 1]]
    # print(cluster_means)
    # rest_label = np.argmin(cluster_means)
    # motion_label = 1 - rest_label

    pos_count = pos.shape[1]
    labels = np.zeros(pos_count)
    rest_height = pos[2, 0]

    # detect "moving" by height
    for i in range(pos_count):
        if abs(pos[2, i] - rest_height) > 0.01:
            labels[i] = 1

    label_changes = np.where(np.diff(labels) != 0)[0] + 1

    start_index = None
    stop_index = None

    minimum_consecutive = 0.05

    c = math.ceil(pos_count * minimum_consecutive)
    # minimum_consecutive percent of the points must be resting to determine idx
    for idx in label_changes:
        if idx < c or idx >= len(labels)-3:
            continue
        if labels[idx-c:idx].all() == 0 and labels[idx:idx+3].all() == 1:
            start_index = idx
            break

    for idx in label_changes[::-1]:
        if idx < 3 or idx >= len(labels)-c:
            continue
        if labels[idx-3:idx].all() == 1 and labels[idx:idx+c].all() == 0:
            stop_index = idx
            break

    if start_index is None or stop_index is None:
        raise RuntimeError("Unable to determine start and stop indices.")

    print("Motion start index: ", start_index)
    print("Motion end index: ", stop_index)

    dist = np.sqrt(pos[0, :]**2 + pos[1, :]**2 + pos[2, :]**2)

    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(pos[0, :])), pos[0, :], label='Pos', color='gray')
    plt.scatter(np.arange(len(pos[0, :]))[labels == 0], pos[0, labels == 0],
                color='blue', label='Rest', alpha=0.6)
    plt.scatter(np.arange(len(pos[0, :]))[labels == 1], pos[0, labels == 1],
                color='red', label='Motion', alpha=0.6)
    
    plt.plot(np.arange(len(pos[1, :])), pos[1, :], label='Pos', color='gray')
    plt.scatter(np.arange(len(pos[1, :]))[labels == 0], pos[1, labels == 0],
                color='blue', label='Rest', alpha=0.6)
    plt.scatter(np.arange(len(pos[1, :]))[labels == 1], pos[1, labels == 1],
                color='red', label='Motion', alpha=0.6)
    
    plt.plot(np.arange(len(pos[2, :])), pos[2, :], label='Pos', color='gray')
    plt.scatter(np.arange(len(pos[2, :]))[labels == 0], pos[2, labels == 0],
                color='blue', label='Rest', alpha=0.6)
    plt.scatter(np.arange(len(pos[2, :]))[labels == 1], pos[2, labels == 1],
                color='red', label='Motion', alpha=0.6)
    
    plt.xlabel('Time')
    plt.ylabel('Pos')
    

    # plt.plot(np.arange(len(pos[0, :])), pos[2, :], label="z")
    # plt.plot(np.arange(len(pos[0, :])), pos[1, :], label="y")
    # plt.plot(np.arange(len(pos[0, :])), pos[0, :], label="x")
    # plt.plot(np.arange(len(pos[0, :])), dist, label="dist")
    # plt.plot(np.arange(len(pos[0, :])-1), speed, label="speed")
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig('startEndGraph.png')

    return start_index, stop_index


def getSplineTime(splines, t):
    def time_function(u):
        return splines[0](u)[0] - t

    u_guess = 0.5
    u_result = root_scalar(time_function, bracket=[splines[0].t[0], splines[0].t[-1]])

    if u_result.converged:
        return u_result.root
    else:
        print("Root finding did not converge.")


# pos : (3, points)
# assumes pos is uniformly parameterized
def trajectoryConstraintsPolyline(pos, startPos=None, endPos=None, waypts=None, floor_height=None):
    pos_count = pos.shape[1]

    startIdx, endIdx = motionStartEnd(pos)

    if startPos is None:
        startPos = pos[:, 0]
    else:
        pos[:, :startIdx] = startPos.reshape(3, 1)  # force points to startPos before hand contact
    if endPos is None:
        endPos = pos[:, pos_count-1]
    else:
        pos[:, endIdx:] = endPos.reshape(3, 1)  # force points to endPos after hand contact

    if waypts == [] or waypts is None:
        pos[:, startIdx:endIdx] = transformSplines(pos[:, startIdx:endIdx], startPos, endPos)
    else:
        waypts_count = len(waypts)
        waypts_pos = np.zeros((waypts_count, 3)) # (n, 3)
        waypts_timestep = np.zeros((waypts_count))
        for i in range(waypts_count):
            waypts_pos[i, :] = waypts[i][0]
            waypts_timestep[i] = waypts[i][1]

        # get point closest to each timestep
        waypts_idx = np.zeros(waypts_count, dtype=int)

        times = np.linspace(0, 1, pos_count)
        
        for j in range(waypts_count):
            waypts_idx[j] = np.abs(times - waypts_timestep[j]).argmin()


        print("startIdx, waypts_idx, endIdx: ", startIdx, waypts_idx, endIdx)

        
        # sort by way point idx
        sort = waypts_idx.argsort()
        waypts_idx = waypts_idx[sort].astype(int)
        waypts_pos = waypts_pos[sort]

        for i in range(len(waypts_idx)):
            if waypts_idx[i] <= startIdx:
                waypts_idx[i] = startIdx+1
                print("WARNING: waypoint index squished to between start and end indices")
            elif waypts_idx[i] >= endIdx:
                waypts_idx[i] = endIdx-1
                print("WARNING: waypoint index squished to between start and end indices")

        # process segments
        # ex: three segments for two waypoints
        for i in range(waypts_count+1):
            if i == 0:
                # move polyline from 0 to waypts_idx to start to waypt
                pos[:, startIdx:waypts_idx[i]+1] = transformSplines(pos[:, startIdx:waypts_idx[i]+1], startPos, waypts_pos[i]) 
            elif i == waypts_count:
                # move polyline from waypts_idx to end to waypt to end
                pos[:, waypts_idx[i-1]:endIdx] = transformSplines(pos[:, waypts_idx[i-1]:endIdx], waypts_pos[i-1, :], endPos)
            else:
                pos[:, waypts_idx[i-1]:waypts_idx[i]+1] = transformSplines(pos[:, waypts_idx[i-1]:waypts_idx[i]+1], waypts_pos[i-1], waypts_pos[i])

        for i in range(pos_count):
            pos[2, i] = max(pos[2, i], floor_height)
    
    return pos, startIdx, endIdx

                

    


def trajectoryConstraints(splines, startPos=None, endPos=None, waypts=None, floor_height=None, bounding_sphere_radius=None, barriers=None):
    # convert spline to ctrlPts
    pos_count = splines[0].c.shape[0]
    x_pos = np.array([splines[0].c[i, 1] for i in range(pos_count)])
    y_pos = np.array([splines[1].c[i, 1] for i in range(pos_count)])
    z_pos = np.array([splines[2].c[i, 1] for i in range(pos_count)])
    times = np.array([splines[0].c[i, 0] for i in range(pos_count)]) # shape: (pos_count,)
    times /= np.max(times)
    pos = np.zeros((3, pos_count))
    pos[0, :] = x_pos
    pos[1, :] = y_pos
    pos[2, :] = z_pos

    startIdx, endIdx = motionStartEnd(pos)

    if startPos is None:
        startPos = pos[:, 0]
    else:
        pos[:, :startIdx] = startPos.reshape(3, 1)
    
    if endPos is None:
        endPos = pos[:, pos_count-1]
    else:
        pos[:, endIdx:] = endPos.reshape(3, 1)
    
    if waypts == [] or waypts is None:
        pos[:, startIdx:endIdx] = transformSplines(pos[:, startIdx:endIdx], startPos, endPos) 
        ctrlPtIdxs = None
    else:
        waypts_count = len(waypts)
        waypts_pos = np.zeros((waypts_count, 3)) # (n, 3)
        waypts_timestep = np.zeros((waypts_count))
        for i in range(waypts_count):
            waypts_pos[i, :] = waypts[i][0]
            waypts_timestep[i] = waypts[i][1]

        # get control point index closest to each waypoint
        # ctrlPtIdxs = np.zeros(len(waypts))
        # for j in range(len(waypts)):
        #     dist = np.zeros(pos_count)
        #     for i in range(pos_count):
        #         dist[i] = np.linalg.norm(pos[:, i] - waypts[j])
        #     ctrlPtIdxs[j] = np.argmin(dist)

        # get control point closest to each timestep
        # TODO: use interpolated spline value instead
        ctrlPtIdxs = np.zeros(waypts_count, dtype=int)
        for j in range(waypts_count):
            ctrlPtIdxs[j] = np.abs(times - waypts_timestep[j]).argmin()
            # print(ctrlPtIdxs[j])
            # print(waypts_timestep[j], times[ctrlPtIdxs[j]])

        print("startIdx, ctrlPtIds, endIdx: ", startIdx, ctrlPtIdxs, endIdx)

        # sort by control point idx
        sort = ctrlPtIdxs.argsort()
        ctrlPtIdxs = ctrlPtIdxs[sort].astype(int)
        waypts_pos = waypts_pos[sort]

        for i in range(len(ctrlPtIdxs)):
            ctrlPtIdxs[i] = max(ctrlPtIdxs[i], startIdx+1)
            ctrlPtIdxs[i] = min(ctrlPtIdxs[i]-1, endIdx-1)
            print("WARNING: ctrlPt index squished to between start and end indices")

        # ex: three segments for two waypoints
        for i in range(waypts_count+1):
            if i == 0:
                # move splines from 0 to ctrlPtIdx to start to waypt
                pos[:, startIdx:ctrlPtIdxs[i]+1] = transformSplines(pos[:, startIdx:ctrlPtIdxs[i]+1], startPos, waypts_pos[i]) 
            elif i == waypts_count:
                # move splines from ctrlPtIdx to end to waypt to end
                pos[:, ctrlPtIdxs[i-1]:endIdx] = transformSplines(pos[:, ctrlPtIdxs[i-1]:endIdx], waypts_pos[i-1, :], endPos)
            else:
                pos[:, ctrlPtIdxs[i-1]:ctrlPtIdxs[i]+1] = transformSplines(pos[:, ctrlPtIdxs[i-1]:ctrlPtIdxs[i]+1], waypts_pos[i-1], waypts_pos[i])


        # convert to initial Bspline
        for i in range(pos_count):
            splines[0].c[i, 1] = pos[0, i]
            splines[1].c[i, 1] = pos[1, i]
            splines[2].c[i, 1] = max(pos[2, i], floor_height)

        


        '''for i in range(len(waypts)):
            dist = 1
            radius = 1 # number of control points to "push"
            #tangents = [splines[0].derivative(1), splines[1].derivative(1), splines[2].derivative(1)]
            while (dist > 0.01):

                # push
                for j in range(ctrlPtIdxs[i]-radius//2, ctrlPtIdxs[i]+radius//2+1):
                    if j < 0 or j >= pos_count:
                        continue
                    t = getSplineTime(splines, splines[0].c[j, 0])

                    splines[0].c[j, 1] += (waypts[i, 0] - splines[0](t)[1]) * 0.01
                    splines[1].c[j, 1] += (waypts[i, 1] - splines[1](t)[1]) * 0.01
                    splines[2].c[j, 1] += (waypts[i, 2] - splines[2](t)[1]) * 0.01

                current = np.zeros(3)
                current[0] = splines[0](t)[1]
                current[1] = splines[1](t)[1]
                current[2] = splines[2](t)[1]

                dist = np.linalg.norm(waypts[i, :] - current)'''


    pos_origin = pos.copy()

    def objective(control_points_flat):
        control_points = control_points_flat.reshape((3, pos_count))

        # weigh points besides waypoints
        #control_points[:, ctrlPtIdxs[0]] = pos_origin[:, ctrlPtIdxs[0]]

        change = np.sum((control_points - pos_origin) ** 2)

        #if bounding_sphere_radius is not None and barriers is not None:
        #    change += energy(torch.tensor(control_points), bounding_sphere_radius, barriers).detach().numpy()
        return change
        

        #return np.linalg.norm(constraint_func(control_points_flat))
    
    def constraint_func(control_points_flat):
        if waypts is not None:
            control_points = control_points_flat.reshape((3, pos_count))
            for i in range(pos_count):
                splines[0].c[i, 1] = control_points[0, i]
                splines[1].c[i, 1] = control_points[1, i]
                splines[2].c[i, 1] = control_points[2, i]
            
            dist = np.zeros(3)
            for i in range(waypts_count):
                t = getSplineTime(splines, splines[0].c[ctrlPtIdxs[i], 0])
                dist[0] += abs(splines[0](t)[1] - waypts_pos[i,0])
                dist[1] += abs(splines[1](t)[1] - waypts_pos[i,1])
                dist[2] += abs(splines[2](t)[1] - waypts_pos[i,2])

                
            #barrier = energy(torch.tensor(control_points), bounding_sphere_radius, barriers).detach().numpy()
            
            return dist
        return [0, 0, 0]
    
    constraints = None
    if waypts is not None:
        constraints = {'type': 'eq', 'fun': constraint_func}
        
    result = minimize(objective, pos.flatten(), constraints=constraints, method = 'SLSQP', options={'maxiter': 500, 'disp': True})
    if result.success:
        pos = result.x.reshape((3, pos_count))
    else:
        print("optimization failed!")


    '''# avoid barriers
    if bounding_sphere_radius is not None and barriers is not None:
        pos = avoid_barriers(pos, bounding_sphere_radius, barriers)'''


    if floor_height is None:
        floor_height = -math.inf


    # convert to Bspline
    for i in range(pos_count):
        splines[0].c[i, 1] = pos[0, i]
        splines[1].c[i, 1] = pos[1, i]
        splines[2].c[i, 1] = max(pos[2, i], floor_height)

    startTime = getSplineTime(splines, splines[0].c[startIdx, 0])
    endTime = getSplineTime(splines, splines[0].c[endIdx, 0])

    return splines, startTime, endTime, ctrlPtIdxs



def create_obj(splines, path, resolution):
    sim_time = np.linspace(0, 1, resolution)
    position = np.array([spline(sim_time) for spline in splines[:3]])
    if position.ndim == 3:
        position = position[:, :, 1] # (3, frames)

    out_str = ''

    for i in range(position.shape[1]):
        #out_str += 'v ' + str(-position[0, i]*20-1) + ' ' + str(position[2, i]*20-22) + ' ' + str(position[1, i]*20+10) + '\n'
        out_str += 'v ' + str(-position[0, i]) + ' ' + str(position[2, i]) + ' ' + str(position[1, i]) + '\n'
    
    for i in range(position.shape[1]-1):
        out_str += 'l ' + str(i+1) + ' ' + str(i+2) + '\n'

    with open(path, "w") as text_file:
        text_file.write(out_str)





# intersection energy
def energy(pos, bounding_sphere_radius, barriers):
    if bounding_sphere_radius is None:
        bounding_sphere_radius = torch.tensor([0.0])
    else:
        bounding_sphere_radius = torch.tensor([bounding_sphere_radius])

    sum = torch.tensor([0.0])

    if barriers is not None:
        for j in range(len(barriers)):
            for i in range(pos.shape[1]):
                sum += 1.0 - torch.min(torch.norm(pos[:, i] - torch.from_numpy(barriers[j][0])) / (bounding_sphere_radius + torch.tensor(barriers[j][1])), torch.tensor([1.0]))
                #print(sum, torch.norm(pos[:, i] - torch.from_numpy(barriers[j][0])) / (bounding_sphere_radius + torch.tensor(barriers[j][1])))
    
    return sum

def avoid_barriers(pos, bounding_sphere_radius, barriers):
    coordinates_t = torch.tensor(pos, requires_grad=True)
    #center_t = torch.from_numpy(avoid[0])
    #radius_t = torch.tensor([avoid[1]])

    optimizer = torch.optim.Adam([coordinates_t], lr=0.2)
    optimizer.zero_grad()

    loss = energy(coordinates_t, bounding_sphere_radius, barriers)
    print(loss)

    loss.backward()
    optimizer.step()

    #loss = energy(coordinates_t, center_t, radius_t.float(), bounding_sphere_radius)
    loss = energy(coordinates_t, bounding_sphere_radius, barriers)
    print(loss)

    return coordinates_t.detach().numpy()