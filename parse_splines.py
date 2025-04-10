from scipy.interpolate import BSpline
import json
import numpy as np
import math
import torch

def parseSplines(path, avoid = None, bounding_sphere_radius = None, object_shift = None):
    with open(path, 'r') as f:
        spline_file = json.load(f)

    numDofs = spline_file['numDofs']
    degree = spline_file['degree']
    #dimension = spline_file['dimension']
    time = spline_file['time']

    data = spline_file['data']

    # change dimensions
    for d in data:
        if d['units'] == 'centimeters':
            d['units'] = 'meters'
            for i in range(1, len(d['controlPointData']), 2):
                d['controlPointData'][i] /= 100

        if d['units'] == 'degrees':
            d['units'] = 'radians'
            for i in range(1, len(d['controlPointData']), 2):
                d['controlPointData'][i] *= math.pi / 180

    # change object's starting location
    '''shiftedControlPointsCount = 15
    if object_shift is not None:
        data[0]['controlPointData'][1:shiftedControlPointsCount*2+1:2] += object_shift[0]
        data[1]['controlPointData'][1:shiftedControlPointsCount*2+1:2] += object_shift[1]
        data[2]['controlPointData'][1:shiftedControlPointsCount*2+1:2] += object_shift[2]'''

    splines = [None] * numDofs

    for i in range(numDofs):
        # shape: [60 x 2]
        controlPoints = np.zeros((data[i]['numControlPoints'], 2))

        controlPoints[:, 0] = data[i]['controlPointData'][::2]  # frame
        controlPoints[:, 1] = data[i]['controlPointData'][1::2] # control points

        nKnots = data[0]['numControlPoints'] + degree + 1

        # create uniform clamped knots in [0, 1]
        clampedKnots = np.concatenate((
            np.zeros(degree),
            np.linspace(0, 1, nKnots - 2 * degree),
            np.ones(degree)
        ))

        #print(clampedKnots)

        splines[i] = BSpline(clampedKnots, controlPoints, degree)

    # change translation
    if avoid is not None or object_shift is not None:
        coordinates = np.zeros(((data[0]['numControlPoints'] - 2) * 3))
        coordinates[::3] = data[0]['controlPointData'][3:-1:2]
        coordinates[1::3] = data[1]['controlPointData'][3:-1:2]
        coordinates[2::3] = data[2]['controlPointData'][3:-1:2]

        if object_shift is not None:
            coordinates = shift(coordinates, object_shift)
            # translate starting point
            data[0]['controlPointData'][1] += object_shift[0]
            data[1]['controlPointData'][1] += object_shift[1]
            data[2]['controlPointData'][1] += object_shift[2]
        if avoid is not None:
            coordinates = minimize(coordinates, avoid, bounding_sphere_radius=bounding_sphere_radius)
            

        # make all points above ground
        # TODO: do this in a smarter way
        for i in range(0, coordinates.shape[0], 3):
            # below ground
            if coordinates[i+2] < coordinates[2]:
                coordinates[i+2] = coordinates[2]

        for i in range(3):
            # shape: [60 x 2]
            controlPoints = np.zeros((data[i]['numControlPoints'], 2))

            controlPoints[:, 0] = data[i]['controlPointData'][::2]  # frame

            data[i]['controlPointData'][3:-1:2] = coordinates[i::3] # replace optimized spline points excluding endpoints
            controlPoints[:, 1] = data[i]['controlPointData'][1::2] # control points

            nKnots = data[0]['numControlPoints'] + degree + 1

            # create uniform clamped knots in [0, 1]
            clampedKnots = np.concatenate((
                np.zeros(degree),
                np.linspace(0, 1, nKnots - 2 * degree),
                np.ones(degree)
            ))

            splines[i] = BSpline(clampedKnots, controlPoints, degree)

    
    return splines, time, data


def shift(coordinates, object_shift):
    # determine when hand moves object
    start_frame = 0
    for i in range(0, coordinates.shape[0], 3):
        if np.linalg.norm(coordinates[:3] - coordinates[i:i+3]) < 0.02:
            start_frame += 1
        else:
            break

    # determine when hand stops moving object
    end_frame = start_frame + 1
    for i in range(end_frame*3, coordinates.shape[0], 3):
        if np.linalg.norm(coordinates[:3] - coordinates[i:i+3]) > 0.02:
            end_frame += 1
        else:
            break

    #print(start_frame, end_frame)

    # shift frames before start_frame by object_shift
    for i in range(0, start_frame*3, 3):
        coordinates[i:i+3] += object_shift

    # shift frames between start and end frames by object_shift * weight
    for i in range(start_frame*3, end_frame*3, 3):
        #coordinates[i:i+3] = coordinates[:3] + object_shift
        weight = (1 - (i - start_frame*3) / (end_frame*3 - start_frame*3))**2
        #print(weight, 1-weight)
        #coordinates[i:i+3] = coordinates[i:i+3] * (1-weight) + (coordinates[:3] + object_shift) * weight
        coordinates[i:i+3] += weight * object_shift

    return coordinates

def energy(points, center, radius, bounding_sphere_radius):
    if bounding_sphere_radius is None:
        bounding_sphere_radius = torch.tensor([0.0])
    else:
        bounding_sphere_radius = torch.tensor([bounding_sphere_radius])

    sum = torch.tensor([0.0])
    for i in range(0, points.shape[0], 3):
        # below ground
        #sum += 0.5 * torch.max(points[2] - points[i+2], torch.tensor([0.0]))
        
        sum += 1.0 - torch.min(torch.norm(points[i:i+3] - center) / (bounding_sphere_radius + radius), torch.tensor([1.0]))
        

    return sum

'''
def shift_error(coordinates, object_shift):
    sum = torch.tensor([0.0])
    for i in range(0, points.shape[0], 3):
        sum += object_shift - points[i:i+3]
        

    return sum'''

def minimize(coordinates, avoid = None, bounding_sphere_radius = None):
    coordinates_t = torch.tensor(coordinates, requires_grad=True)
    center_t = torch.from_numpy(avoid[0])
    radius_t = torch.tensor([avoid[1]])

    optimizer = torch.optim.Adam([coordinates_t], lr=0.2)
    optimizer.zero_grad()

    loss = energy(coordinates_t, center_t, radius_t.float(), bounding_sphere_radius)

    '''if object_shift is None:
        loss = energy(coordinates_t, center_t, radius_t.float(), bounding_sphere_radius)
    else:
        loss = shift_error(coordinates_t, torch.from_numpy(object_shift))'''

    loss.backward()
    optimizer.step()

    loss = energy(coordinates_t, center_t, radius_t.float(), bounding_sphere_radius)
    print(loss)

    return coordinates_t.detach().numpy()


if __name__ == '__main__':

    path = 'startingTrajectories/MANO_right/apple_pass/hand.smexp'

    splines, _, data = parseSplines(path)

    import matplotlib.pyplot as plt

    u = np.linspace(0, 1, 100)

    points = splines[7](u)

    plt.figure(figsize=(8, 6))

    # Control points
    plt.plot(data[7]['controlPointData'][::2], data[7]['controlPointData'][1::2], 'o--', label='Control Points', color='gray')

    # Clamped B-spline
    plt.plot(points[:, 0], points[:, 1], label='B-spline', color='blue')

    plt.grid()
    plt.show()