import math
import numpy as np
import numbers


def delta_function_roma(r, r0, dr):
    # Roma, Peskin, & Berger (JCP 1999)
    if abs(r - r0) > (1.5 * dr):
        return 0
    elif abs(r - r0) <= (0.5 * dr):
        return (1 + math.sqrt(-3 * (((r - r0) / dr) ** 2) + 1)) / (3 * dr)
    else:  # abs(r - r0) <= (1.5 * dr)
        return (5 - 3 * abs((r - r0) / dr) - math.sqrt(-3 * ((1 - abs((r - r0) / dr)) ** 2) + 1)) / (6 * dr)


def interpolate(field, grid_x, grid_y, grid_delta_x, grid_delta_y, interpolation_points, xThreshold=math.inf, yThreshold=math.inf):
    if isinstance(interpolation_points, numbers.Number):
        interpolation_value = np.array([0])
        interpolation_points = np.array([interpolation_points])
    else:
        interpolation_value = np.zeros(interpolation_points.size)

    for i in range(0, grid_x.size):
        for j in range(0, grid_y.size):
            for k in range(interpolation_points.size):
                if (abs(grid_x[i] - interpolation_points.x[k]) < xThreshold) and \
                   (abs(grid_y[j] - interpolation_points.y[k]) < yThreshold):
                    interpolation_value[k] += field[j][i] * grid_delta_x[i] * grid_delta_y[j] * (
                            delta_function_roma(grid_x[i], interpolation_points.x[k], grid_delta_x[i]) *
                            delta_function_roma(grid_y[j], interpolation_points.y[k], grid_delta_y[j]))
    return interpolation_value

# def regularize(fieldToRegularize, BodyXCoords, BodyYCoords, DeltaX, DeltaY, xToRegularize, yToRegularize,
#                xThreshold, yThreshold):
#     temp = 0
#     for i in range(1, fieldToRegularize.size):
#         if (abs(BodyXCoords[i] - xToRegularize) < xThreshold) & (abs(BodyYCoords[i] - yToRegularize) < yThreshold):
#             temp = temp + fieldToRegularize * DeltaX * DeltaY * (
#                     delta_function_roma(xCoordsOfField[i], xToInterpolate, DeltaX) *
#                     delta_function_roma(yCoordsOfField[j], yToInterpolate, DeltaY))
#     return temp
