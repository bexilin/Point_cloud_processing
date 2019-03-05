#!/usr/bin/env python
import utils
import numpy
###YOUR IMPORTS HERE###
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    utils.view_pc([pc])

    #Rotate the points to align with the XY plane
    for i in range(len(pc)):
        if i == 0:
            sum = pc[0]
        else:
            sum = sum + pc[i]

    mu = sum/(i+1)

    for j in range(len(pc)):
        if j == 0:
            pcx = pc[j] - mu
        else:
            pcx = numpy.concatenate((pcx, pc[j] - mu), axis=1)

    u, s, w = numpy.linalg.svd(pcx*pcx.T/j)
    pcx_new = w*pcx
    print 'W:\n', w

    # Show the resulting point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc1 = utils.convert_matrix_to_pc(pcx_new)
    utils.view_pc([pc1],fig)
    ax.set_xlim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    plt.title('pca')

    # Rotate the points to align with the XY plane AND eliminate the noise
    wv = w.copy()
    wk = w.copy()
    for i in range(len(s)):
        if s[i] <= 0.01:
            wv = numpy.delete(wv, i, 0)
            wk[i] = [0., 0., 0.]

    pcx_new_2 = wk*pcx
    print '\nWv:\n', wv

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    pc2 = utils.convert_matrix_to_pc(pcx_new_2)
    utils.view_pc([pc2],fig2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    plt.title('pca and noise filtering')

    # plot plane
    fig = utils.view_pc([pc])
    vector = numpy.matrix([[0., 0., 1.]])
    planev = numpy.linalg.inv(w)*vector.T
    planev = planev / numpy.linalg.norm(planev)
    origin = numpy.matrix([[0., 0., 0.]])
    planep = numpy.linalg.inv(w)*origin.T + mu
    fig = utils.draw_plane(fig, planev, planep, (0.1, 0.7, 0.1, 0.5), [-0.4, 0.9], [-0.4, 1])
    plt.title('fitting plane with pca')
    print '\nplane function is: \n', planev.tolist()[0][0], '(x-', planep.tolist()[0][0], ')+', planev.tolist()[1][0], '(y-', planep.tolist()[1][0], ')+', planev.tolist()[2][0], '(z-', planep.tolist()[2][0], ')=0'

    ###YOUR CODE HERE###


    raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
