#!/usr/bin/env python
import utils
import numpy
###YOUR IMPORTS HERE###
import random
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')

    ###YOUR CODE HERE###
    print 'please wait for about 10 seconds'
    # Fit a plane to the data using ransac
    error_best = 1000
    for i in range(200):
        pcp = pc[:]

        # choose 3 samples
        for j in range(3):
            a = random.choice(range(len(pcp)))
            if j == 0:
                p1 = pcp[a]
                pcp.pop(a)
            else:
                p1 = numpy.concatenate((p1, pcp[a]), axis=1)
                pcp.pop(a)

        # compute model plane and normal vector
        if numpy.linalg.matrix_rank(p1.T) < 3:
            u, s, w = numpy.linalg.svd(p1.T)
            normal = w.T[:, 3]
        else:
            normal = numpy.linalg.inv(p1.T)*numpy.matrix([[1.], [1.], [1.]])

        # build consensus set and outliers set
        c = []
        o = []
        for k in range(len(pcp)):
            if numpy.linalg.matrix_rank(p1.T) < 3:
                error = - normal.T*pcp[k]/numpy.linalg.norm(normal)
            else:
                error = (1.0 - normal.T*pcp[k])/numpy.linalg.norm(normal)
            if abs(error.tolist()[0][0]) < 0.05:
                c.append(pcp[k])
            else:
                o.append(pcp[k])

        # re-fit model if applicable
        if len(c) > 100:
            for l in range(3):
                c.append(p1[:, l])

            # use pca to find the plane that fits inliers
            for m in range(len(c)):
                if m == 0:
                    sum = c[0]
                else:
                    sum = sum + c[m]

            mu = sum / (m + 1)

            for n in range(len(c)):
                if n == 0:
                    cx = c[n] - mu
                else:
                    cx = numpy.concatenate((cx, c[n] - mu), axis=1)

            u, s, w = numpy.linalg.svd(cx * cx.T / n)
            cx_new = w * cx

            error_new = 0

            for p in range(len(cx_new.tolist()[2])):
                error_new = error_new + cx_new[2, p]**2

            if error_new < error_best:
                error_best = error_new
                vector = numpy.matrix([[0., 0., 1.]])
                planev = numpy.linalg.inv(w) * vector.T
                origin = numpy.matrix([[0., 0., 0.]])
                planep = numpy.linalg.inv(w) * origin.T + mu
                inliers = c[:]
                outliers = o[:]

    print '\nplane function is:\n', planev.tolist()[0][0], '(x-', planep.tolist()[0][0], ')+', planev.tolist()[1][0], '(y-', planep.tolist()[1][0], ')+', planev.tolist()[2][0], '(z-', planep.tolist()[2][0], ')=0'
    fig = utils.view_pc([outliers])
    fig = utils.view_pc([inliers], fig, ['r'], ['^'])
    a = [planep]
    fig = utils.view_pc([a], fig, ['r'], ['^'])
    fig = utils.draw_plane(fig, planev, planep, (0.1, 0.7, 0.1, 0.5), [-0.5, 1.0], [-0.4, 1.2])

    # Show the resulting point cloud

    # Draw the fitted plane


    ###YOUR CODE HERE###
    raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
