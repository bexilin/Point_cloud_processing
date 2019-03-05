#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib
###YOUR IMPORTS HERE###

###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    print 'please wait for about 20 seconds'
    pc = utils.load_pc('cloud_pca.csv')
    error_pca = [0]*10
    error_ransac = [1000]*10
    num_tests = 10
    t = [0]*10
    fig = None
    for num in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        #fig = utils.view_pc([pc])

        ###YOUR CODE HERE###
        print '\niteration', num+1, 'begins:\n'

        t[num]=(num+1)*10

        startp = time.clock()
        # use pca
        for i in range(len(pc)):
            if i == 0:
                sum = pc[0]
            else:
                sum = sum + pc[i]

        mu = sum / (len(pc))

        for j in range(len(pc)):
            if j == 0:
                pcx = pc[j] - mu
            else:
                pcx = numpy.concatenate((pcx, pc[j] - mu), axis=1)

        u, s, w = numpy.linalg.svd(pcx*pcx.T / (len(pc)-1))
        pcx_new = w * pcx

        c_p = []
        o_p = []
        for k in range(len(pcx_new.tolist()[2])):
            if abs(pcx_new.tolist()[2][k]) < 0.05:
                c_p.append(pc[k])
                error_pca[num] = error_pca[num] + abs(pcx_new.tolist()[2][k])**2
            else:
                o_p.append(pc[k])

        endp = time.clock()
        print 'pca runtime: ', endp-startp
        # print 'pca inliers: ', len(c_p)

        if num == num_tests-1:
            fig2 = utils.view_pc([o_p])
            fig2 = utils.view_pc([c_p], fig2, ['r'], ['^'])
            vector = numpy.matrix([[0., 0., 1.]])
            planev = numpy.linalg.inv(w) * vector.T
            planev = planev/numpy.linalg.norm(planev)
            origin = numpy.matrix([[0., 0., 0.]])
            planep = numpy.linalg.inv(w) * origin.T + mu
            fig2 = utils.draw_plane(fig2, planev, planep, (0.1, 0.7, 0.1, 0.5), [-0.4, 0.9], [-0.4, 1])
            matplotlib.pyplot.title('pca')
            print '\npca fitting plane function is:\n', planev.tolist()[0][0], '(x-', planep.tolist()[0][0], ')+', planev.tolist()[1][0], '(y-', planep.tolist()[1][0], ')+', planev.tolist()[2][0], '(z-', planep.tolist()[2][0], ')=0'

        # use ransac
        startr = time.clock()
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
                normal = numpy.linalg.inv(p1.T) * numpy.matrix([[1.], [1.], [1.]])

            # build consensus set and outliers set
            c = []
            o = []
            for k in range(len(pcp)):
                if numpy.linalg.matrix_rank(p1.T) < 3:
                    error = - normal.T * pcp[k] / numpy.linalg.norm(normal)
                else:
                    error = (1.0 - normal.T * pcp[k]) / numpy.linalg.norm(normal)
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
                    error_new = error_new + abs(cx_new[2, p])**2

                if error_new < error_ransac[num]:
                    error_ransac[num] = error_new
                    vector = numpy.matrix([[0., 0., 1.]])
                    planev = numpy.linalg.inv(w) * vector.T
                    planev = planev / numpy.linalg.norm(planev)
                    origin = numpy.matrix([[0., 0., 0.]])
                    planep = numpy.linalg.inv(w) * origin.T + mu
                    inliers = c[:]
                    outliers = o[:]

        endr = time.clock()
        print'\nransac runtime: ',endr-startr

        if num == num_tests-1:
            fig3 = utils.view_pc([outliers])
            fig3 = utils.view_pc([inliers], fig3, ['r'], ['^'])
            a = [planep]
            fig3 = utils.view_pc([a], fig3, ['r'], ['^'])
            fig3 = utils.draw_plane(fig3, planev, planep, (0.1, 0.7, 0.1, 0.5), [-0.4, 0.9], [-0.4, 1])
            matplotlib.pyplot.title('ransac')
            print '\nransac fitting plane function is:\n', planev.tolist()[0][0], '(x-', planep.tolist()[0][0], ')+', planev.tolist()[1][0], '(y-', planep.tolist()[1][0], ')+', planev.tolist()[2][0], '(z-', planep.tolist()[2][0], ')=0'

        #print '\nransac inliers: ', len(inliers)

        #this code is just for viewing, you can remove or change it
        #raw_input("Press enter for next test:")
        #matplotlib.pyplot.close(fig)
        #matplotlib.pyplot.close(fig2)
        #matplotlib.pyplot.close(fig3)

    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('pca')
    matplotlib.pyplot.plot(t,error_pca,'ro-')
    matplotlib.pyplot.xlabel('Number of Outliers')
    matplotlib.pyplot.ylabel('least squares error')
    matplotlib.pyplot.ylim(0, 0.2)
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title('ransac')
    matplotlib.pyplot.plot(t,error_ransac,'ro-')
    matplotlib.pyplot.xlabel('Number of Outliers')
    matplotlib.pyplot.ylabel('least squares error')
    matplotlib.pyplot.ylim(0, 0.2)
        ###YOUR CODE HERE###

    raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
