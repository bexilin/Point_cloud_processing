#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import random
import random_walk
import time
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    #pc_target = utils.load_pc('cloud_icp_target3.csv') # Change this to load in a different target


    for tg in range(4):
        if tg == 0:
            pc_target = utils.load_pc('cloud_icp_target0.csv')
            utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
            print 'test target 0:\n\n'
        elif tg == 1:
            pc_source = utils.load_pc('cloud_icp_source.csv')
            pc_target = utils.load_pc('cloud_icp_target1.csv')
            utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
            print 'test target 1:\n\n'
        elif tg == 2:
            pc_source = utils.load_pc('cloud_icp_source.csv')
            pc_target = utils.load_pc('cloud_icp_target2.csv')
            utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
            print 'test target 2:\n\n'
        elif tg == 3:
            pc_source = utils.load_pc('cloud_icp_source.csv')
            pc_target = utils.load_pc('cloud_icp_target3.csv')
            utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
            print 'test target 3:\n\n'

        p = utils.convert_pc_to_matrix(pc_source)
        q = utils.convert_pc_to_matrix(pc_target)
        T_list = []
        iteration = []
        error_all = []
        success = 0
        print 'stop criterion: distance error converges to the threshold or not able to converge within 2000 iterations. So please wait for at most 2000 iterations, which takes only a few minutes'
        raw_input('\npress enter to start\n')

        for num in range(2000):
            print 'iteration',num+1,':\n'
            iteration.append(num+1)
            pf = numpy.matrix([[],[],[]])
            qf = numpy.matrix([[],[],[]])

            while p.shape[1] > 0:
                i = random.choice(range(p.shape[1]))
                j = numpy.argmin(numpy.linalg.norm(q-p[:, i], axis=0))
                pf = numpy.hstack((pf, p[:, i]))
                p = numpy.delete(p,i,1)
                qf = numpy.hstack((qf, q[:, j]))
                q = numpy.delete(q,j,1)

            p = pf.copy()
            q = qf.copy()

            p_avg = p.sum(axis=1)/(p.shape[1]*1.0)
            q_avg = q.sum(axis=1)/(q.shape[1]*1.0)
            X=numpy.subtract(p,p_avg)
            Y=numpy.subtract(q,q_avg)
            u,s,w=numpy.linalg.svd(X*Y.T)
            m = numpy.matrix([[1.,0.,0.],[0.,1.,0.],[0.,0.,numpy.linalg.det(w.T*u.T)]])
            R = w.T*m*u.T
            t = q_avg-R*p_avg

            T = numpy.concatenate((R,t),axis=1)
            T = numpy.concatenate((T,numpy.matrix([[0.,0.,0.,1.]])))
            T_list.append(T)

            fit_error = numpy.add(R*p,t)-q
            error_all.append(numpy.linalg.norm(fit_error)**2)
            print 'distance least square error:',numpy.linalg.norm(fit_error)**2,'\n\n'
            p = R*p + t

            if tg == 3 and random.randint(1,20) == 1 and numpy.linalg.norm(fit_error)**2 > 0.1:
                R_random = random_walk.random_walk()
                p = R_random * (p-p_avg) + p_avg
                R = R_random
                t = p_avg-R_random*p_avg
                T = numpy.concatenate((R, t), axis=1)
                T = numpy.concatenate((T, numpy.matrix([[0., 0., 0., 1.]])))
                T_list.append(T)

            if numpy.linalg.norm(fit_error) < 0.1:
                for i in range(len(T_list)):
                    if i==0:
                        T_final = T_list[i]
                    else:
                        T_final = T_list[i]*T_final
                    success = 1
                break

        pc = utils.convert_pc_to_matrix(pc_source)
        if success == 0:
            for i in range(len(T_list)):
                if i == 0:
                    T_final = T_list[i]
                else:
                    T_final = T_list[i] * T_final

        print 'transformation from source to target point cloud:\n'
        print 'R =\n', T_final[:3,:3], '\n\nt =\n', T_final[:3,3]
        pc = T_final[:3,:3]*pc+T_final[:3,3]
        pc_source = utils.convert_matrix_to_pc(pc)
        utils.view_pc([pc_source], None, ['b'], ['o'])
        plt.axis([-0.15, 0.15, -0.15, 0.15])
        plt.figure()
        plt.title('ICP Error vs Iteration')
        plt.plot(iteration,error_all,'ro-')
        plt.xlabel('Iteration')
        plt.ylabel('Least squares error')
        raw_input('press enter and test the next target\n')
        plt.close()
        plt.close()
        plt.close()
    ###YOUR CODE HERE###

    raw_input("\nPress enter to end:")


if __name__ == '__main__':
    main()


