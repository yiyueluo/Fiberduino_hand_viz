from math import pi

import pyvista as pv
import torch
from trimesh import Trimesh
import math
import time

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput


def main():
    mano_layer = ManoLayer(rot_mode="axisang",
                           center_idx=9,
                           mano_assets_root="assets/mano",
                           use_pca=False,
                           flat_hand_mean=True)
    hand_faces = mano_layer.th_faces  # (NF, 3)

    axisFK = AxisLayerFK(mano_assets_root="assets/mano")
    composed_ee = torch.zeros((1, 16, 3))

    #  transform order of right hand
    #         15-14-13-\
    #                   \
    #*   3-- 2 -- 1 -----0   < NOTE: demo on this finger
    #   6 -- 5 -- 4 ----/
    #   12 - 11 - 10 --/
    #    9-- 8 -- 7 --/


    ''' previous sample joint angle'''
    # ax2 = [1.39, 1.33, 1.78, 1.42, 1.68, 1.23]
    # ay2 = [8.1, 8.1, 7.94, 7.99, 7.83, 8.23]
    # az2 = [-19.61, -19.61, -19.61, -19.61, -19.61, -19.61]
    # float_pitch2 = []
    #
    # ax1 = [19.01, 18.83, 18.69, 18.68, 18.75, 18.83]
    # ay1 = [19.23, 19.01, 19.26, 19.21, 19.15, 19]
    # az1 = [13.27, 13.42, 12.91, 13.44, 13.30, 13.44]
    # float_pitch1 = []
    #
    # for i in range(len(ax2)):
    #     float_pitch2.append(math.atan2(-ax2[i], math.sqrt(ay2[i] * ay2[i] + az2[i] * az2[i])))
    #     float_pitch1.append(math.atan2(-ax1[i], math.sqrt(ay1[i] * ay1[i] + az1[i] * az1[i])))


    ''' initiate viz '''
    composed_aa = axisFK.compose(composed_ee).clone()  # (B=1, 16, 3)
    composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)
    zero_shape = torch.zeros((1, 10))

    mano_output: MANOOutput = mano_layer(composed_aa, zero_shape)

    T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
    T_g_a, R, ee = axisFK(T_g_p)
    T_g_a = T_g_a.squeeze(0)
    hand_verts = mano_output.verts.squeeze(0)  # (NV, 3)
    hand_faces = mano_layer.th_faces  # (NF, 3)
    mesh = pv.wrap(Trimesh(hand_verts, hand_faces))

    pl = pv.Plotter(off_screen=False, polygon_smoothing=True)
    # pl.camera.view_angle = 15
    pl.set_viewup([-1,0.3, 1])
    pl.add_mesh(mesh, opacity=0.8, name="mesh", smooth_shading=True)
    pl.set_background('white')
    pl.add_camera_orientation_widget()
    pl.show(interactive_update=True)


    '''insert code to extract data from arduino script HERE:'''

    while True:
        ''' update joint angle here '''
        composed_ee[:, 1] = torch.tensor([0, 0, pi / 6]).unsqueeze(0)
        composed_ee[:, 2] = torch.tensor([0, 0, pi / 6]).unsqueeze(0)
        composed_ee[:, 3] = torch.tensor([0, 0, pi / 6]).unsqueeze(0)

        composed_aa = axisFK.compose(composed_ee).clone()  # (B=1, 16, 3)
        composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)
        zero_shape = torch.zeros((1, 10))

        mano_output: MANOOutput = mano_layer(composed_aa, zero_shape)

        T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
        T_g_a, R, ee = axisFK(T_g_p)
        T_g_a = T_g_a.squeeze(0)
        hand_verts = mano_output.verts.squeeze(0)  # (NV, 3)
        hand_faces = mano_layer.th_faces  # (NF, 3)
        mesh = pv.wrap(Trimesh(hand_verts, hand_faces))

        # pl = pv.Plotter(off_screen=False, polygon_smoothing=True)
        # pl.camera.view_angle = 15
        pl.set_viewup([-1, 0.3, 1])
        pl.add_mesh(mesh, opacity=0.8, name="mesh", smooth_shading=True)
        pl.set_background('white')
        pl.add_camera_orientation_widget()
        # pl.show(interactive_update=True)
        time.sleep(0.1)
        pl.update()

        # pl.show(interactive=True)
        # pl.show(auto_close=False)

        # ===== NOTE: common the above pl.show(), and uncommnet the following code to generate a gif >>>>>
        # view_up = [-1, 0, 0]
        # path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=view_up, shift=0.1)
        # pl.open_gif("orbit.gif")
        # pl.orbit_on_path(path, write_frames=True, step=0.05, viewup=view_up)
        # pl.close()


if __name__ == "__main__":
    main()