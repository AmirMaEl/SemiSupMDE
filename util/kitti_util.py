import numpy as np
import os
from collections import Counter
from scipy.interpolate import LinearNDInterpolator

class KITTI:
    """
    Utility class for handling KITTI dataset calibration and depth information.
    """
    
    def read_calib_file(self, path):
        """
        Read KITTI calibration file.
        
        Args:
            path: Path to the calibration file.
        
        Returns:
            Dictionary containing calibration data.
        """
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # Try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        pass

        return data

    def get_fb(self, calib_dir, cam=2):
        """
        Get the focal length and baseline for a given camera.
        
        Args:
            calib_dir: Directory containing calibration files.
            cam: Camera index (default: 2).
        
        Returns:
            Tuple containing focal length * baseline, focal length, and baseline.
        """
        cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
        P3_rect = cam2cam['P_rect_03'].reshape(3, 4)

        b2 = P2_rect[0, 3] / -P2_rect[0, 0]
        b3 = P3_rect[0, 3] / -P3_rect[0, 0]
        baseline = b3 - b2

        if cam == 2:
            focal_length = P2_rect[0, 0]
        elif cam == 3:
            focal_length = P3_rect[0, 0]

        return focal_length * baseline, focal_length, baseline

    def load_velodyne_points(self, file_name):
        """
        Load Velodyne points from a file.
        
        Args:
            file_name: Path to the Velodyne points file.
        
        Returns:
            Numpy array of Velodyne points.
        """
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # Homogeneous coordinates
        return points

    def lin_interp(self, shape, xyd):
        """
        Perform linear interpolation to fill in depth map holes.
        
        Args:
            shape: Shape of the depth map.
            xyd: XYD coordinates for interpolation.
        
        Returns:
            Interpolated depth map.
        """
        m, n = shape
        ij, d = xyd[:, 1::-1], xyd[:, 2]
        f = LinearNDInterpolator(ij, d, fill_value=0)
        J, I = np.meshgrid(np.arange(n), np.arange(m))
        IJ = np.vstack([I.flatten(), J.flatten()]).T
        disparity = f(IJ).reshape(shape)
        return disparity

    def sub2ind(self, matrixSize, rowSub, colSub):
        """
        Convert subscripts to linear indices.
        
        Args:
            matrixSize: Size of the matrix.
            rowSub: Row subscripts.
            colSub: Column subscripts.
        
        Returns:
            Linear indices.
        """
        m, n = matrixSize
        return rowSub * (n - 1) + colSub - 1

    def get_depth(self, calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
        """
        Get the depth map from Velodyne points.
        
        Args:
            calib_dir: Directory containing calibration files.
            velo_file_name: Path to the Velodyne points file.
            im_shape: Shape of the image.
            cam: Camera index (default: 2).
            interp: Boolean indicating whether to interpolate the depth map.
            vel_depth: Boolean indicating whether to use Velodyne depth.
        
        Returns:
            Depth map (and interpolated depth map if interp=True).
        """
        # Load calibration files
        cam2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # Compute projection matrix Velodyne -> image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        # Load Velodyne points and remove all points behind image plane
        velo = self.load_velodyne_points(velo_file_name)
        velo = velo[velo[:, 0] >= 0, :]

        # Project the points to the camera
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

        if vel_depth:
            velo_pts_im[:, 2] = velo[:, 0]

        # Check if in bounds
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # Project to image
        depth = np.zeros(im_shape)
        depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

        # Find the duplicate points and choose the closest depth
        inds = self.sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        if interp:
            # Interpolate the depth map to fill in holes
            depth_interp = self.lin_interp(im_shape, velo_pts_im)
            return depth, depth_interp
        else:
            return depth
