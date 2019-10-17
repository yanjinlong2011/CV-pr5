"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.]).T  # state
        self.Q = Q
        self.R = R
        self.sig = 2000 * np.eye(4)

        self.M = np.matrix(np.eye(4))[:2]

        self.D = np.eye(4)
        self.D[0][2] = 1
        self.D[1][3] = 1

    def predict(self):
        self.state = self.D * self.state
        self.sig = self.D * self.sig * self.D.T + self.Q

    def correct(self, meas_x, meas_y):
        A = self.M * self.sig * self.M.T + self.R
        K = self.sig * self.M.T * np.linalg.inv(A)
        Y = np.matrix([meas_x, meas_y]).T

        self.state = self.state + K * (Y - self.M * self.state)
        self.sig = self.sig - K*self.M*self.sig

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0, 0], self.state[1, 0]


def adj_crop_from_center(frame, center, shape):
    bound = [[max(0, int(center[0]-shape[1]/2)), min(frame.shape[1], int(center[0]+shape[1]/2))],
             [max(0, int(center[1]-shape[0]/2)), min(frame.shape[0], int(center[1]+shape[0]/2))]]

    if bound[0][0] == 0:
        bound[0][1] = bound[0][0] + shape[1]
    if bound[0][1] == frame.shape[1]:
        bound[0][0] = bound[0][1] - shape[1]

    if bound[1][0] == 0:
        bound[1][1] = bound[1][0] + shape[0]
    if bound[1][1] == frame.shape[0]:
        bound[1][0] = bound[1][1] - shape[0]

    return frame[bound[1][0]:bound[1][1], bound[0][0]:bound[0][1]]

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    return np.sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]))

def show(img_in, img_name='image'):
    cv2.imshow(img_name, img_in)
    cv2.waitKey(0)
    return

class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template = adjust_to_even_dims(self.template)
        self.frame = frame
        self.particles = None  # Initialize your particles array. Read the docstring.
        # self.weights = np.array([1.0]*self.num_particles) / self.num_particles
        self.weights = None
        # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.
        self.state = [self.template_rect['x'] + self.template_rect['w'] // 2,
                      self.template_rect['y'] + self.template_rect['h'] // 2]

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        # show(frame_cutout)
        return np.sum((template.astype(float) - frame_cutout.astype(float)) ** 2) / template.shape[0] / template.shape[1]

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        # print(self.particles, self.weights)
        return np.random.choice(self.num_particles, self.num_particles, p=self.weights, replace=True)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample_x = np.random.normal(self.state[0], self.sigma_dyn, self.num_particles)
        sample_y = np.random.normal(self.state[1], self.sigma_dyn, self.num_particles)
        self.particles = np.array([(a,b) for (a,b) in zip(sample_x, sample_y)])
        self.weights = [0.0] * self.num_particles

        for i in range(0, self.num_particles):
            # if self.particles[i][0] > new_frame.shape[1] - self.template.shape[1] / 8 or \
            #         self.particles[i][1] > new_frame.shape[0] - self.template.shape[0] / 8 or\
            #         self.particles[i][0] < self.template.shape[1] / 8 or \
            #         self.particles[i][1] < self.template.shape[0] / 8:
            #     # print(self.particles[i])
            #     continue

            mse = self.get_error_metric(self.template,
                                        adj_crop_from_center(new_frame, self.particles[i], self.template.shape))

            self.weights[i] = np.exp(-mse/(2.0*self.sigma_exp*self.sigma_exp))
            # print(self.weights[i])
        self.weights /= np.sum(self.weights)
        self.particles = self.particles[self.resample_particles()]
        self.state = np.average(self.particles, axis=0, weights=self.weights)
        # print(self.state)

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        # print((self.state[0] - self.template.shape[0] / 2, self.state[1] - self.template.shape[1] / 2),
        #               (self.state[0] + self.template.shape[0] / 2, self.state[1] + self.template.shape[1] / 2))
        cv2.rectangle(frame_in,
                      (int(self.state[0] - self.template.shape[1] / 2), int(self.state[1] - self.template.shape[0] / 2)),
                      (int(self.state[0] + self.template.shape[1] / 2), int(self.state[1] + self.template.shape[0] / 2)),
                      (128, 128, 128), 2)
        avg_dist = 0
        for a, b in zip(self.particles.astype(int), self.weights):
            cv2.circle(frame_in, tuple(a), 2, (0, 255, 0), 1)
            avg_dist += euclidean_distance(a, self.state) * b

        cv2.circle(frame_in, tuple(self.state.astype(int)), int(avg_dist), (255, 255, 255), 2)

        # show(frame_in)

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.prev_particles = None
        self.prev_weights = None

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_template = self.alpha * adj_crop_from_center(new_frame, self.state, self.template.shape) +\
                   (1 - self.alpha) * self.template
        self.template = new_template.astype(np.uint8)

        if self.prev_particles is not None:
            self.particles = np.concatenate((self.particles, self.prev_particles))
            self.weights = np.concatenate((self.alpha * self.weights, (1 - self.alpha) * self.prev_weights))
            resample_inds = np.random.choice(len(self.weights), self.num_particles, p=self.weights, replace=True)
            self.particles = self.particles[resample_inds]
            self.weights = self.weights[resample_inds]
            self.weights /= np.sum(self.weights)

        self.prev_particles = self.particles
        self.prev_weights = self.weights
        self.state = np.average(self.particles, axis=0, weights=self.weights)

        # print(self.state)


def adjust_to_even_dims(template):
    return template[0:template.shape[0] // 2 * 2, 0:template.shape[1] // 2 * 2]


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.alpha = 0.5

        self.blocked_frames = 0

        self.frame = 0

        self.original_size = self.template.shape[0]*self.template.shape[1]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_mse = 10000000
        best_scale = 1.0
        for i in range(0,20):
            scale = 0.82 + 0.01 * i
            scaled_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
            scaled_template = adjust_to_even_dims(scaled_template)
            # show(scaled_template)

            mse = self.get_error_metric(scaled_template,
                                  adj_crop_from_center(new_frame, self.state, scaled_template.shape))
            if mse < min_mse:
                min_mse = mse
                best_scale = scale
            # show(scaled_template)
            # print(mse)
        # print(min_mse, (1600 - self.frame*2 + 100 * self.blocked_frames))

        # if self.frame > 200:
        #     show(scaled_template)

        if min_mse < (1600 - self.frame*2 + 100 * self.blocked_frames):
            self.template = cv2.resize(self.template, (0, 0), fx=best_scale, fy=best_scale)

            self.template = adjust_to_even_dims(self.template)
            super(MDParticleFilter, self).process(frame)
            self.blocked_frames = 0
        else:
            self.blocked_frames += 1

        self.frame += 1