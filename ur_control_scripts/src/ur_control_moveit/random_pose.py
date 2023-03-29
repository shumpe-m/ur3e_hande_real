class Random_pose():
    def __init__(self):
        # workspace
        self.max_x = 0.2
        self.min_x = -0.21
        self.max_y = 0.45
        self.min_y = 0.25
        self.max_z = 0.2
        self.min_z = 0.01 + 0.02 # 0.02=offset
        # dish radius
        self.dish_radius = 0.09
        # dish1
        self.dish1_center = [0.0925, 0.35, 0.7801]
        # dish2 = bin1
        self.dish2_center = [-0.1075, 0.35, 0.7801]
        # dish3 = bin2
        self.dish3_center = [0.0935, -0.35, 0.7801]
        # dish4 = dish2
        self.dish4_center = [-0.1065, -0.35, 0.7801]


    def random_pose(self, area = "dish1"):
        rand = np.random.rand(5)
        state_msg = [0]*6

        # workspace
        if area == "workspace":
            state_msg[0] = rand[0] * (abs(self.min_x) + abs(self.max_x)) + self.min_x
            state_msg[1] = rand[1] * (abs(self.min_y) + abs(self.max_y)) + self.min_y
        # dish 1
        elif area == "dish1":
            state_msg[0] = np.cos(rand[0] * 2 * np.pi) * (self.dish_radius) + self.dish1_center[0]
            state_msg[1] = np.sin(rand[1] * 2 * np.pi) * (self.dish_radius) + self.dish1_center[1]
        # bin 1
        elif area == "bin1":
            state_msg[0] = np.cos(rand[0] * 2 * np.pi) * (self.dish_radius) + self.dish2_center[0]
            state_msg[1] = np.sin(rand[1] * 2 * np.pi) * (self.dish_radius) + self.dish2_center[1]
        # dish 2
        elif area == "dish2":
            state_msg[0] = np.cos(rand[0] * 2 * np.pi) * (self.dish_radius) + self.dish4_center[0]
            state_msg[1] = np.sin(rand[1] * 2 * np.pi) * (self.dish_radius) + self.dish4_center[1]
        # bin 2
        elif area == "bin2":
            state_msg[0] = np.cos(rand[0] * 2 * np.pi) * (self.dish_radius) + self.dish3_center[0]
            state_msg[1] = np.sin(rand[1] * 2 * np.pi) * (self.dish_radius) + self.dish3_center[1]
        else:
            print("Input Error: 'area'")
        state_msg[2] = 0.9

        e = tf.transformations.quaternion_from_euler((rand[2] - 0.5) * np.pi + np.pi/3, (rand[3] - 0.5) * np.pi + np.pi/3, (rand[4] - 0.5) * np.pi + np.pi/3)
        state_msg[3] = e[0]
        state_msg[4] = e[1]
        state_msg[5] = e[2]


        return state_msg