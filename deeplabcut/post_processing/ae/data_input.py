import numpy as np
import random
import copy
import math


factor = 10  # arbitrary constant for ensuring data isn't too large


class Data:
    def __init__(self, points, p, p_bound, zero_joint=0):
        ''' Data object for creating data for autoencoder
        _________
        points: Numpy float array
          pose data across video
        p: Numpy float array
          confidence outputs for poses in points
        p_bound: float
          cutoff for "missing data"
        zero_joint: int
          translate poses so this joint is (0,0)
        '''
        Data = points
        lenData = len(Data)

        self.filtered_trainSegments = self._sanitize_data(Data, p, p_bound, zero_joint)

        self.all_data = Data
        self.std = 0.75 * np.std(np.concatenate(self.filtered_trainSegments))
        self.factor = factor
        self.time_pair = 5  # concatenate 5 frames
        self.inter_pair = 3  # number of intermediary frames
        self.zero_joint = zero_joint

    def _sanitize_data(self, data, p, p_bound, zero_joint):
        ''' Break data at poses with confidence level below pbound '''
        return_array = []

        is_sanitary = p > p_bound
        is_sanitary = np.all(is_sanitary, axis=1)
        slice_indexes = np.where(is_sanitary == False)
        slice_indexes = np.insert(slice_indexes, 0, -1)
        slice_indexes = np.append(slice_indexes, len(data))

        input_size = len(data[0])

        # Zero out the data
        temp_data = copy.deepcopy(data)

        for i in range(len(data[0])):
            if i % 2 == 0:
                data[:, i] = data[:, i] - temp_data[:, zero_joint]
            else:
                data[:, i] = data[:, i] - temp_data[:, zero_joint + 1]

        for index in range(len(slice_indexes) - 1):
            currappend = data[slice_indexes[index] + 1:slice_indexes[index + 1]]
            if len(currappend) < 5:
                continue
            else:
                return_array.append(currappend)
        return return_array

    def reconstruct_original_data(self, vae_return):
        vae_return = (vae_return + self.npmean) / self.npfactor

        # Unzero out the data
        temp_data = copy.deepcopy(vae_return)
        originalLength = len(self.all_data)
        for i in range(len(vae_return[0])):
            if i % 2 == 0:
                vae_return[:, i] = temp_data[:, i] + self.all_data[int((self.time_pair - 1) / 2):originalLength - int(
                    (self.time_pair - 1) / (2)), self.zero_joint]
            else:
                vae_return[:, i] = temp_data[:, i] + self.all_data[int((self.time_pair - 1) / 2):originalLength - int(
                    (self.time_pair - 1) / (2)), self.zero_joint + 1]
        vae_return = np.vstack(
            [self.all_data[:int((self.time_pair - 1) / 2)], vae_return, self.all_data[-int((self.time_pair - 1) / 2):]])
        return vae_return

    def _pair_time_data(self, data, num_times):
        removed_versions = [data[i:] for i in range(num_times)]
        official_versions = [removed_versions[i][:len(removed_versions[i]) + (-1 * num_times) + 1 + i] for i in
                             range(num_times)]
        return np.concatenate(official_versions, axis=1)

    def _get_nn_training(self, data):
        original_data = copy.deepcopy(data)
        originalLength = len(data)
        input_size = len(data[0])

        three_versions = [data, 0.75 * data, 1.25 * data]

        new_noisy_data = []
        new_inter_data = []
        new_denoised_data = []

        for m in range(len(three_versions)):
            currdata = copy.deepcopy(three_versions[m])
            currdata_inter = self._pair_time_data(currdata, self.inter_pair)[1:-1]
            currdata_parallel = self._pair_time_data(currdata, self.time_pair)

            new_noisy_data.append(currdata_parallel)
            new_inter_data.append(currdata_inter)
            new_denoised_data.append(
                currdata[int((self.time_pair - 1) / 2):originalLength - int((self.time_pair - 1) / 2)])

            # AR(1) added in
            std = self.std
            noiseadd = [[0 for z in range(len(currdata_parallel[0]))] for l in range(len(currdata_parallel))]
            for r in range(len(currdata_parallel)):
                possibleChoices = [q for q in range(int(input_size / 2))]
                numberOfLimbs = random.randint(0, int(input_size / 2) - 1)

                rand_items = random.sample(possibleChoices, numberOfLimbs)
                for z in range(int(input_size / 2)):
                    if z in rand_items:
                        prev_dist = 0
                        prev_angle = 0
                        for m in range(self.time_pair):
                            prev_dist = (0.8 * prev_dist) + np.random.normal(0, std)
                            prev_angle = (0.8 * prev_angle) + random.uniform(-math.pi / 2, math.pi / 2)
                            distx = math.cos(prev_angle) * prev_dist
                            disty = math.sin(prev_angle) * prev_dist
                            noiseadd[r][2 * z + input_size * m] = distx
                            noiseadd[r][2 * z + 1 + input_size * m] = disty

            new_noisy_data.append(currdata_parallel + noiseadd)
            new_inter_data.append(currdata_inter)
            new_denoised_data.append(
                currdata[int((self.time_pair - 1) / 2):originalLength - int((self.time_pair - 1) / (2))])

            # Gaussian Noise added in
            std = self.std
            noiseadd = [[0 for z in range(len(currdata[0]))] for l in range(len(currdata))]
            for r in range(len(currdata)):
                possibleChoices = [q for q in range(int(input_size / 2))]
                numberOfLimbs = random.randint(0, int(input_size / 2))

                rand_items = random.sample(possibleChoices, numberOfLimbs)
                for z in range(int(input_size / 2)):
                    if z in rand_items:
                        dist = np.random.normal(0, std)
                        angle = random.uniform(-math.pi / 2, math.pi / 2)
                        distx = math.cos(angle) * dist
                        disty = math.sin(angle) * dist
                        noiseadd[r][2 * z] = distx
                        noiseadd[r][2 * z + 1] = disty

            new_noisy_data.append(self._pair_time_data(currdata + noiseadd, self.time_pair))
            new_inter_data.append(currdata_inter)
            new_denoised_data.append(
                currdata[int((self.time_pair - 1) / 2):originalLength - int((self.time_pair - 1) / (2))])

            # Occlusions added in
            std = 10 * self.std
            noiseadd = [[0 for z in range(len(currdata_parallel[0]))] for l in range(len(currdata_parallel))]
            for r in range(len(currdata_parallel)):
                possibleChoices = [q for q in range(int(input_size / 2))]
                numberOfLimbs = random.randint(0, int(input_size / 4))

                rand_items = random.sample(possibleChoices, numberOfLimbs)
                for z in range(int(input_size / 2)):
                    if z in rand_items:
                        dist = np.random.normal(0, std)
                        angle = random.uniform(-math.pi / 2, math.pi / 2)
                        distx = math.cos(angle) * dist
                        disty = math.sin(angle) * dist
                        for m in range(self.time_pair):
                            noiseadd[r][2 * z + input_size * m] = (-1 * currdata_parallel[r][
                                2 * z + input_size * m]) + distx
                            noiseadd[r][2 * z + 1 + input_size * m] = (-1 * currdata_parallel[r][
                                2 * z + 1 + input_size * m]) + disty

            new_noisy_data.append(currdata_parallel + noiseadd)
            new_inter_data.append(currdata_inter)
            new_denoised_data.append(
                currdata[int((self.time_pair - 1) / 2):originalLength - int((self.time_pair - 1) / (2))])

        new_noisy_data = np.concatenate(new_noisy_data, axis=0)
        new_inter_data = np.concatenate(new_inter_data, axis=0)
        new_denoised_data = np.concatenate(new_denoised_data, axis=0)

        return new_noisy_data, new_inter_data, new_denoised_data

    def get_training_data(self):
        input_size = len(self.filtered_trainSegments[0][0])

        nn_train_noisy, nn_train_inter, nn_train_denoised = [], [], []
        for i in range(len(self.filtered_trainSegments)):
            temp_noisy, temp_inter, temp_denoised = self._get_nn_training(self.filtered_trainSegments[i])
            nn_train_noisy.append(temp_noisy)
            nn_train_inter.append(temp_inter)
            nn_train_denoised.append(temp_denoised)
        nn_train_noisy = np.concatenate(nn_train_noisy, axis=0)
        nn_train_inter = np.concatenate(nn_train_inter, axis=0)
        nn_train_denoised = np.concatenate(nn_train_denoised, axis=0)

        # Apply appropriate conversions to get neural network data
        self.npmean = np.mean(nn_train_denoised, axis=0)
        npmax = np.amax(nn_train_denoised)
        npmin = np.amin(nn_train_denoised)
        nprange = npmax - npmin
        self.npfactor = self.factor / nprange

        repeated_npmean = np.concatenate([self.npmean for i in range(self.time_pair)])
        repeated_npmean2 = np.concatenate([self.npmean for i in range(self.inter_pair)])
        nn_train_noisy = (nn_train_noisy - repeated_npmean) * self.npfactor
        nn_train_inter = (nn_train_inter - repeated_npmean2) * self.npfactor
        nn_train_denoised = (nn_train_denoised - self.npmean) * self.npfactor

        return nn_train_noisy, nn_train_inter, nn_train_denoised, self.npfactor

    def get_all_original_data(self):
        input_size = len(self.all_data[0])

        currdata = copy.deepcopy(self.all_data)
        originalLength = len(currdata)
        original_parallel = self._pair_time_data(currdata, self.time_pair)
        original_single = currdata[int((self.time_pair - 1) / 2):originalLength - int((self.time_pair - 1) / 2)]

        repeated_npmean = np.concatenate([self.npmean for i in range(self.time_pair)])
        original_parallel = (original_parallel - repeated_npmean) * self.npfactor
        original_single = (original_single - self.npmean) * self.npfactor

        return original_parallel, original_single
