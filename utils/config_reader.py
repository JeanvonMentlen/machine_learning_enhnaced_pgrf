import configobj


class ConfigLoader:
    """Class for loading settings from a .config file."""

    def __init__(self, config_file_name):
        self.config_object = configobj.ConfigObj(config_file_name)

    def get_real_data_file_name(self):
        real_data_file_name = self.config_object['training']['real_data_file_name']

        return real_data_file_name

    def get_file_names(self):
        training_data_file_name = self.config_object['training']['training_data_file_name']
        training_labels_file_name = self.config_object['training']['training_labels_file_name']

        validation_data_file_name = self.config_object['training']['validation_data_file_name']
        validation_labels_file_name = self.config_object['training']['validation_labels_file_name']

        test_data_file_name = self.config_object['training']['test_data_file_name']
        test_labels_file_name = self.config_object['training']['test_labels_file_name']

        return training_data_file_name, training_labels_file_name, validation_data_file_name, \
               validation_labels_file_name, test_data_file_name, test_labels_file_name

    def get_generation_file_names(self):
        generation_data_file_name = self.config_object['training']['generation_data_file_name']
        generation_labels_file_name = self.config_object['training']['generation_labels_file_name']

        return generation_data_file_name, generation_labels_file_name

    def get_random_seed(self):
        random_seed = int(self.config_object['training']['random_seed'])

        return random_seed

    def get_number_of_training_samples(self):
        number_of_training_samples = int(self.config_object['training']['number_of_training_samples'])

        return number_of_training_samples

    def get_data_path(self):
        return self.config_object['training']['data_path']

    def get_model_name(self):
        model_name = self.config_object['training']['model_name']

        return model_name

    def get_training_session_name(self):
        ts_name = self.config_object['training']['training_session']

        return ts_name

    def get_number_of_epochs(self):
        number_of_epochs = int(self.config_object['training']['number_of_epochs'])

        return number_of_epochs

    def get_batch_sizes(self):
        train_bs = int(self.config_object['training']['train_batch_size'])
        validation_bs = int(self.config_object['training']['validation_batch_size'])
        test_bs = int(self.config_object['training']['test_batch_size'])

        return train_bs, validation_bs, test_bs

    def get_parameters(self):
        parameters = {}
        parameters['b_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['b_min'])
        parameters['b_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['b_max'])

        parameters['beta_angle_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['beta_angle_min'])
        parameters['beta_angle_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['beta_angle_max'])

        parameters['int_const_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['int_const_min'])
        parameters['int_const_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['int_const_max'])

        parameters['l_y_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['l_y_min'])
        parameters['l_y_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['l_y_max'])

        parameters['l_z_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['l_z_min'])
        parameters['l_z_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['l_z_max'])

        parameters['phi_A_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['phi_A_min'])
        parameters['phi_A_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['phi_A_max'])

        parameters['porod_const_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['porod_const_min'])
        parameters['porod_const_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['porod_const_max'])

        parameters['vol_factor_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['vol_factor_min'])
        parameters['vol_factor_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['vol_factor_max'])

        parameters['d_z_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['d_z_min'])
        parameters['d_z_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['d_z_max'])

        parameters['d_y_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['d_y_min'])
        parameters['d_y_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['d_y_max'])

        parameters['rho_A_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['rho_A_min'])
        parameters['rho_A_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['rho_A_max'])

        parameters['rho_B_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['rho_B_min'])
        parameters['rho_B_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['rho_B_max'])

        parameters['rho_S_min'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['rho_S_min'])
        parameters['rho_S_max'] = self._convert_to_list_of_floats(self.config_object['PGRF-parameters']['rho_S_max'])

        return parameters

    @staticmethod
    def _convert_to_list_of_floats(list_of_strings):
        if type(list_of_strings) == str:
            return [float(list_of_strings)]
        else:
            list_of_floats = [float(item) for item in list_of_strings]

        return list_of_floats
