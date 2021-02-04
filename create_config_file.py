import configparser

config = configparser.ConfigParser()

# will be set tu running time
config['Run id'] = {'datetime': ''}

config['Edge Models'] = {'n_epochs': 300}

# if models_out_path = '' => no models saved
config['Save Edge Models'] = {
    'models_out_path': '/data/multi-domain-graph/models/',
    'epochs_distance': 50
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
