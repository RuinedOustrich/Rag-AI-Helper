import os
import warnings
from pipeline.agent import Agent

warnings.filterwarnings("ignore")


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'config.yaml')

if __name__ == '__main__':

    agent = Agent(CONFIG_PATH, ROOT_DIR).run()

