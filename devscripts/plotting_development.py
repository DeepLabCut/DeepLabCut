import deeplabcut

CONFIG_PATH = '/Volumes/GoogleDrive/My Drive/DLC_test/testproject-jkoppanyi-2022-02-15/config.yaml'
VIDEO_1 = '/Volumes/GoogleDrive/My Drive/DLC_test/testproject-jkoppanyi-2022-02-15/videos/BellaRoseRotterdamGPS-2ndPiaf.mp4'


def main():

    deeplabcut.plot_trajectories(CONFIG_PATH, videos=[VIDEO_1], videotype='mp4', showfigures=True)


if __name__ == '__main__':
    main()
