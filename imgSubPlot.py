def imgSubPlot(image, title):
    import numpy as np
    from matplotlib import pyplot as plt
    import cv2

    """print the images in sub plot"""
    row = np.floor(np.sqrt(len(image)))
    col = np.ceil(len(image) / row)
    fig = plt.figure(figsize=(12, 10))
    for i in range(len(image)):
        ax = fig.add_subplot(row, col, i + 1)
        ax.imshow(cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB))
        ax.set_title(title[i])
        ax.set_axis_off()
    plt.show()

