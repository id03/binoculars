import matplotlib.colors

def plot(space, fig, ax, log=True, clipping=0.0):
    if space.dimension == 1:
        data = space.get_masked()
        xrange = space.axes[0][:]
        if log:
            ax.semilogy(xrange, data)
        else:
            ax.plot(xrange, data)
        
        ax.set_xlabel(space.axes[0].label)
        ax.set_ylabel('Intensity (a.u.)')

    elif space.dimension == 2:
        data = space.get_masked()

        # COLOR CLIPPING
        colordata = data.compressed()
        if log:
            colordata = colordata[colordata > 0]

        if clipping:
            chop = int(round(colordata.size * clipping))
            clip = sorted(colordata)[chop:-(1+chop)]
            vmin, vmax = clip[0], clip[-1]
            del clip
        else:
            vmin, vmax = colordata.min(),colordata.max()
        del colordata

        # 2D IMSHOW PLOT
        data = space.get_masked()

        xmin = space.axes[0].min
        xmax = space.axes[0].max
        ymin = space.axes[1].min
        ymax = space.axes[1].max
        
        if log:
            norm = matplotlib.colors.LogNorm(vmin, vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin, vmax)

        im = ax.imshow(data.transpose(), origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto', norm=norm)

        ax.set_xlabel(space.axes[0].label)
        ax.set_ylabel(space.axes[1].label)
        fig.colorbar(im)

    elif space.dimension > 2:
        raise ValueError("Cannot plot 3 or higher dimensional spaces, use projections or slices to decrease dimensionality.")
