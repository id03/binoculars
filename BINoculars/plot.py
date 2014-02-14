import numpy
import matplotlib.colors, matplotlib.cm
import mpl_toolkits.mplot3d

# Adapted from http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
# which in turn is based on an example from http://matplotlib.org/users/event_handling.html
class DraggableColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(matplotlib.cm) if hasattr(getattr(matplotlib.cm,i),'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)
        self.canvas = self.cbar.patch.figure.canvas

    def connect(self):
        self.cidpress = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkeypress = self.canvas.mpl_connect('key_press_event', self.key_press)

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)
        self.canvas.mpl_disconnect(self.cidkeypress)

    def on_press(self, event):
        if event.inaxes == self.cbar.ax:
            self.press = event.x, event.y

    def key_press(self, event):
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1
        if self.index<0:
            self.index = len(self.cycle)
        elif self.index>=len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.mappable.set_cmap(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.cbar.ax:
            return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x, event.y

        if isinstance(self.cbar.norm, matplotlib.colors.LogNorm):
            scale = 0.999 * numpy.log10(self.cbar.norm.vmax / self.cbar.norm.vmin)
            if event.button==1:
                self.cbar.norm.vmin *= scale**numpy.sign(dy)
                self.cbar.norm.vmax *= scale**numpy.sign(dy)
            elif event.button==3:
                self.cbar.norm.vmin *= scale**numpy.sign(dy)
                self.cbar.norm.vmax /= scale**numpy.sign(dy)
        else:
            scale = 0.03 * (self.cbar.norm.vmax - self.cbar.norm.vmin)
            if event.button==1:
                self.cbar.norm.vmin -= scale*numpy.sign(dy)
                self.cbar.norm.vmax -= scale*numpy.sign(dy)
            elif event.button==3:
                self.cbar.norm.vmin -= scale*numpy.sign(dy)
                self.cbar.norm.vmax += scale*numpy.sign(dy)

        self.mappable.set_norm(self.cbar.norm)
        self.canvas.draw()

    def on_release(self, event):
        # force redraw on mouse release
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.canvas.draw()


def get_clipped_norm(data, clipping=0.0, log=True):
    if hasattr(data, 'compressed'):
        data = data.compressed()

    if log:
        data = data[data > 0]

    if clipping:
        chop = int(round(data.size * clipping))
        clip = sorted(data)[chop:-(1+chop)]
        vmin, vmax = clip[0], clip[-1]
    else:
        vmin, vmax = data.min(), data.max()

    if log:
        return matplotlib.colors.LogNorm(vmin, vmax)
    else:
        return matplotlib.colors.Normalize(vmin, vmax)


def plot(space, fig, ax, log=True, clipping=0.0, fit=None, **plotopts):
    if space.dimension == 1:
        data = space.get_masked()
        xrange = numpy.ma.array(space.axes[0][:], mask=data.mask)
        if fit is not None:   
            if log:
                ax.semilogy(xrange, data, 'wo', **plotopts)
                ax.semilogy(xrange, fit, 'r', linewidth=2, **plotopts)
            else:
                ax.plot(xrange, data, 'wo', **plotopts)
                ax.plot(xrange, fit, 'r', linewidth=2, **plotopts)
        else:
            if log:
                ax.semilogy(xrange, data, **plotopts)
            else:
                ax.plot(xrange, data, **plotopts)
        
        ax.set_xlabel(space.axes[0].label)
        ax.set_ylabel('Intensity (a.u.)')

    elif space.dimension == 2:
        data = space.get_masked()

        # 2D IMSHOW PLOT
        xmin = space.axes[0].min
        xmax = space.axes[0].max
        ymin = space.axes[1].min
        ymax = space.axes[1].max
        
        norm = get_clipped_norm(data, clipping, log)

        if fit is not None:
            im = ax.imshow(fit.transpose(), origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto', norm=norm, **plotopts)
        else:
            im = ax.imshow(data.transpose(), origin='lower', extent=(xmin, xmax, ymin, ymax), aspect='auto', norm=norm, **plotopts)

        ax.set_xlabel(space.axes[0].label)
        ax.set_ylabel(space.axes[1].label)
        cbarwidget = fig.colorbar(im)
        fig._draggablecbar = DraggableColorbar(cbarwidget, im) # we need to store this instance somewhere
        fig._draggablecbar.connect()
    
    elif space.dimension == 3:
        if not isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
            raise ValueError("For 3D plots, the 'ax' parameter must be an Axes3D instance (use for example gca(projection='3d') to get one)")

        cmap = getattr(matplotlib.cm, plotopts.pop('cmap', 'jet'))
        norm = get_clipped_norm(space.get_masked(), clipping, log)

        gridx, gridy, gridz = space.get_grid()
        ax.plot_surface(gridx[0,:,:], gridy[0,:,:], gridz[0,:,:],  facecolors=cmap(norm(space.project(0).get_masked())), shade=False, cstride=1, rstride=1)
        ax.plot_surface(gridx[:,-1,:], gridy[:,-1,:], gridz[:,-1,:], facecolors=cmap(norm(space.project(1).get_masked())), shade=False, cstride=1, rstride=1)
        ax.plot_surface(gridx[:,:,0], gridy[:,:,0], gridz[:,:,0],  facecolors=cmap(norm(space.project(2).get_masked())), shade=False, cstride=1, rstride=1)

        ax.set_xlabel(space.axes[0].label)
        ax.set_ylabel(space.axes[1].label)
        ax.set_zlabel(space.axes[2].label)

    elif space.dimension > 3:
        raise ValueError("Cannot plot 4 or higher dimensional spaces, use projections or slices to decrease dimensionality.")
