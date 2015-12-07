# -*- encoding: utf-8 -*-
'''
 This file is part of the binoculars project.

  The BINoculars library is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  The BINoculars library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the hkl library.  If not, see <http://www.gnu.org/licenses/>.

  Copyright (C)      2015 Synchrotron SOLEIL
                          L'Orme des Merisiers Saint-Aubin
                          BP 48 91192 GIF-sur-YVETTE CEDEX

  Copyright (C) 2012-2015 European Synchrotron Radiation Facility
                          Grenoble, France

  Authors: Willem Onderwaater <onderwaa@esrf.fr>
           Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>

'''
import sys
import os
import itertools
import numpy
import tables
import math

from pyFAI.detectors import ALL_DETECTORS
from math import cos, sin
from collections import namedtuple
from numpy.linalg import inv
from networkx import DiGraph, dijkstra_path

from .. import backend, errors, util


class realspace(backend.ProjectionBase):
    # scalars: mu, theta, [chi, phi, "omitted"] delta, gamR, gamT, ty, wavelength
    # 3x3 matrix: UB
    def project(self, index, dataframe, pixels):
        return pixels[1], pixels[2]
        #return numpy.meshgrid(numpy.arange(pixels[0].shape[1]), numpy.arange(pixels[0].shape[0]))

    def get_axis_labels(self):
        return 'x', 'y'

class Pixels(backend.ProjectionBase):
    # scalars: mu, theta, [chi, phi, "omitted"] delta, gamR, gamT, ty, wavelength
    # 3x3 matrix: UB
    def project(self, index, dataframe, pixels):
        return numpy.meshgrid(numpy.arange(pixels[0].shape[1]), numpy.arange(pixels[0].shape[0]))

    def get_axis_labels(self):
        return 'x', 'y'



class HKLProjection(backend.ProjectionBase):
    # scalars: mu, theta, [chi, phi, "omitted"] delta, gamR, gamT, ty, wavelength
    # 3x3 matrix: UB
    def project(self, index, dataframe, pixels):
        # put the detector at the right position

        UB = dataframe.diffractometer.ub

        s_axes = rotation_axes(dataframe.diffractometer.axes.graph,
                               dataframe.diffractometer.axes.sample)
        d_axes = rotation_axes(dataframe.diffractometer.axes.graph,
                               dataframe.diffractometer.axes.detector)

        # the ki vector should be in the NexusFile or easily extracted
        # from the hkl library.
        ki = [1, 0, 0]
        k = 2 * math.pi / dataframe.source.wavelength
        values = dataframe.mu[index], dataframe.omega[index], dataframe.delta[index], dataframe.gamma[index]
        s_values = values[0], values[1]
        d_values = values[0], values[2], values[3]
        R = reduce(numpy.dot, (zip_with(M, numpy.radians(s_values), s_axes)))
        P = reduce(numpy.dot, (zip_with(M, numpy.radians(d_values), d_axes)))
        RUB_1 = inv(numpy.dot(R, UB))
        RUB_1P = numpy.dot(RUB_1, P)
        # rotate the detector around x of 90 degrees
        RUB_1P = numpy.dot(RUB_1P, M(math.pi/2., [1, 0, 0]))
        kf = normalized(pixels, axis=0)
        hkl_f = numpy.tensordot(RUB_1P, kf, axes=1)
        hkl_i = numpy.dot(RUB_1, ki)
        hkl = hkl_f - hkl_i[:, numpy.newaxis, numpy.newaxis]

        h,k,l = hkl * k

        return (h, k, l)

    def get_axis_labels(self):
        return 'H', 'K', 'L'

class HKProjection(HKLProjection):
    def project(self, index, dataframe, pixels):
        h,k,l = super(HKProjection, self).project(index, dataframe, pixels)
        return h,k

    def get_axis_labels(self):
        return 'H', 'K'



def get_nxclass(hfile, nxclass, path="/"):
    """
    :param hfile: the hdf5 file.
    :type hfile: tables.file.
    :param nxclass: the nxclass to extract
    :type nxclass: str
    """
    for node in hfile.walk_nodes(path):
        try:
            if nxclass == node._v_attrs['NX_class']:
                return node
        except KeyError:
            pass
    return None

Axes = namedtuple("Axes", ["sample", "detector", "graph"])

Rotation = namedtuple("Rotation", ["axis", "value"])


def get_axes(name):
    """
    :param name: the diffractometer name
    :type name: str
    """
    sample = []
    detector = []
    graph = DiGraph()
    if name == 'ZAXIS':
        # axis
        graph.add_node("mu", transformation=Rotation([0, 0, 1], 0))
        graph.add_node("omega", transformation=Rotation([0, -1, 0], 0))
        graph.add_node("delta", transformation=Rotation([0, -1, 0], 0))
        graph.add_node("gamma", transformation=Rotation([0, 0, 1], 0))

        # topology
        graph.add_edges_from([("mu", "omega"),
                              ("mu", "delta"), ("delta", "gamma")])

        sample = dijkstra_path(graph, "mu", "omega")
        detector = dijkstra_path(graph, "mu", "gamma")

    return Axes(sample, detector, graph)


Diffractometer = namedtuple('Diffractometer',
                            ['name',  # name of the hkl diffractometer
                             'ub',  # the UB matrix
                             'axes'])  # the Axes namedtuple


def get_diffractometer(hfile):
    """ Construct a Diffractometer from a NeXus file """
    node = get_nxclass(hfile, 'NXdiffractometer')

    name = node.type[0][:-1]
    ub = node.UB[:]
    axes = get_axes(name)

    return Diffractometer(name, ub, axes)


Sample = namedtuple("Sample", ["a", "b", "c",
                               "alpha", "beta", "gamma",
                               "ux", "uy", "uz", "graph"])


def get_sample(hfile):
    graph = DiGraph()
    graph.add_node("ux", transformation=Rotation([1, 0, 0], 0))
    graph.add_node("uy", transformation=Rotation([0, 1, 0], 0))
    graph.add_node("uz", transformation=Rotation([0, 0, 1], 0))
    graph.add_edges_from([("ux", "uy"),
                          ("uy", "uz")])

    return Sample(1.54, 1.54, 1.54, 90, 90, 90, 0, 0, 0, graph)


Detector = namedtuple("Detector", ["name"])


def get_detector(hfile):
    return Detector("imxpads140")

Source = namedtuple("Source", ["wavelength"])


def get_source(hfile):
    wavelength = get_nxclass(hfile, 'NXmonochromator').wavelength[0]
    return Source(wavelength)


DataFrame = namedtuple("DataFrame", ["diffractometer",
                                     "sample", "detector", "source",
                                     "mu", "omega", "delta", "gamma",
                                     "image", "graph"])


def dataframes(hfile, data_path=None):
    diffractometer = get_diffractometer(hfile)
    sample = get_sample(hfile)
    detector = get_detector(hfile)
    source = get_source(hfile)
    graph = DiGraph()
    # this should be generalized
    for g in [diffractometer.axes.graph, sample.graph]:
        graph.add_nodes_from(g)
        graph.add_edges_from(g.edges())
    # connect the sample with the right axis.
    graph.add_edge("omega", "ux")

    for group in hfile.get_node('/'):
        scan_data = group._f_get_child("scan_data")
        dataframe = {
            "diffractometer": diffractometer,
            "sample": sample,
            "detector": detector,
            "source": source,
            "graph": graph,
        }

        # now instantiate the pytables objects
        for key, value in data_path.iteritems():
            child = scan_data._f_get_child(value)
            dataframe[key] = child

        yield DataFrame(**dataframe)


def M(theta, u):
    """
    :param theta: the axis value in radian
    :type theta: float
    :param u: the axis vector [x, y, z]
    :type u: [float, float, float]
    :return: the rotation matrix
    :rtype: numpy.ndarray (3, 3)
    """
    c = cos(theta)
    one_minus_c = 1 - c
    s = sin(theta)
    return numpy.array([[c + u[0]**2 * one_minus_c,
                         u[0] * u[1] * one_minus_c - u[2] * s,
                         u[0] * u[2] * one_minus_c + u[1] * s],
                        [u[0] * u[1] * one_minus_c + u[2] * s,
                         c + u[1]**2 * one_minus_c,
                         u[1] * u[2] * one_minus_c - u[0] * s],
                        [u[0] * u[2] * one_minus_c - u[1] * s,
                         u[1] * u[2] * one_minus_c + u[0] * s,
                         c + u[2]**2 * one_minus_c]])


def rotation_axes(graph, nodes):
    """
    :param graph: descrition of the diffractometer geometry
    :type graph: DiGraph
    :param nodes: list of the nodes to use
    :type nodes: list(str)
    :return: the list of the rotation axes expected by zip_with
    """
    return [graph.node[idx]["transformation"].axis for idx in nodes]


def zip_with(f, *coll):
    return itertools.starmap(f, itertools.izip(*coll))


def rotation_matrix(values, axes):
    """
    :param values: the rotation axes values in radian
    :type values: list(float)
    :param axes: the rotation axes
    :type axes: list of [x, y, z]
    :return: the rotation matrix
    :rtype: numpy.ndarray (3, 3)
    """
    return reduce(numpy.dot, (zip_with(M, values, axes)))

def get_ki(wavelength):
    """
    for now the direction is always along x
    """
    TAU = 2 * math.pi
    return numpy.array([TAU / wavelength, 0, 0])

def normalized(a, axis=-1, order=2):
    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / numpy.expand_dims(l2, axis)

class SIXS(backend.InputBase):
    # OFFICIAL API

    dbg_scanno = None
    dbg_pointno = None

    def generate_jobs(self, command):
        scans = util.parse_multi_range(','.join(command).replace(' ', ','))
        if not len(scans):
            sys.stderr.write('error: no scans selected, nothing to do\n')
        for scanno in scans:
            util.status('processing scan {0}...'.format(scanno))
            if self.config.pr:
                pointcount = self.config.pr[1] - self.config.pr[0] + 1
                start = self.config.pr[0]
            else:
                # just open the file in order to extract the number of step.
                with tables.open_file(self.get_filename(scanno), 'r') as scan:
                    start = 0
                    pointcount = get_nxclass(scan, "NXdata").UHV_MU.shape[0]
            if pointcount > self.config.target_weight * 1.4:
                for s in util.chunk_slicer(pointcount, self.config.target_weight):
                    yield backend.Job(scan=scanno, firstpoint=start+s.start, lastpoint=start+s.stop-1, weight=s.stop-s.start)
            else:
                yield backend.Job(scan=scanno, firstpoint=start, lastpoint=start+pointcount-1, weight=pointcount)

    def process_job(self, job):
        super(SIXS, self).process_job(job)
        with tables.open_file(self.get_filename(job.scan), 'r') as scan:
            self.metadict = dict()
            try:
                for dataframe in dataframes(scan, self.HPATH):
                    pixels = self.get_pixels(dataframe.detector)
                    for index in range(job.firstpoint, job.lastpoint + 1):
                        yield self.process_image(index, dataframe, pixels)
                util.statuseol()
            except Exception as exc:
                exc.args = errors.addmessage(exc.args, ', An error occured for scan {0} at point {1}. See above for more information'.format(self.dbg_scanno, self.dbg_pointno))
                raise
            self.metadata.add_section('sixs_backend', self.metadict)

    def parse_config(self, config):
        super(SIXS, self).parse_config(config)
        self.config.xmask = util.parse_multi_range(config.pop('xmask', None))#Optional, select a subset of the image range in the x direction. all by default
        self.config.ymask = util.parse_multi_range(config.pop('ymask', None))#Optional, select a subset of the image range in the y direction. all by default
        self.config.nexusfile = config.pop('nexusfile')#Location of the specfile
        self.config.pr = config.pop('pr', None) #Optional, all range by default
        if self.config.xmask is None:
            self.config.xmask = slice(None)
        if self.config.ymask is None:
            self.config.ymask = slice(None)
        if self.config.pr:
            self.config.pr = util.parse_tuple(self.config.pr, length=2, type=int)
        self.config.sdd = float(config.pop('sdd'))# sample to detector distance (mm)
        self.config.centralpixel = util.parse_tuple(config.pop('centralpixel'), length=2, type=int)  #x,y
        self.config.maskmatrix = config.pop('maskmatrix', None)#Optional, if supplied pixels where the mask is 0 will be removed
    def get_destination_options(self, command):
        if not command:
            return False
        command = ','.join(command).replace(' ', ',')
        scans = util.parse_multi_range(command)
        return dict(first=min(scans), last=max(scans), range=','.join(str(scan) for scan in scans))

    # CONVENIENCE FUNCTIONS
    def get_filename(self, scanno):
        filename = self.config.nexusfile.format(scanno = str(scanno).zfill(5))
        if not os.path.exists(filename):
	    raise errors.ConfigError('nexus filename does not exist: {0}'.format(filename))
	return filename


    @staticmethod
    def apply_mask(data, xmask, ymask):
        roi = data[ymask, :]
        return roi[:, xmask]

class FlyScanUHV(SIXS):
    HPATH = {
        "image": "xpad_image",
        "mu": "UHV_MU",
        "omega": "UHV_OMEGA",
        "delta": "UHV_DELTA",
        "gamma": "UHV_GAMMA",
    }

    def process_image(self, index, dataframe, pixels):
        util.status(str(index))
        detector = ALL_DETECTORS[dataframe.detector.name]()
        maskmatrix = load_matrix(self.config.maskmatrix)
        if maskmatrix is not None:
            mask = numpy.bitwise_or(detector.mask, maskmatrix)
        else:
            mask = detector.mask

        intensity = dataframe.image[index, ...]
        weights = numpy.ones_like(intensity)
        weights *= ~mask 
        #util.status('{4}| gamma: {0}, delta: {1}, theta: {2}, mu: {3}'.format(gamma, delta, theta, mu, time.ctime(time.time())))

        return intensity, weights, (index, dataframe, pixels)

    def get_pixels(self, detector):
        detector = ALL_DETECTORS[detector.name]()
        y, x, _ = detector.calc_cartesian_positions()
        y0 = y[self.config.centralpixel[1], self.config.centralpixel[0]]
        x0 = x[self.config.centralpixel[1], self.config.centralpixel[0]]
        z = numpy.ones(x.shape) * -1 * self.config.sdd
        # return converted to the hkl library coordinates
        # x -> -y
        # y -> z
        # z -> -x
        return numpy.array([-z, -(x - x0), (y - y0)])


def load_matrix(filename):
    if filename == None:
        return None
    if os.path.exists(filename):
        ext = os.path.splitext(filename)[-1]
        if ext == '.txt':
            return numpy.array(numpy.loadtxt(filename), dtype = numpy.bool)
        elif ext == '.npy':
            return numpy.array(numpy.load(filename), dtype = numpy.bool)
        else:
            raise ValueError('unknown extension {0}, unable to load matrix!\n'.format(ext))
    else:
       raise IOError('filename: {0} does not exist. Can not load matrix'.format(filename))
