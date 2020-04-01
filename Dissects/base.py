import numpy as np


class CriticalPoint():

    def __init__(self, typ, pos, val, pair, boundary, destCritId, filId):
        self.typ = typ
        self.pos = pos
        self.val = val
        self.pair = pair
        self.boundary = boundary
        self.destCritId = destCritId
        self.filId = filId

    @property
    def nfil(self):
        return len(self.filId)

    def unconnect_fil(self, filidx):
        del self.filId[filidx]
        del self.destCritId[filidx]

    def convert_distance(self, convert):
        self.pos = convert(self.pos)


class Filament():

    def __init__(self, cp1, cp2, points):
        #
        self._cp1 = cp1
        self._cp2 = cp2
        self._points = points

    @property
    def cp1(self):
        return self._cp1

    @property
    def cp2(self):
        return self._cp2

    @property
    def points(self):
        return self._points

    @property
    def nsamp(self):
        return self._points.shape[0]

    def mid_segments(self):
        return (np.roll(self._points, 1, axis=0) + self._points)[1:] / 2.

    @property
    def segments_len(self):
        try:
            return self._segs_len
        except AttributeError:
            d = (np.roll(self._points, 1, axis=0) - self._points)[1:]
            self._segs_len = np.sqrt(np.sum(d**2, axis=1))
            return self._segs_len

    @property
    def len(self):
        try:
            return self._len
        except AttributeError:
            self._len = np.sum(self.segments_len)
            return self._len

    @property
    def segments_cumlen_from_cp1(self):
        try:
            return self._segs_clen_cp1
        except AttributeError:
            self._segs_clen_cp1 = self._compute_segments_cumlen()
            return self._segs_clen_cp1

    @property
    def segments_cumlen_from_cp2(self):
        try:
            return self._segs_clen_cp2
        except AttributeError:
            self._segs_clen_cp2 = self._compute_segments_cumlen(reverse=True)
            return self._segs_clen_cp2

    def _compute_segments_cumlen(self, reverse=False):
        """distance from cp1 to each mid-segments
                    from cp2 if reverse is True
        """
        slen = self.segments_len
        if reverse:
            slen = slen[::-1]
            return (np.cumsum(slen) - slen / 2.)[::-1]
        else:
            return np.cumsum(slen) - slen / 2.

    def convert_distance(self, convert):
        self._points = convert(self._points)


